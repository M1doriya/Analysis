import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st


# Streamlit wrapper only.
# - Upload multiple JSON statements
# - Detect/override company name accurately (tuned for your statement structure)
# - Patch globals in bank_analysis_v5_2_1.py and call analyze()
# - Post-process/enrich results to fill empty HTML categories:
#     - counterparties.top_payers/top_payees
#     - flags.round_figure_transactions.top_10_transactions/all_transactions
#     - bank and counterparty fields for top transactions
# - DOES NOT change core engine logic


@dataclass
class AccountConfig:
    account_id: str
    bank_name: str
    account_number: str
    account_type: str
    classification: str


# -------------------------
# Basic utilities
# -------------------------

def _sanitize_company_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _generate_company_keywords(company_name: str) -> List[str]:
    """
    Generate robust partial-match keywords to reduce false negatives.
    """
    name = _sanitize_company_name(company_name).upper()
    if not name:
        return []

    cleaned = re.sub(
        r"\b(SDN\.?|BHD\.?|BERHAD|SENDIRIAN|LIMITED|LTD\.?|CO\.?|COMPANY)\b",
        "",
        name,
    )
    cleaned = re.sub(r"[^A-Z0-9& ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 2]

    kws: List[str] = []
    kws.append(name)
    if words:
        kws.append(words[0])
    if len(words) >= 2:
        kws.append(" ".join(words[:2]))
    if len(words) >= 3:
        kws.append(" ".join(words[:3]))
    if words and len(words[0]) >= 8:
        kws.append(words[0][:8])

    seen = set()
    out: List[str] = []
    for k in kws:
        k2 = k.strip().upper()
        if k2 and k2 not in seen:
            seen.add(k2)
            out.append(k2)
    return out


def _safe_json_load(uploaded_file) -> dict:
    try:
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception:
        return json.loads(uploaded_file.getvalue())


def _write_temp_json(payload: dict, suffix: str = ".json") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p)


def _import_engine():
    import importlib
    return importlib.import_module("bank_analysis_v5_2_1")


# -------------------------
# Company detection (tuned for your dataset)
# -------------------------

def _extract_candidate_company_names(statement_json: dict, filename: str = "") -> List[str]:
    """
    Robust extraction for your statement JSONs:
      {summary, monthly_summary, transactions}
    There is typically no explicit account-holder field, so we mine:
      - filename hints (e.g., contains "MTC")
      - transactions[].description patterns (including 'SDN.' without 'BHD')
    """
    candidates: List[str] = []

    fn = (filename or "").upper()
    if "MTC" in fn:
        candidates.append("MTC ENGINEERING SDN BHD")
        candidates.append("MTC ENGINEERING")

    txns = statement_json.get("transactions") or []
    if not isinstance(txns, list):
        return candidates

    pat_legal = re.compile(
        r"([A-Z][A-Z0-9&.\- ]{2,80}\b(?:SDN\.?\s*BHD\.?|SDN\.?|BERHAD|BHD)\b)"
    )
    pat_type = re.compile(
        r"([A-Z][A-Z0-9&.\- ]{2,60}\b(?:ENGINEERING|HOLDINGS|TRADING|SOLUTIONS|CONSULT|ENTERPRISE)\b)"
    )

    for t in txns[:3000]:
        if not isinstance(t, dict):
            continue
        desc = t.get("description")
        if not isinstance(desc, str) or not desc.strip():
            continue

        u = desc.upper()

        for m in pat_legal.finditer(u):
            candidates.append(m.group(1).strip())

        if ("SDN" not in u) and ("BHD" not in u) and ("BERHAD" not in u):
            for m in pat_type.finditer(u):
                candidates.append(m.group(1).strip())

        if "MTC" in u:
            m = re.search(r"(MTC[ A-Z0-9&.\-]{0,30}ENGINEERING(?:[ A-Z0-9&.\-]{0,20})?)", u)
            if m:
                candidates.append(m.group(1).strip())

    return candidates


def _pick_best_company_name(candidates: List[str]) -> str:
    """
    Choose most plausible company name using frequency + scoring.
    Strongly prefer 'MTC ENGINEERING' and penalize bank-like entities.
    """
    if not candidates:
        return ""

    norm: List[str] = []
    for c in candidates:
        c = _sanitize_company_name(c).upper()
        c = re.sub(r"\s+", " ", c).strip()
        if c:
            norm.append(c)

    freq: Dict[str, int] = {}
    for c in norm:
        freq[c] = freq.get(c, 0) + 1

    scored: List[Tuple[float, str]] = []
    for c, f in freq.items():
        score = 0.0
        score += f * 2.0

        if "MTC" in c:
            score += 30
        if "MTC ENGINEERING" in c:
            score += 60

        if re.search(r"\bSDN\b", c):
            score += 8
        if re.search(r"\bBHD\b", c):
            score += 6
        if re.search(r"\bBERHAD\b", c):
            score += 4

        if "BANK" in c:
            score -= 50
        if "ISLAMIC" in c:
            score -= 15

        ln = len(c)
        if 8 <= ln <= 70:
            score += 5
        else:
            score -= 5

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    if "MTC ENGINEERING" in best and "SDN" in best and "BHD" not in best and "BERHAD" not in best:
        best = best.replace("SDN.", "SDN").strip()
        best = best + " BHD"

    return _sanitize_company_name(best)


# -------------------------
# Engine patching (minimal)
# -------------------------

def _patch_engine_config(
    engine,
    company_name: str,
    account_rows: List[Tuple[str, dict, AccountConfig]],
) -> None:
    """
    Patch engine globals minimally (no change to analysis logic):
      - COMPANY_NAME
      - COMPANY_KEYWORDS
      - FILE_PATHS
      - ACCOUNT_INFO (MUST include account_holder for your engine)
      - RELATED_PARTIES (keep if exists)
    """
    company_name = _sanitize_company_name(company_name)
    company_keywords = _generate_company_keywords(company_name)

    file_paths: Dict[str, str] = {}
    account_info: Dict[str, dict] = {}

    for fname, _payload, cfg in account_rows:
        file_paths[cfg.account_id] = fname  # placeholder until temp files are written
        account_info[cfg.account_id] = {
            "account_number": cfg.account_number,
            "bank_name": cfg.bank_name,
            "account_holder": company_name,  # REQUIRED by your engine (prevents KeyError)
            "account_type": cfg.account_type,
            "classification": cfg.classification,
            "description": f"{cfg.bank_name} ({cfg.classification})",
        }

    engine.COMPANY_NAME = company_name
    engine.COMPANY_KEYWORDS = company_keywords
    engine.FILE_PATHS = file_paths
    engine.ACCOUNT_INFO = account_info

    if not hasattr(engine, "RELATED_PARTIES") or engine.RELATED_PARTIES is None:
        engine.RELATED_PARTIES = {}


def _build_temp_files_and_update_paths(
    engine,
    account_rows: List[Tuple[str, dict, AccountConfig]],
) -> None:
    new_paths: Dict[str, str] = {}
    for _fname, payload, cfg in account_rows:
        temp_path = _write_temp_json(payload, suffix=f"_{cfg.account_id}.json")
        new_paths[cfg.account_id] = temp_path
    engine.FILE_PATHS = new_paths


def _infer_bank_name(payload: dict) -> Optional[str]:
    for key in ["bank", "bank_name", "bankName", "institution", "institutionName"]:
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    acct = payload.get("account") or {}
    if isinstance(acct, dict):
        for key in ["bank", "bank_name", "bankName", "institution", "institutionName"]:
            v = acct.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _infer_account_number(payload: dict) -> Optional[str]:
    for key in ["account_number", "accountNumber", "acct_no", "acctNo", "number"]:
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    acct = payload.get("account") or {}
    if isinstance(acct, dict):
        for key in ["account_number", "accountNumber", "acct_no", "acctNo", "number"]:
            v = acct.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


# -------------------------
# Enrichment layer (fills empty HTML categories)
# -------------------------

ROUND_FIGURE_THRESHOLD = 10000


def _is_round_figure(amount: float) -> bool:
    try:
        return float(amount) >= ROUND_FIGURE_THRESHOLD and float(amount) % 1000 == 0
    except Exception:
        return False


def _normalize_party(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_bank(name: str) -> bool:
    u = _normalize_party(name)
    if not u:
        return False
    bank_markers = [
        "BANK", "ISLAMIC BANK", "BERHAD", "CIMB", "MAYBANK", "PUBLIC BANK", "RHB",
        "HONG LEONG", "HLB", "MUAMALAT", "AFFIN", "AMBANK", "UOB", "HSBC", "OCBC"
    ]
    return any(m in u for m in bank_markers)


def _extract_counterparty(description: str, company_keywords_upper: List[str]) -> Optional[str]:
    """
    Heuristic counterparty extraction from transaction description.
    We keep it simple + robust for Malaysian bank narratives.
    """
    if not description:
        return None

    d = description.upper().strip()
    d = re.sub(r"\s+", " ", d)

    # Remove common noise prefixes
    d = re.sub(r"^(TRANSFER|TRF|IBG|GIRO|RENTAS|DUITNOW|INSTANT|PAYMENT|PYMT|FT|FUND TRANSFER)\b[:\- ]*", "", d)
    d = re.sub(r"\b(REF|REFERENCE|NO|NUM|ID)\b[:\- ]*[A-Z0-9\-\/]+", "", d)
    d = re.sub(r"\b(AUTH|TRACE|TXN|TRANSACTION)\b[:\- ]*[A-Z0-9\-\/]+", "", d)
    d = re.sub(r"\s+", " ", d).strip()

    # Common patterns: "TO <NAME>", "FROM <NAME>"
    m = re.search(r"\b(?:TO|FROM)\s+([A-Z0-9& ]{3,70})$", d)
    if m:
        cand = _normalize_party(m.group(1))
        if cand and not any(k in cand for k in company_keywords_upper):
            return cand.title()

    # If description contains clear company-like phrase
    m2 = re.search(r"\b([A-Z][A-Z0-9& ]{2,60}\b(?:SDN\.?\s*BHD|SDN\.?|BHD|BERHAD)\b)", d)
    if m2:
        cand = _normalize_party(m2.group(1))
        if cand and not any(k in cand for k in company_keywords_upper):
            return cand.title()

    # Fallback: take first 2-4 tokens if they look like a name/vendor
    tokens = [t for t in d.split() if len(t) >= 3]
    if len(tokens) >= 2:
        cand = _normalize_party(" ".join(tokens[:4]))
        if cand and not any(k in cand for k in company_keywords_upper):
            # avoid returning pure bank names as "counterparty"
            if _looks_like_bank(cand):
                return None
            return cand.title()

    return None


def _collect_all_transactions(account_rows: List[Tuple[str, dict, AccountConfig]]) -> List[dict]:
    """
    Your uploaded JSON transactions already include:
      date, description, debit, credit, balance, bank, source_file, ...
    We also attach account_id/bank_name from UI config.
    """
    all_tx: List[dict] = []
    for _fname, payload, cfg in account_rows:
        txns = payload.get("transactions") or []
        if not isinstance(txns, list):
            continue
        for t in txns:
            if not isinstance(t, dict):
                continue
            debit = float(t.get("debit") or 0)
            credit = float(t.get("credit") or 0)
            if debit == 0 and credit == 0:
                continue

            bank_from_txn = t.get("bank")
            bank_final = (bank_from_txn or cfg.bank_name or "").strip()

            all_tx.append(
                {
                    "account_id": cfg.account_id,
                    "account_label": cfg.bank_name,
                    "bank": bank_final,
                    "date": t.get("date"),
                    "description": t.get("description", ""),
                    "debit": debit,
                    "credit": credit,
                    "amount": credit if credit > 0 else debit,
                    "type": "CREDIT" if credit > 0 else "DEBIT",
                    "source_file": t.get("source_file"),
                }
            )
    return all_tx


def _enrich_results(
    results: dict,
    account_rows: List[Tuple[str, dict, AccountConfig]],
    company_name: str,
) -> dict:
    """
    Fill empty categories expected by HTML:
      - counterparties top payers/payees
      - round figure transactions lists
      - inject counterparty + bank into category top transactions (best-effort match)
    """
    enriched = results  # mutate in place
    company_keywords_upper = _generate_company_keywords(company_name)

    all_tx = _collect_all_transactions(account_rows)

    # Build a lookup for matching (date, amount, desc80) -> tx
    lookup: Dict[Tuple[str, float, str], List[dict]] = {}
    for t in all_tx:
        key = (
            str(t.get("date") or ""),
            round(float(t.get("amount") or 0), 2),
            (t.get("description") or "")[:80].upper(),
        )
        lookup.setdefault(key, []).append(t)

    # 1) Round figure lists (credits only, same definition as engine)
    round_fig = [t for t in all_tx if t["type"] == "CREDIT" and _is_round_figure(t["amount"])]
    round_fig_sorted = sorted(round_fig, key=lambda x: float(x["amount"]), reverse=True)

    rf_all = [
        {
            "date": t["date"],
            "desc": (t["description"] or "")[:120],
            "amount": round(float(t["amount"]), 2),
            "type": t["type"],
            "account": t.get("account_label") or t.get("account_id"),
            "bank": t.get("bank") or "",
            "counterparty": _extract_counterparty(t.get("description", ""), company_keywords_upper),
        }
        for t in round_fig_sorted
    ]

    flags = enriched.setdefault("flags", {})
    rft = flags.setdefault("round_figure_transactions", {})
    rft["top_10_transactions"] = rf_all[:10]
    rft["all_transactions"] = rf_all

    # 2) Counterparty aggregation (top payers/payees)
    payers: Dict[str, Dict[str, Any]] = {}
    payees: Dict[str, Dict[str, Any]] = {}

    for t in all_tx:
        cp = _extract_counterparty(t.get("description", ""), company_keywords_upper)
        if not cp:
            continue

        bank = t.get("bank") or ""
        amt = float(t["amount"])

        if t["type"] == "CREDIT":
            rec = payers.setdefault(cp, {"name": cp, "total": 0.0, "count": 0, "banks": {}})
            rec["total"] += amt
            rec["count"] += 1
            rec["banks"][bank] = rec["banks"].get(bank, 0) + 1
        else:
            rec = payees.setdefault(cp, {"name": cp, "total": 0.0, "count": 0, "banks": {}})
            rec["total"] += amt
            rec["count"] += 1
            rec["banks"][bank] = rec["banks"].get(bank, 0) + 1

    def _top_list(d: Dict[str, Dict[str, Any]], n: int = 10) -> List[dict]:
        items = list(d.values())
        items.sort(key=lambda x: x["total"], reverse=True)
        out = []
        for it in items[:n]:
            # flatten bank counts into "most_common_bank"
            banks = it.get("banks", {})
            most_common_bank = ""
            if banks:
                most_common_bank = sorted(banks.items(), key=lambda kv: kv[1], reverse=True)[0][0]
            out.append(
                {
                    "name": it["name"],
                    "total_amount": round(it["total"], 2),
                    "transaction_count": it["count"],
                    "most_common_bank": most_common_bank,
                }
            )
        return out

    counterparties = enriched.setdefault("counterparties", {})
    counterparties["top_payers"] = _top_list(payers, 10)
    counterparties["top_payees"] = _top_list(payees, 10)

    # 3) Inject bank + counterparty into categories top_5_transactions (best effort)
    cats = enriched.get("categories", {})
    for side in ["credits", "debits"]:
        if side not in cats or not isinstance(cats[side], list):
            continue
        for bucket in cats[side]:
            tx_list = bucket.get("top_5_transactions") or []
            if not isinstance(tx_list, list):
                continue
            for row in tx_list:
                try:
                    k = (
                        str(row.get("date") or ""),
                        round(float(row.get("amount") or 0), 2),
                        (row.get("description") or "")[:80].upper(),
                    )
                    candidates = lookup.get(k, [])
                    if candidates:
                        t = candidates[0]
                        row["bank"] = t.get("bank") or ""
                        row["account"] = t.get("account_label") or t.get("account_id")
                        row["counterparty"] = _extract_counterparty(t.get("description", ""), company_keywords_upper)
                    else:
                        # fallback: at least attempt counterparty from the short desc
                        row["counterparty"] = row.get("counterparty") or _extract_counterparty(
                            row.get("description", ""), company_keywords_upper
                        )
                except Exception:
                    continue

    return enriched


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.set_page_config(page_title="Bank Analysis (Streamlit)", layout="wide")
    st.title("Bank Analysis")
    st.caption("Upload multiple bank statement JSON files, confirm company name, then run analysis.")

    st.subheader("1) Upload statement JSON files")
    uploaded_files = st.file_uploader(
        "Upload one or more JSON statement files",
        type=["json"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one statement JSON file to continue.")
        return

    parsed: List[Tuple[str, dict]] = []
    candidate_names: List[str] = []

    for f in uploaded_files:
        payload = _safe_json_load(f)
        parsed.append((f.name, payload))
        candidate_names.extend(_extract_candidate_company_names(payload, filename=f.name))

    detected_company = _pick_best_company_name(candidate_names)

    st.subheader("2) Company name (accuracy critical)")
    company_name = st.text_input(
        "Company name (edit to ensure it is correct)",
        value=detected_company or "",
        help="If auto-detect is wrong, overwrite here. This affects core matching.",
    )

    if not company_name.strip():
        st.warning("Please enter the company name to avoid mis-detection.")
        return

    with st.expander("Show generated company keywords"):
        st.code("\n".join(_generate_company_keywords(company_name)), language="text")

    st.subheader("3) Account metadata (per uploaded file)")
    st.caption("These feed the engine's ACCOUNT_INFO. Ensure they are correct.")

    account_rows: List[Tuple[str, dict, AccountConfig]] = []

    for idx, (fname, payload) in enumerate(parsed, start=1):
        with st.container(border=True):
            st.markdown(f"**File:** `{fname}`")

            col1, col2, col3 = st.columns(3)
            with col1:
                account_id = st.text_input(
                    f"Account ID ({fname})",
                    value=f"acc_{idx}",
                    key=f"accid_{idx}",
                )
                bank_name = st.text_input(
                    f"Bank name label ({fname})",
                    value=_infer_bank_name(payload) or "",
                    key=f"bank_{idx}",
                    help="Label for the account. Transaction-level bank is read from each txn['bank'] when available.",
                )
            with col2:
                account_number = st.text_input(
                    f"Account number ({fname})",
                    value=_infer_account_number(payload) or "",
                    key=f"accno_{idx}",
                )
                account_type = st.selectbox(
                    f"Account type ({fname})",
                    options=["checking", "savings", "credit", "other"],
                    index=0,
                    key=f"acctyp_{idx}",
                )
            with col3:
                classification = st.selectbox(
                    f"Classification ({fname})",
                    options=["operational", "cash_account", "inter_account", "credit_facility", "other"],
                    index=0,
                    key=f"class_{idx}",
                )

            cfg = AccountConfig(
                account_id=account_id.strip() or f"acc_{idx}",
                bank_name=bank_name.strip() or "(Unknown bank)",
                account_number=account_number.strip() or "(Not provided)",
                account_type=account_type,
                classification=classification,
            )

            account_rows.append((fname, payload, cfg))

    st.subheader("4) Run analysis")
    run = st.button("Run", type="primary")
    if not run:
        return

    try:
        engine = _import_engine()
    except Exception as e:
        st.error("Failed to import core engine. Ensure bank_analysis_v5_2_1.py is in the repo root.")
        st.exception(e)
        return

    try:
        _patch_engine_config(engine, company_name, account_rows)
        _build_temp_files_and_update_paths(engine, account_rows)
    except Exception as e:
        st.error("Failed to configure engine.")
        st.exception(e)
        return

    try:
        results = engine.analyze()
    except Exception as e:
        st.error("Core analysis failed.")
        st.exception(e)
        return

    # Enrich results so HTML sections stop being empty and bank detection is present
    results = _enrich_results(results, account_rows, company_name)

    st.success("Analysis complete (enriched with bank + counterparty details).")

    st.subheader("Results (enriched)")
    st.json(results)

    st.subheader("Download output")
    out_str = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        "Download results.json",
        data=out_str.encode("utf-8"),
        file_name="results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
