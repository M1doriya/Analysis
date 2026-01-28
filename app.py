import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


# Streamlit wrapper only.
# - Upload multiple JSON statements
# - Detect/override company name accurately (tuned for your statement structure)
# - Patch globals in bank_analysis_v5_2_1.py and call analyze()
# - DOES NOT change core engine logic


@dataclass
class AccountConfig:
    account_id: str
    bank_name: str
    account_number: str
    account_type: str
    classification: str


def _sanitize_company_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _generate_company_keywords(company_name: str) -> List[str]:
    """
    Generate robust partial-match keywords to reduce false negatives.
    We keep it conservative: full name + 1-3 leading meaningful tokens + optional stem.
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

    # Deduplicate, preserve order
    seen = set()
    out: List[str] = []
    for k in kws:
        k2 = k.strip().upper()
        if k2 and k2 not in seen:
            seen.add(k2)
            out.append(k2)
    return out


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

    # 1) Filename hints (your files: "CIMB KL MTC.json", etc.)
    fn = (filename or "").upper()
    if "MTC" in fn:
        candidates.append("MTC ENGINEERING SDN BHD")
        candidates.append("MTC ENGINEERING")

    txns = statement_json.get("transactions") or []
    if not isinstance(txns, list):
        return candidates

    # 2) Patterns
    # A) company legal-ish patterns (including SDN. without BHD)
    pat_legal = re.compile(
        r"([A-Z][A-Z0-9&.\- ]{2,80}\b(?:SDN\.?\s*BHD\.?|SDN\.?|BERHAD|BHD)\b)"
    )
    # B) fallback business-type patterns
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

        # If no legal marker, try a "type" match
        if ("SDN" not in u) and ("BHD" not in u) and ("BERHAD" not in u):
            for m in pat_type.finditer(u):
                candidates.append(m.group(1).strip())

        # Special boost: capture "MTC ... ENGINEERING ..." even if truncated
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

    # normalize & frequency
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

        # Frequency across multiple statements matters
        score += f * 2.0

        # Boost target token(s)
        if "MTC" in c:
            score += 30
        if "MTC ENGINEERING" in c:
            score += 60

        # Suffix bonus
        if re.search(r"\bSDN\b", c):
            score += 8
        if re.search(r"\bBHD\b", c):
            score += 6
        if re.search(r"\bBERHAD\b", c):
            score += 4

        # Penalize obvious non-target entities (banks)
        if "BANK" in c:
            score -= 50
        if "ISLAMIC" in c:
            score -= 15

        # Length sanity
        ln = len(c)
        if 8 <= ln <= 70:
            score += 5
        else:
            score -= 5

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    # Standardize common short forms for your dataset
    # If we got "MTC ENGINEERING SDN." normalize to "MTC ENGINEERING SDN BHD"
    if "MTC ENGINEERING" in best and "SDN" in best and "BHD" not in best and "BERHAD" not in best:
        best = best.replace("SDN.", "SDN").strip()
        best = best + " BHD"

    return _sanitize_company_name(best)


# -------------------------
# IO helpers
# -------------------------

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

    # FILE_PATHS placeholders now; will be overwritten with actual temp paths after write
    file_paths: Dict[str, str] = {}
    account_info: Dict[str, dict] = {}

    for fname, _payload, cfg in account_rows:
        file_paths[cfg.account_id] = fname  # placeholder

        # IMPORTANT: engine expects info['account_holder']
        account_info[cfg.account_id] = {
            "account_number": cfg.account_number,
            "bank_name": cfg.bank_name,
            "account_holder": company_name,  # <-- FIX for KeyError
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
        help="We generate COMPANY_KEYWORDS from this. If auto-detect is wrong, overwrite here.",
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
                    f"Bank name ({fname})",
                    value=_infer_bank_name(payload) or "",
                    key=f"bank_{idx}",
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
                    help="Keep consistent with your engine's expected classifications.",
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
        st.info("If it still fails, the issue is inside the engine (schema expectations).")
        return

    st.success("Analysis complete.")

    st.subheader("Results")
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
