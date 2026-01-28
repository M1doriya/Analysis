import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


# We keep the core deterministic analysis logic in bank_analysis_v5_2_1.py.
# This Streamlit app only:
# 1) Accepts uploads
# 2) Detects/sets company name + keywords
# 3) Maps uploaded files to FILE_PATHS/ACCOUNT_INFO
# 4) Calls analyze() and shows the JSON output


@dataclass
class AccountConfig:
    account_id: str
    bank_name: str
    account_number: str
    account_type: str
    classification: str


def _sanitize_company_name(name: str) -> str:
    # Basic cleanup for UI/keywords.
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _generate_company_keywords(company_name: str) -> List[str]:
    """Generate robust partial-match keywords to reduce false negatives."""
    name = _sanitize_company_name(company_name).upper()
    if not name:
        return []

    # Remove common suffix noise for keyword generation
    cleaned = re.sub(
        r"\b(SDN\.?|BHD\.?|BERHAD|SENDIRIAN|LIMITED|LTD\.?|CO\.?|COMPANY)\b",
        "",
        name,
    )
    cleaned = re.sub(r"[^A-Z0-9& ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 2]

    kws: List[str] = []
    kws.append(name)  # full name still useful
    if words:
        kws.append(words[0])
    if len(words) >= 2:
        kws.append(" ".join(words[:2]))
    if len(words) >= 3:
        kws.append(" ".join(words[:3]))

    # Add a short "stem" keyword (first 8-10 chars of first word) if long
    if words and len(words[0]) >= 8:
        kws.append(words[0][:8])

    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in kws:
        k2 = k.strip().upper()
        if k2 and k2 not in seen:
            seen.add(k2)
            out.append(k2)
    return out


def _extract_candidate_company_names(statement_json: dict) -> List[str]:
    """Heuristic: mine plausible company names from statement fields."""
    candidates: List[str] = []

    # Common places a statement might store account holder / company name
    for key in [
        "account_holder",
        "accountHolder",
        "company_name",
        "companyName",
        "customer_name",
        "customerName",
        "name",
    ]:
        v = statement_json.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    # Sometimes nested in account details
    acct = statement_json.get("account") or statement_json.get("accountDetails") or {}
    if isinstance(acct, dict):
        for key in ["holder", "account_holder", "accountHolder", "name", "customerName"]:
            v = acct.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

    # Look into transactions for payee/description patterns like "... SDN BHD"
    txns = statement_json.get("transactions") or statement_json.get("transactionDetails") or []
    if isinstance(txns, list):
        for t in txns[:2000]:
            if not isinstance(t, dict):
                continue
            for k in ["description", "details", "merchant", "payee", "narration", "remarks"]:
                v = t.get(k)
                if not isinstance(v, str):
                    continue
                # Match typical MY company formats
                m = re.search(
                    r"([A-Z][A-Z0-9&.\- ]{3,80}\b(?:SDN\.?\s*BHD|BERHAD|BHD)\b)",
                    v.upper(),
                )
                if m:
                    candidates.append(m.group(1).strip())
    return candidates


def _pick_best_company_name(candidates: List[str]) -> str:
    """Choose the most plausible company name from extracted candidates."""
    if not candidates:
        return ""

    # Prefer names with SDN BHD/BERHAD/BHD markers
    scored: List[Tuple[int, str]] = []
    for c in candidates:
        uc = c.upper()
        score = 0
        if re.search(r"\bSDN\.?\s*BHD\b", uc):
            score += 50
        if re.search(r"\bBERHAD\b", uc):
            score += 40
        if re.search(r"\bBHD\b", uc):
            score += 30
        # Penalize very short/very long
        ln = len(c.strip())
        if 10 <= ln <= 60:
            score += 10
        else:
            score -= 10
        # Favor more "company-like" (multiple words)
        if len(c.strip().split()) >= 2:
            score += 5
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return _sanitize_company_name(scored[0][1])


def _safe_json_load(uploaded_file) -> dict:
    try:
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception:
        # fallback attempt if file already decoded differently
        return json.loads(uploaded_file.getvalue())


def _write_temp_json(payload: dict, suffix: str = ".json") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p)


def _import_engine():
    """
    Import core analysis engine.

    Keep bank_analysis_v5_2_1.py unchanged; we only patch its global config values.
    """
    import importlib

    return importlib.import_module("bank_analysis_v5_2_1")


def _patch_engine_config(
    engine,
    company_name: str,
    account_rows: List[Tuple[str, dict, AccountConfig]],
) -> None:
    """
    Patch engine globals minimally:
      - COMPANY_NAME
      - COMPANY_KEYWORDS
      - FILE_PATHS
      - ACCOUNT_INFO
      - RELATED_PARTIES (optional; keep existing if present)
    """
    company_name = _sanitize_company_name(company_name)
    company_keywords = _generate_company_keywords(company_name)

    # FILE_PATHS in the engine is used by load_data(). We map to our temp files.
    file_paths: Dict[str, str] = {}
    account_info: Dict[str, dict] = {}

    for fname, _payload, cfg in account_rows:
        # We write temp file per account and map it
        # Note: fname already corresponds to uploaded filename; unique-enough for UI.
        # We'll use account_id as primary key for engine mapping.
        file_paths[cfg.account_id] = fname
        account_info[cfg.account_id] = {
            "account_number": cfg.account_number,
            "bank_name": cfg.bank_name,
            "account_type": cfg.account_type,
            "classification": cfg.classification,
            "description": f"{cfg.bank_name} ({cfg.classification})",
        }

    # Patch
    engine.COMPANY_NAME = company_name
    engine.COMPANY_KEYWORDS = company_keywords

    # The engine expects FILE_PATHS to be account_id -> path on disk
    # Our fname values in file_paths currently store uploaded filenames; we will overwrite
    # them after writing temp files in the calling scope.
    engine.FILE_PATHS = file_paths
    engine.ACCOUNT_INFO = account_info

    # RELATED_PARTIES: keep existing if present, otherwise set empty dict
    if not hasattr(engine, "RELATED_PARTIES") or engine.RELATED_PARTIES is None:
        engine.RELATED_PARTIES = {}


def _build_temp_files_and_update_paths(
    engine,
    account_rows: List[Tuple[str, dict, AccountConfig]],
) -> None:
    """
    Write each uploaded JSON to a temp file and update engine.FILE_PATHS accordingly.
    """
    new_paths: Dict[str, str] = {}
    for _fname, payload, cfg in account_rows:
        temp_path = _write_temp_json(payload, suffix=f"_{cfg.account_id}.json")
        new_paths[cfg.account_id] = temp_path
    engine.FILE_PATHS = new_paths


def main():
    st.set_page_config(page_title="Bank Analysis (Streamlit)", layout="wide")
    st.title("Bank Analysis")
    st.caption("Upload bank statement JSON files, set company name, and run analysis.")

    st.subheader("1) Upload statement JSON files")
    uploaded_files = st.file_uploader(
        "Upload one or more JSON statement files",
        type=["json"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one statement JSON file to continue.")
        return

    # Parse uploads
    parsed: List[Tuple[str, dict]] = []
    candidate_names: List[str] = []
    for f in uploaded_files:
        payload = _safe_json_load(f)
        parsed.append((f.name, payload))
        candidate_names.extend(_extract_candidate_company_names(payload))

    detected_company = _pick_best_company_name(candidate_names)

    st.subheader("2) Company name (accuracy critical)")
    company_name = st.text_input(
        "Company name (edit to ensure it is correct)",
        value=detected_company or "",
        help="We will generate robust COMPANY_KEYWORDS from this to improve detection.",
    )

    if not company_name.strip():
        st.warning("Please enter the company name to avoid mis-detection.")
        return

    with st.expander("Show generated company keywords"):
        st.code("\n".join(_generate_company_keywords(company_name)), language="text")

    st.subheader("3) Account metadata (per uploaded file)")
    st.caption("These feed the engine's ACCOUNT_INFO. Keep accurate to avoid mislabeling.")

    # Defaults
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
                    help="Match your engine's expected classifications. Defaults provided.",
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

    # Patch configuration globals, then write temp files and update FILE_PATHS
    try:
        _patch_engine_config(engine, company_name, account_rows)
        _build_temp_files_and_update_paths(engine, account_rows)
    except Exception as e:
        st.error("Failed to configure engine.")
        st.exception(e)
        return

    # Run analysis (core logic)
    try:
        results = engine.analyze()
    except Exception as e:
        st.error("Core analysis failed.")
        st.exception(e)
        st.info("Tip: check that your uploaded JSON schema matches what the engine expects.")
        return

    st.success("Analysis complete.")

    st.subheader("Results")
    st.json(results)

    # Download
    st.subheader("Download output")
    out_str = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        "Download results.json",
        data=out_str.encode("utf-8"),
        file_name="results.json",
        mime="application/json",
    )


def _infer_bank_name(payload: dict) -> Optional[str]:
    # Optional heuristics from statement metadata
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


if __name__ == "__main__":
    main()
