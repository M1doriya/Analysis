import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import streamlit as st


# ------------------------------------------------------------
# Robust engine import (fixes ModuleNotFoundError)
# ------------------------------------------------------------
def import_engine():
    """
    Try importing the core engine from common filenames.
    This prevents Streamlit Cloud ModuleNotFoundError when filenames differ.
    """
    tried = []

    for module_name in [
        "bank_analysis_v5_2_1_fixed",  # if you created this
        "bank_analysis_v5_2_1",        # your original upload name
        "bank_analysis",               # common fallback
    ]:
        try:
            mod = __import__(module_name)
            return mod
        except Exception as e:
            tried.append((module_name, str(e)))

    # Helpful diagnostics: show repo files
    repo_files = []
    try:
        repo_files = sorted([p.name for p in Path(".").glob("*.py")])
    except Exception:
        pass

    msg = (
        "Could not import engine module.\n\n"
        "Tried imports:\n"
        + "\n".join([f"- {m}: {err}" for m, err in tried])
        + "\n\n"
        + "Python files found in repo root:\n"
        + "\n".join([f"- {f}" for f in repo_files]) if repo_files else "\n(no .py files listed)"
    )
    raise ModuleNotFoundError(msg)


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class AccountConfig:
    account_id: str
    bank_name: str
    account_number: str
    account_type: str
    classification: str


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_json_load(uploaded_file) -> dict:
    try:
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception:
        return json.loads(uploaded_file.getvalue())


def write_temp_json(payload: dict, suffix: str = ".json") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p)


def infer_bank_from_transactions(payload: dict) -> str:
    txns = payload.get("transactions") or []
    banks = []
    if isinstance(txns, list):
        for t in txns:
            if isinstance(t, dict):
                b = t.get("bank")
                if isinstance(b, str) and b.strip():
                    banks.append(b.strip())
    if not banks:
        return ""
    return Counter(banks).most_common(1)[0][0]


def validate_statement_json(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return "Top-level JSON must be an object"
    txns = payload.get("transactions")
    if not isinstance(txns, list) or not txns:
        return "JSON must contain a non-empty 'transactions' list"
    needed = {"date", "description", "debit", "credit", "balance"}
    for i, t in enumerate(txns[:5]):
        if not isinstance(t, dict):
            return f"transactions[{i}] is not an object"
        missing = [k for k in needed if k not in t]
        if missing:
            return f"transactions[{i}] missing keys: {', '.join(missing)}"
    return None


def extract_candidate_company_names(statement_json: dict, filename: str = "") -> List[str]:
    """
    Best-effort company candidates from narrations + filename.
    Keeps it simple and safe; you can override manually in UI.
    """
    candidates: List[str] = []
    fn = (filename or "").upper()
    if "MTC" in fn:
        candidates += ["MTC ENGINEERING SDN BHD", "MTC ENGINEERING"]

    txns = statement_json.get("transactions") or []
    if not isinstance(txns, list):
        return candidates

    pat_legal = re.compile(
        r"([A-Z][A-Z0-9&.\- ]{2,80}\b(?:SDN\.?\s*BHD\.?|SDN\.?|BERHAD|BHD)\b)"
    )

    for t in txns[:2500]:
        if not isinstance(t, dict):
            continue
        desc = t.get("description")
        if not isinstance(desc, str) or not desc.strip():
            continue
        u = desc.upper()
        for m in pat_legal.finditer(u):
            candidates.append(m.group(1).strip())

    return candidates


def pick_best_company_name(candidates: List[str]) -> str:
    if not candidates:
        return ""
    norm = []
    for c in candidates:
        c = re.sub(r"\s+", " ", str(c).strip()).upper()
        if c:
            norm.append(c)
    freq = Counter(norm)
    # prefer those containing SDN/BHD and MTC
    def score(name: str) -> Tuple[int, int, int]:
        return (
            (1 if "MTC" in name else 0),
            (1 if ("SDN" in name or "BHD" in name or "BERHAD" in name) else 0),
            len(name),
        )
    best = sorted(freq.keys(), key=lambda n: (freq[n],) + score(n), reverse=True)[0]
    return best.title()


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Bank Analysis", layout="wide")
    st.title("Bank Analysis (Multi-Statement Upload)")

    # import engine with robust fallback
    try:
        engine = import_engine()
    except Exception as e:
        st.error("Failed to import analysis engine module.")
        st.code(str(e))
        st.stop()

    st.caption(
        "Upload multiple JSON statements. This app wires them into the core engine, "
        "ensuring required fields (like account_holder) exist to avoid runtime errors."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more JSON statement files",
        type=["json"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one statement JSON file to continue.")
        return

    parsed: List[Tuple[str, dict]] = []
    errors: List[str] = []
    candidates: List[str] = []

    for f in uploaded_files:
        try:
            payload = safe_json_load(f)
        except Exception as e:
            errors.append(f"{f.name}: JSON parse failed ({e})")
            continue

        err = validate_statement_json(payload)
        if err:
            errors.append(f"{f.name}: {err}")
            continue

        parsed.append((f.name, payload))
        candidates.extend(extract_candidate_company_names(payload, filename=f.name))

    if errors:
        st.error("Some files could not be used:")
        for e in errors:
            st.write(f"- {e}")
        if not parsed:
            st.stop()

    detected_company = pick_best_company_name(candidates)
    company_name = st.text_input(
        "Company name (edit to ensure correct)",
        value=detected_company or "",
    )
    if not company_name.strip():
        st.warning("Please enter the company name.")
        st.stop()

    st.subheader("Account setup (one per uploaded statement)")
    st.caption("Bank name defaults to the most common `transactions[].bank` in each file.")

    account_rows: List[Tuple[str, dict, AccountConfig]] = []

    used_ids = set()
    for idx, (fname, payload) in enumerate(parsed, start=1):
        inferred_bank = infer_bank_from_transactions(payload)
        default_acc_id = f"acc_{idx}"
        if default_acc_id in used_ids:
            default_acc_id = f"acc_{idx}_{len(used_ids)+1}"
        used_ids.add(default_acc_id)

        with st.container(border=True):
            st.markdown(f"**File:** `{fname}`")
            c1, c2, c3 = st.columns(3)

            with c1:
                account_id = st.text_input(
                    f"Account ID ({fname})",
                    value=default_acc_id,
                    key=f"accid_{idx}",
                )
                bank_name = st.text_input(
                    f"Bank Name ({fname})",
                    value=inferred_bank or "",
                    key=f"bank_{idx}",
                )

            with c2:
                account_number = st.text_input(
                    f"Account Number ({fname})",
                    value="",
                    key=f"accno_{idx}",
                )
                account_type = st.selectbox(
                    f"Account Type ({fname})",
                    options=["Current", "Savings", "Other"],
                    index=0,
                    key=f"acctype_{idx}",
                )

            with c3:
                classification = st.selectbox(
                    f"Classification ({fname})",
                    options=["PRIMARY", "SECONDARY", "OTHER"],
                    index=0,
                    key=f"class_{idx}",
                )

            cfg = AccountConfig(
                account_id=(account_id.strip() or default_acc_id),
                bank_name=(bank_name.strip() or inferred_bank or "(Unknown bank)"),
                account_number=(account_number.strip() or "(Not provided)"),
                account_type=account_type,
                classification=classification,
            )
            account_rows.append((fname, payload, cfg))

    st.subheader("Run analysis")
    if not st.button("Run", type="primary"):
        return

    # Patch engine globals safely (no logic change)
    try:
        engine.COMPANY_NAME = company_name.strip()

        # Some engines expect COMPANY_KEYWORDS
        if hasattr(engine, "COMPANY_KEYWORDS"):
            # keep whatever format engine expects; use a safe minimal set
            engine.COMPANY_KEYWORDS = [company_name.strip().upper()]

        file_paths: Dict[str, str] = {}
        account_info: Dict[str, dict] = {}

        for _fname, payload, cfg in account_rows:
            temp_path = write_temp_json(payload, suffix=f"_{cfg.account_id}.json")
            file_paths[cfg.account_id] = temp_path

            # IMPORTANT: include account_holder to avoid KeyError
            account_info[cfg.account_id] = {
                "bank_name": cfg.bank_name,
                "account_number": cfg.account_number,
                "account_holder": company_name.strip(),
                "account_type": cfg.account_type,
                "classification": cfg.classification,
                "description": f"{cfg.bank_name} ({cfg.classification})",
            }

        engine.FILE_PATHS = file_paths
        engine.ACCOUNT_INFO = account_info

        # Some engines use RELATED_PARTIES and may not have it set
        if not hasattr(engine, "RELATED_PARTIES") or engine.RELATED_PARTIES is None:
            engine.RELATED_PARTIES = {}

    except Exception as e:
        st.error("Failed to configure engine globals.")
        st.exception(e)
        st.stop()

    # Run core analysis
    try:
        results = engine.analyze()
    except Exception as e:
        st.error("Core analysis failed.")
        st.exception(e)
        st.stop()

    st.success("Analysis completed.")
    st.subheader("Output JSON")
    st.json(results)

    st.download_button(
        "Download results.json",
        data=json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
