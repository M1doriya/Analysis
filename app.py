import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Use the patched engine (core logic preserved; only missing fields filled)
import bank_analysis_v5_2_1_fixed as engine


@dataclass
class UploadItem:
    filename: str
    bytes_data: bytes


def _slug(s: str) -> str:
    s = re.sub(r"\.[^.]+$", "", s)  # remove extension
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s.upper()[:40] or "ACCOUNT"


def _infer_account_id(filename: str, bank_name: str, used: set) -> str:
    base = _slug(filename)

    fn_u = filename.upper()
    bank_u = (bank_name or "").upper()

    # Optional normalization to keep compatibility with previous report labels
    if "CIMB" in bank_u or "CIMB" in fn_u:
        if any(k in fn_u for k in ["KL", "KUALA", "MAIN"]):
            base = "CIMB_KL"
        else:
            base = "CIMB_2"
    elif "HONG LEONG" in bank_u or "HLB" in fn_u or "HLBB" in fn_u:
        base = "HLB"
    elif "MUAMALAT" in bank_u or "MUAMALAT" in fn_u:
        base = "MUAMALAT"

    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _infer_bank_name(transactions: List[dict]) -> str:
    banks = [t.get("bank") for t in transactions if t.get("bank")]
    if not banks:
        return ""
    freq = {}
    for b in banks:
        freq[b] = freq.get(b, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _validate_json(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return "Top-level JSON must be an object"
    if "transactions" not in payload or not isinstance(payload["transactions"], list):
        return "JSON must contain a 'transactions' list"
    needed = {"date", "description", "debit", "credit", "balance"}
    for i, t in enumerate(payload["transactions"][:5]):
        if not isinstance(t, dict):
            return f"transactions[{i}] is not an object"
        missing = [k for k in needed if k not in t]
        if missing:
            return f"transactions[{i}] is missing required keys: {', '.join(missing)}"
    return None


st.set_page_config(page_title="Bank Statement Analysis (v5.2.1)", layout="wide")
st.title("Bank Statement Analysis Engine (v5.2.1)")

st.markdown(
    """
Upload **multiple** bank statement JSON files (the JSON output from your PDF->JSON extractor).
The app will:
- infer bank names automatically from transaction rows,
- infer company name from narrations (you can override),
- run the deterministic analysis engine,
- produce a consolidated JSON report.
"""
)

uploads = st.file_uploader(
    "Upload one or more statement JSON files",
    type=["json"],
    accept_multiple_files=True
)

if not uploads:
    st.info("Upload your JSON files to begin.")
    st.stop()

items: List[UploadItem] = []
for u in uploads:
    items.append(UploadItem(filename=u.name, bytes_data=u.getvalue()))

# Load & validate
loaded = []
errors = []
for it in items:
    try:
        payload = json.loads(it.bytes_data.decode("utf-8"))
    except Exception as e:
        errors.append(f"{it.filename}: JSON decode error: {e}")
        continue
    err = _validate_json(payload)
    if err:
        errors.append(f"{it.filename}: {err}")
        continue
    loaded.append((it.filename, payload))

if errors:
    st.error("Some files could not be used:")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

# Company override (optional)
default_company = engine.infer_company_name_from_transactions(
    [t for _, p in loaded for t in p["transactions"]],
    fallback="UNKNOWN"
)
company_override = st.text_input("Company name (optional override)", value=default_company)

# Build temp files + mappings
tmp_dir = Path(tempfile.mkdtemp(prefix="bank_uploads_"))
used_ids = set()
file_paths: Dict[str, str] = {}
account_info: Dict[str, dict] = {}

st.subheader("Detected accounts (edit if needed)")
for filename, payload in loaded:
    txns = payload["transactions"]
    bank_name = _infer_bank_name(txns) or "(Unknown Bank)"
    suggested_id = _infer_account_id(filename, bank_name, used_ids)

    with st.expander(f"{filename} -> {suggested_id}", expanded=False):
        st.write(f"Detected bank: {bank_name}")
        account_id = st.text_input("Account ID (used in report)", value=suggested_id, key=f"accid_{filename}")
        acc_no = st.text_input("Account number (optional)", value="", key=f"accno_{filename}")
        acc_type = st.selectbox("Account type", ["Current", "Savings", "Other"], index=0, key=f"acctype_{filename}")
        classification = st.selectbox("Classification", ["PRIMARY", "SECONDARY"], index=0, key=f"class_{filename}")

    out_path = tmp_dir / filename
    # write original bytes to preserve formatting
    out_path.write_bytes(next(i.bytes_data for i in items if i.filename == filename))

    file_paths[account_id] = str(out_path)
    account_info[account_id] = {
        "bank_name": bank_name,
        "account_number": acc_no or "(Not provided)",
        "account_holder": company_override or "UNKNOWN",
        "account_type": acc_type,
        "classification": classification
    }

# Wire into engine (no change to core analysis logic)
engine.FILE_PATHS = file_paths
engine.ACCOUNT_INFO = account_info
engine.COMPANY_NAME = company_override or "UNKNOWN"

run = st.button("Run analysis")
if not run:
    st.stop()

with st.spinner("Running analysis..."):
    report = engine.analyze()

st.success("Analysis complete.")

st.subheader("Output JSON")
st.json(report)

st.download_button(
    "Download JSON report",
    data=json.dumps(report, indent=2).encode("utf-8"),
    file_name=f"{(company_override or 'company').replace(' ', '_')}_analysis.json",
    mime="application/json"
)
