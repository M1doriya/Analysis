import os
import re
import json
from pathlib import Path
import streamlit as st

# IMPORTANT: this must match your actual engine filename in the repo
# If your engine file is bank_analysis_v5_2_1.py, keep this import exactly:
import bank_analysis_v5_2_1 as engine


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "", name)
    return name or "uploaded.json"


def _detect_bank_key(file_name: str, bank_name: str) -> str:
    """
    Maps an uploaded statement JSON to the engine’s expected FILE_PATHS keys.
    Adjust here if you add more banks/accounts.
    """
    fn = file_name.lower()
    bn = (bank_name or "").lower()

    # CIMB (two accounts in your setup)
    if "cimb" in bn:
        # use filename hint for KL account
        if "kl" in fn or "kuala" in fn or "main" in fn:
            return "CIMB_KL"
        return "CIMB"

    # Hong Leong
    if "hong leong" in bn or "hlb" in fn or "hlib" in fn:
        return "HLB"

    # Muamalat
    if "muamalat" in bn or "bmmb" in fn:
        return "BMMB"

    # fallback (still stored, but may not be used by engine if no matching key)
    return "UNKNOWN"


def _read_bank_name_from_statement_json(obj: dict) -> str:
    """
    Your statement JSON format includes bank name on each transaction row.
    Example: "bank": "CIMB Islamic Bank" :contentReference[oaicite:2]{index=2}
    """
    try:
        txns = obj.get("transactions", [])
        if txns and isinstance(txns, list):
            b = txns[0].get("bank")
            if b:
                return str(b)
    except Exception:
        pass
    return ""


st.set_page_config(page_title="Bank Statement Analysis", layout="wide")
st.title("Bank Statement Analysis (Streamlit)")

st.markdown(
    """
Upload **multiple bank statement JSON outputs** (one per account/bank).
Then set the **Company Name** explicitly (so it doesn’t get guessed from transaction text).
"""
)

company_name = st.text_input("Company name (required)", value="MTC ENGINEERING SDN BHD")

uploaded_files = st.file_uploader(
    "Upload statement JSON files (multiple)",
    type=["json"],
    accept_multiple_files=True,
)

run_btn = st.button("Run analysis")


if run_btn:
    if not company_name.strip():
        st.error("Company name is required. Please fill it in.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one statement JSON.")
        st.stop()

    # 1) Save uploads + detect bank_name + map to keys
    detected = []
    for f in uploaded_files:
        raw = f.read()
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            st.error(f"Invalid JSON: {f.name}")
            st.stop()

        bank_name = _read_bank_name_from_statement_json(obj)
        bank_key = _detect_bank_key(f.name, bank_name)

        out_name = _safe_filename(f.name)
        out_path = UPLOAD_DIR / out_name
        out_path.write_bytes(raw)

        detected.append((f.name, bank_key, bank_name, str(out_path)))

    # Show what we mapped
    st.subheader("Detected uploads")
    st.write(
        [
            {
                "filename": fn,
                "mapped_key": key,
                "detected_bank_name": bn,
                "saved_path": p,
            }
            for (fn, key, bn, p) in detected
        ]
    )

    # 2) Override engine globals (keeps core logic intact)
    engine.COMPANY_NAME = company_name.strip()

    # Ensure account_holder uses the overridden company name
    for k, info in engine.ACCOUNT_INFO.items():
        if isinstance(info, dict):
            info["account_holder"] = engine.COMPANY_NAME

    # 3) Update FILE_PATHS based on uploads
    # Only set keys we recognize (CIMB_KL, CIMB, HLB, BMMB)
    for (fn, key, bn, p) in detected:
        if key in engine.FILE_PATHS:
            engine.FILE_PATHS[key] = p

            # Update bank_name in ACCOUNT_INFO if detected (helps report metadata)
            if bn and key in engine.ACCOUNT_INFO:
                engine.ACCOUNT_INFO[key]["bank_name"] = bn

    # 4) Run engine
    try:
        result = engine.analyze()
    except Exception as e:
        st.error("Core analysis failed.")
        st.exception(e)
        st.stop()

    # 5) Output
    st.success("Analysis completed.")
    st.subheader("Report info")
    st.json(result.get("report_info", {}))

    st.subheader("Full JSON")
    st.json(result)

    # Save result to file for download
    output_path = UPLOAD_DIR / "results.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    st.download_button(
        "Download results.json",
        data=output_path.read_bytes(),
        file_name="results.json",
        mime="application/json",
    )
