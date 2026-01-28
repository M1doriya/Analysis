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
    cleaned = re.sub(r"\b(SDN\.?|BHD\.?|BERHAD|SENDIRIAN|LIMITED|LTD\.?|CO\.?|COMPANY)\b", "", name)
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
    """Heuristic: extract company-like names from descriptions/account fields."""
    cands: List[str] = []

    # Some generators include these keys; we handle if present.
    for k in ["account_holder", "account_name", "company_name"]:
        v = statement_json.get(k)
        if isinstance(v, str) and v.strip():
            cands.append(v.strip())

    txns = statement_json.get("transactions")
    if isinstance(txns, list):
        # Match patterns like "... MTC ENGINEERING SDN BHD" or "... SDN. BHD"
        # Keep it conservative to reduce garbage.
        patt = re.compile(
            r"\b([A-Z0-9&.,/\- ]{3,}?(?:SDN\.?\s*BHD\.?|BERHAD|BHD\.?))\b",
            re.IGNORECASE,
        )
        for t in txns[:2000]:  # cap for performance
            d = t.get("description") if isinstance(t, dict) else None
            if not isinstance(d, str):
                continue
            m = patt.search(d.upper())
            if m:
                cands.append(m.group(1).strip())
    return cands


def _detect_company_name(uploaded_jsons: List[dict]) -> Optional[str]:
    """Pick most frequent candidate; returns None if not enough signal."""
    freq: Dict[str, int] = {}
    for j in uploaded_jsons:
        for c in _extract_candidate_company_names(j):
            c = _sanitize_company_name(c).upper()
            if not c:
                continue
            # Normalize punctuation spacing
            c = c.replace("SDN. BHD", "SDN BHD").replace("SDN.BHD", "SDN BHD")
            c = re.sub(r"\s+", " ", c).strip()
            freq[c] = freq.get(c, 0) + 1
    if not freq:
        return None
    # Prefer the longest among equally frequent to avoid truncated matches
    best = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
    return best


def _default_account_id_from_filename(name: str, idx: int) -> str:
    stem = Path(name).stem.upper()
    # common mappings users use (you can extend this)
    for key in ["CIMB_KL", "CIMBKL", "CIMB", "HLB", "BMMB", "MUAMALAT"]:
        if key.replace("_", "") in stem.replace("_", ""):
            return "CIMB_KL" if key in {"CIMB_KL", "CIMBKL"} else key
    return f"ACC_{idx+1}"


def main() -> None:
    st.set_page_config(page_title="Bank Statement Analysis", layout="wide")
    st.title("Bank Statement Analysis Engine (Streamlit)")

    st.info(
        "Upload one or more statement JSON files (the input format described in README). "
        "The app will run the deterministic analysis and let you download the output JSON."
    )

    uploads = st.file_uploader(
        "Bank statement JSON files",
        type=["json"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.stop()

    uploaded_jsons: List[dict] = []
    uploaded_names: List[str] = []
    for f in uploads:
        try:
            uploaded_jsons.append(json.load(f))
            uploaded_names.append(f.name)
        except Exception as e:
            st.error(f"Failed to read {f.name} as JSON: {e}")
            st.stop()

    # Company name detection + override
    detected = _detect_company_name(uploaded_jsons)
    st.subheader("Company identification")
    company_name = st.text_input(
        "Company name (auto-detected; edit if wrong)",
        value=_sanitize_company_name(detected or ""),
        help="Used for report metadata and for inter-account marker matching.",
    )
    company_name = _sanitize_company_name(company_name)

    company_keywords = _generate_company_keywords(company_name) if company_name else []
    st.caption("Company keywords used for partial matching (auto-generated)")
    st.code("\n".join(company_keywords) if company_keywords else "(no keywords)")

    st.subheader("Accounts mapping")
    st.write(
        "Map each uploaded file to an account id and metadata. "
        "Account IDs must be unique (e.g., CIMB_KL, CIMB, HLB, BMMB)."
    )

    default_bank_names = {
        "CIMB": "CIMB Islamic Bank",
        "CIMB_KL": "CIMB Islamic Bank",
        "HLB": "Hong Leong Islamic Bank",
        "BMMB": "Bank Muamalat Malaysia",
        "MUAMALAT": "Bank Muamalat Malaysia",
    }

    account_rows: List[Tuple[str, dict, AccountConfig]] = []

    for i, (fname, payload) in enumerate(zip(uploaded_names, uploaded_jsons)):
        with st.expander(f"{fname}", expanded=(i == 0)):
            acc_id_default = _default_account_id_from_filename(fname, i)
            acc_id = st.text_input(
                "Account ID",
                value=acc_id_default,
                key=f"acc_id_{i}",
            ).strip().upper()

            bank_name = st.text_input(
                "Bank name",
                value=default_bank_names.get(acc_id, ""),
                key=f"bank_name_{i}",
            ).strip()

            account_number = st.text_input(
                "Account number (optional)",
                value="",
                key=f"acc_no_{i}",
            ).strip()

            account_type = st.selectbox(
                "Account type",
                options=["Current", "Savings", "OD"],
                index=0,
                key=f"acc_type_{i}",
            )

            classification = st.selectbox(
                "Classification",
                options=["PRIMARY", "SECONDARY", "ESCROW", "PROJECT"],
                index=0,
                key=f"acc_class_{i}",
            )

            account_rows.append(
                (
                    fname,
                    payload,
                    AccountConfig(
                        account_id=acc_id,
                        bank_name=bank_name or "(Unknown bank)",
                        account_number=account_number or "(Not provided)",
                        account_type=account_type,
                        classification=classification,
                    ),
                )

    # Validate unique account IDs
    ids = [r[2].account_id for r in account_rows]
    if len(ids) != len(set(ids)):
        st.error("Account IDs must be unique. Please fix duplicates above.")
        st.stop()

    st.subheader("Related parties")
    st.write("Optional: add related parties (one per line). Use format: NAME | Relationship")
    rp_text = st.text_area(
        "Related parties",
        value="",
        placeholder="MTC FLOATING SOLUTIONS SDN BHD | Sister Company\nDIRECTOR NAME | Director",
    )

    related_parties = []
    for line in rp_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            name, rel = [x.strip() for x in line.split("|", 1)]
        else:
            name, rel = line, "Related Party"
        if name:
            related_parties.append({"name": name, "relationship": rel or "Related Party"})

    run = st.button("Run analysis")
    if not run:
        st.stop()

    if not company_name:
        st.error("Please provide a company name (auto-detect may be blank).")
        st.stop()

    # Write uploads to temp files, then patch globals in the core module.
    try:
        import bank_analysis_v5_2_1 as core
    except Exception as e:
        st.error(
            "Could not import bank_analysis_v5_2_1.py. "
            "Ensure it is in the same repository folder as app.py.\n\n"
            f"Import error: {e}"
        )
        st.stop()

    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmpdir.name)
    file_paths: Dict[str, str] = {}
    account_info: Dict[str, dict] = {}

    for fname, payload, cfg in account_rows:
        out_path = tmp_path / f"{cfg.account_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False))
        file_paths[cfg.account_id] = str(out_path)
        account_info[cfg.account_id] = {
            "bank_name": cfg.bank_name,
            "account_number": cfg.account_number,
            "account_holder": company_name,
            "account_type": cfg.account_type,
            "classification": cfg.classification,
        }

    # Patch configuration (no change to core logic)
    core.COMPANY_NAME = company_name
    core.COMPANY_KEYWORDS = company_keywords
    core.RELATED_PARTIES = related_parties
    core.FILE_PATHS = file_paths
    core.ACCOUNT_INFO = account_info

    with st.spinner("Analyzing..."):
        try:
            result = core.analyze()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.success("Analysis completed")
    st.subheader("Output")
    st.json(result, expanded=False)

    out_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "Download output JSON",
        data=out_bytes,
        file_name=f"{company_name.replace(' ', '_')}_analysis.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
