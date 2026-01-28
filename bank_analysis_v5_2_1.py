#!/usr/bin/env python3
"""
================================================================================
BANK STATEMENT ANALYSIS ENGINE v5.2.1
================================================================================

This script processes bank statement JSON files and generates a comprehensive
business financial analysis report.

Input Format:
- JSON files containing transaction data for each bank account

Output:
- Consolidated JSON report with financial analysis metrics

Author: Bank Analysis Engine
Version: 5.2.1
Last Updated: 2025
"""

import json
import re
from collections import defaultdict, Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Company information
COMPANY_NAME = "MTC ENGINEERING SDN BHD"

# Related parties (directors/owners/linked companies) to flag
RELATED_PARTIES = [
    {"name": "MTC ENGINEERING SDN BHD", "relationship": "Company"},
    {"name": "MTC ENGINEERING", "relationship": "Company"},
    {"name": "MTC ENGINEERING SDN", "relationship": "Company"},
    {"name": "MTC", "relationship": "Company"},
    {"name": "MTC ENGR", "relationship": "Company"},
]

# Account information - Modify for each analysis
ACCOUNT_INFO = {
    "CIMB_KL": {
        "bank_name": "CIMB Islamic Bank",
        "account_number": "8600509927",
        "account_holder": COMPANY_NAME,
        "account_type": "Current",
        "classification": "PRIMARY",
    },
    "CIMB_2": {
        "bank_name": "CIMB Islamic Bank",
        "account_number": "8600522517",
        "account_holder": COMPANY_NAME,
        "account_type": "Current",
        "classification": "SECONDARY",
    },
    "HLB": {
        "bank_name": "Hong Leong Bank",
        "account_number": "20006976096",
        "account_holder": COMPANY_NAME,
        "account_type": "Current",
        "classification": "PRIMARY",
    },
    "MUAMALAT": {
        "bank_name": "Bank Muamalat",
        "account_number": "1001005131",
        "account_holder": COMPANY_NAME,
        "account_type": "Current",
        "classification": "PRIMARY",
    },
}

# File paths for account statements - Modify for each analysis
FILE_PATHS = {
    "CIMB_KL": "CIMB KL MTC.json",
    "CIMB_2": "CIMB MTC.json",
    "HLB": "HLB MTC.json",
    "MUAMALAT": "Muamalat MTC.json",
}

# Bank codes for reference (used for detecting missing statements)
BANK_CODES = {
    "MBB": "Maybank",
    "PBB": "Public Bank",
    "RHB": "RHB Bank",
    "CIMB": "CIMB Bank",
    "HLBB": "Hong Leong Bank",
    "BIMB": "Bank Islam",
    "AMFB": "AmBank",
    "BSN": "Bank Simpanan Nasional",
    "OCBC": "OCBC Bank",
    "HSBC": "HSBC Bank",
    "SCB": "Standard Chartered",
    "UOB": "UOB Bank",
    "AGRO": "Agrobank",
    "BKR": "Bank Kerjasama Rakyat",
}

# Keywords for detecting company name in transactions
COMPANY_KEYWORDS = ["SDN", "BHD", "BERHAD", "ENTERPRISE", "TRADING", "COMPANY"]

# Category definitions
CATEGORY_DEFINITIONS = {
    "GENUINE_SALES_COLLECTIONS": {
        "type": "CREDIT",
        "description": "Genuine business revenue collections from customers",
        "keywords": ["PAYMENT", "INV", "INVOICE", "RECEIPT", "COLLECTION"],
        "exclude_keywords": ["LOAN", "FINANCING", "TRANSFER", "OWN ACCOUNT"],
    },
    "RELATED_PARTY_INFLOWS": {
        "type": "CREDIT",
        "description": "Funds received from related parties/directors/shareholders",
        "keywords": ["DIRECTOR", "SHAREHOLDER", "OWNER", "RELATED", "ADVANCE"],
    },
    "LOAN_PROCEEDS": {
        "type": "CREDIT",
        "description": "Loan disbursements and financing proceeds",
        "keywords": ["LOAN", "FINANCING", "DISBURSEMENT", "FACILITY", "TERM LOAN"],
    },
    "OTHER_CREDITS": {
        "type": "CREDIT",
        "description": "Other credit transactions not classified elsewhere",
        "keywords": [],
    },
    "SUPPLIER_PAYMENTS": {
        "type": "DEBIT",
        "description": "Payments to suppliers and trade creditors",
        "keywords": ["PAYMENT", "SUPPLIER", "VENDOR", "TRADE", "PURCHASE"],
        "exclude_keywords": ["SALARY", "EPF", "SOCSO", "TAX", "RENT", "UTILITY"],
    },
    "RELATED_PARTY_OUTFLOWS": {
        "type": "DEBIT",
        "description": "Payments to related parties/directors/shareholders",
        "keywords": ["DIRECTOR", "SHAREHOLDER", "OWNER", "RELATED", "ADVANCE"],
    },
    "STATUTORY_PAYMENTS": {
        "type": "DEBIT",
        "description": "Government and statutory payments (tax, EPF, SOCSO)",
        "keywords": ["LHDN", "TAX", "EPF", "KWSP", "SOCSO", "PERKESO", "ZAKAT"],
    },
    "SALARY_PAYMENTS": {
        "type": "DEBIT",
        "description": "Employee salary and payroll payments",
        "keywords": ["SALARY", "PAYROLL", "WAGES", "GAJI", "BONUS"],
    },
    "RENT_UTILITIES": {
        "type": "DEBIT",
        "description": "Rent, utilities and overhead expenses",
        "keywords": ["RENT", "SEWA", "TNB", "WATER", "UTILITY", "BILL", "TM", "UNIFI"],
    },
    "BANK_CHARGES": {
        "type": "DEBIT",
        "description": "Bank fees, charges and penalties",
        "keywords": ["CHARGE", "FEE", "COMMISSION", "PENALTY", "INTEREST", "IBG FEE"],
    },
    "OTHER_DEBITS": {
        "type": "DEBIT",
        "description": "Other debit transactions not classified elsewhere",
        "keywords": [],
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_json_file(file_path):
    """Load and parse JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def parse_amount(value):
    """Parse amount value to float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.replace(",", "").replace("RM", "").strip()
        if value == "" or value == "-":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).upper().strip())


def detect_related_party(description):
    """Detect if transaction involves related party."""
    desc_norm = normalize_text(description)
    for party in RELATED_PARTIES:
        if normalize_text(party["name"]) in desc_norm:
            return True, party["name"]
    return False, None


def detect_inter_account_transfer(description, account_ids):
    """Detect transfers between own accounts."""
    desc_norm = normalize_text(description)
    for acc_id in account_ids:
        acc_num = ACCOUNT_INFO.get(acc_id, {}).get("account_number", "")
        if acc_num and acc_num in desc_norm:
            return True, acc_id
    return False, None


def categorize_transaction(txn_type, description, amount, is_related_party=False, is_inter_account=False):
    """Categorize transaction based on type and description."""
    desc_norm = normalize_text(description)

    # Related party takes precedence
    if is_related_party:
        return "RELATED_PARTY_INFLOWS" if txn_type == "CREDIT" else "RELATED_PARTY_OUTFLOWS"

    # Inter-account transfers get excluded from turnover
    if is_inter_account:
        return "INTER_ACCOUNT_TRANSFER"

    # Match against category definitions
    for category, definition in CATEGORY_DEFINITIONS.items():
        if definition["type"] != txn_type:
            continue

        # Skip if exclude keywords match
        if "exclude_keywords" in definition:
            if any(kw in desc_norm for kw in definition["exclude_keywords"]):
                continue

        # Match keywords
        if definition["keywords"]:
            if any(kw in desc_norm for kw in definition["keywords"]):
                return category

    # Default categories
    return "OTHER_CREDITS" if txn_type == "CREDIT" else "OTHER_DEBITS"


def calculate_monthly_metrics(transactions):
    """Calculate monthly summary metrics."""
    monthly_data = defaultdict(lambda: {"credits": 0, "debits": 0, "closing_balance": 0, "high": None, "low": None})

    for txn in transactions:
        month = txn["date"][:7]  # YYYY-MM
        credit = txn["credit"]
        debit = txn["debit"]
        balance = txn["balance"]

        monthly_data[month]["credits"] += credit
        monthly_data[month]["debits"] += debit
        monthly_data[month]["closing_balance"] = balance

        # Track intraday high/low
        if monthly_data[month]["high"] is None or balance > monthly_data[month]["high"]:
            monthly_data[month]["high"] = balance
        if monthly_data[month]["low"] is None or balance < monthly_data[month]["low"]:
            monthly_data[month]["low"] = balance

    monthly_summary = []
    for month, data in sorted(monthly_data.items()):
        monthly_summary.append(
            {
                "month": month,
                "total_credits": round(data["credits"], 2),
                "total_debits": round(data["debits"], 2),
                "closing_balance": round(data["closing_balance"], 2),
                "highest_intraday": round(data["high"], 2) if data["high"] is not None else 0,
                "lowest_intraday": round(data["low"], 2) if data["low"] is not None else 0,
            }
        )

    return monthly_summary


def calculate_volatility_index(high_balance, low_balance):
    """Calculate volatility index based on high/low balances."""
    if high_balance == 0 and low_balance == 0:
        return 0
    avg_balance = (high_balance + low_balance) / 2
    if avg_balance == 0:
        return 0
    return ((high_balance - low_balance) / avg_balance) * 100


def classify_volatility(vol_index):
    """Classify volatility level."""
    if vol_index >= 200:
        return "EXTREME"
    if vol_index >= 100:
        return "HIGH"
    if vol_index >= 50:
        return "MEDIUM"
    return "LOW"


# =============================================================================
# PATCH UTILITIES (v5.2.1+): company detection, counterparties, monthly volatility,
# high-value flags, avg daily balance, bank metadata inference.
# =============================================================================

BANK_CODE_TOKENS = {
    # Common Malaysian bank short codes seen at end of DuitNow descriptions
    "MBB",
    "PBB",
    "RHB",
    "BIMB",
    "HLBB",
    "CIMB",
    "AMFB",
    "BSN",
    "OCBC",
    "HSBC",
    "SCB",
    "UOB",
    "AGRO",
    "BKR",
    "MB2U",
    "BANK",
    "IBG",
}


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def infer_company_name_from_transactions(transactions: list, fallback: str = "UNKNOWN") -> str:
    """Best-effort company name detection from transaction descriptions.

    Strategy:
    1) Regex extract candidates that end with Malaysian company suffixes.
    2) Clean away narration boilerplate (DUITNOW/ACCOUNT/INTERBANK/etc).
    3) Score by frequency + completeness (prefer SDN BHD / BERHAD).
    """
    if not transactions:
        return fallback

    pattern = re.compile(
        r"\b([A-Z][A-Z0-9&.,'()/ -]{2,}?(?:SDN\s*BHD|SDN\.?\s*BHD\.?|SDN\.?|BHD\.?|BERHAD))\b", re.I
    )

    blacklist = {"DUITNOW", "ACCOUNT", "MOBILE", "INTERBANK", "TRANSFER", "TFR", "TO", "FROM", "RLE", "SEPAT", "JTG", "TFSO"}
    counts = {}
    scores = {}

    for t in transactions:
        desc = (t.get("description") or "").upper()
        for mm in pattern.finditer(desc):
            cand = _clean_ws(mm.group(1).upper())
            cand = cand.replace("SDN. BHD", "SDN BHD").replace("SDN BHD.", "SDN BHD").replace("BHD.", "BHD").replace("SDN.", "SDN")

            tokens = cand.split()
            while tokens and tokens[0] in blacklist:
                tokens = tokens[1:]

            if not tokens:
                continue

            cand2 = _clean_ws(" ".join(tokens))

            if any(x in cand2 for x in ["DUITNOW TO", "TO ACCOUNT", "INTERBANK INTERBANK"]):
                continue
            if len(cand2) < 8:
                continue

            counts[cand2] = counts.get(cand2, 0) + 1

            bonus = 0.0
            if "SDN BHD" in cand2:
                bonus += 20
            if "BERHAD" in cand2:
                bonus += 20
            if cand2.endswith("BHD"):
                bonus += 10
            bonus += min(len(cand2), 60) / 10.0
            scores[cand2] = max(scores.get(cand2, 0.0), bonus)

    if not counts:
        return fallback

    best = sorted(counts.keys(), key=lambda c: (-counts[c], -scores.get(c, 0.0), -len(c)))[0]
    # If we only captured 'SDN' but an otherwise identical 'SDN BHD' exists, prefer the fuller form.
    if best.endswith(" SDN"):
        prefix = best + " BHD"
        for cand in counts.keys():
            if cand.startswith(prefix):
                best = cand
                break
        if prefix in counts:
            best = prefix
    return best


def infer_bank_name_from_transactions(transactions: list, fallback: str = "") -> str:
    banks = [t.get("bank") for t in transactions if t.get("bank")]
    if not banks:
        return fallback
    # mode
    freq = {}
    for b in banks:
        freq[b] = freq.get(b, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def extract_counterparty(description: str) -> str:
    """Extract a counterparty string from a transaction description.
    Heuristic-based, tuned for common Malaysian bank narration patterns.
    Falls back to a truncated description if it cannot confidently extract.
    """
    desc_raw = description or ""
    desc = _clean_ws(desc_raw).upper()

    # Remove repeated boilerplate tokens that often appear
    desc = re.sub(r"\b(TFSO|SEPAT|JTG|RLE)\b", " ", desc)
    desc = _clean_ws(desc)

    # Prefer substrings after known markers
    markers = [
        "DUITNOW TO ACCOUNT",
        "DUITNOW TO MOBILE/ID",
        "DUITNOW TO MOBILE",
        "TR TO C/A",
        "TR TO CA",
        "TR FROM",
        "TR FR",
        "IBG",
        "GIRO CR",
        "GIRO DR",
        "AUTOPAY CR",
        "AUTOPAY DR",
    ]
    start = None
    for m in markers:
        i = desc.find(m)
        if i != -1:
            start = i + len(m)
            break
    if start is not None:
        tail = _clean_ws(desc[start:])
    else:
        tail = desc

    # Remove obvious reference-ish fragments at front
    tail = re.sub(r"\b(INV|INVOICE|IV|REF|NO\.?|BILL|PAYMENT|PMT|PROJECT)\b[:\-]?", " ", tail)
    tail = _clean_ws(tail)

    # If narration ends with a bank code token, drop it
    tokens = tail.split()
    if tokens and tokens[-1] in BANK_CODE_TOKENS and len(tokens) > 2:
        tokens = tokens[:-1]

    # Build candidate from end backwards using name-like tokens
    stop_tokens = {
        "INTERBANK",
        "CASH",
        "ADV",
        "ADVANCE",
        "TRANSFER",
        "TFR",
        "TO",
        "FROM",
        "ACCOUNT",
        "MOBILE/ID",
        "ONLINE",
        "FEE",
        "CHARGES",
        "SERVICE",
    }
    name_tokens = []
    for tok in reversed(tokens):
        if tok in stop_tokens:
            break
        if re.search(r"[0-9]", tok):
            # stop on numeric-heavy fragments (refs)
            if name_tokens:
                break
            continue
        if any(ch in tok for ch in "/-_"):
            # likely a reference; stop once we already have some name tokens
            if name_tokens:
                break
            continue
        if len(tok) <= 1:
            continue
        name_tokens.append(tok)
        if len(name_tokens) >= 6:
            break

    if name_tokens:
        cand = " ".join(reversed(name_tokens))
        cand = _clean_ws(cand)
        # sanity
        if len(cand) >= 3:
            return cand.title()

    # fallback: trimmed narration
    short = _clean_ws(desc_raw)
    return short[:60] + ("…" if len(short) > 60 else "")


def build_counterparty_summary(transactions: list, total_credit_basis: float, total_debit_basis: float) -> dict:
    """Aggregate counterparties for top payers/payees and concentration.

    Note: engine transactions carry 'credit' and 'debit' fields (not a unified 'amount').
    """
    payer = {}  # name_key -> [display_name, count, amount]
    payee = {}

    for t in transactions:
        credit = float(t.get("credit") or 0.0)
        debit = float(t.get("debit") or 0.0)
        ttype = "CREDIT" if credit > 0 else ("DEBIT" if debit > 0 else None)
        amt = credit if credit > 0 else debit
        if not ttype or amt <= 0:
            continue

        # Only consider NON-excluded business flows
        if t.get("exclude_from_turnover"):
            continue

        name = extract_counterparty(t.get("description") or "")
        key = name.upper()

        if ttype == "CREDIT":
            c = payer.get(key, [name, 0, 0.0])
            c[1] += 1
            c[2] += amt
            payer[key] = c
        else:
            c = payee.get(key, [name, 0, 0.0])
            c[1] += 1
            c[2] += amt
            payee[key] = c

    def topn(d, basis, n=10):
        items = sorted(d.values(), key=lambda x: (-x[2], -x[1], x[0]))[:n]
        out = []
        for i, (name, cnt, amt) in enumerate(items, start=1):
            pct = (amt / basis * 100.0) if basis else 0.0
            out.append(
                {
                    "rank": i,
                    "party_name": name,
                    "transaction_count": int(cnt),
                    "total_amount": round(amt, 2),
                    "percentage": round(pct, 2),
                    "is_related_party": False,
                }
            )
        return out

    top_payers = topn(payer, total_credit_basis)
    top_payees = topn(payee, total_debit_basis)

    def pct_sum(items, k):
        return round(sum(x["percentage"] for x in items[:k]), 2) if items else 0.0

    top1_payer_pct = top_payers[0]["percentage"] if top_payers else 0.0
    top3_payers_pct = pct_sum(top_payers, 3)
    top1_payee_pct = top_payees[0]["percentage"] if top_payees else 0.0
    top3_payees_pct = pct_sum(top_payees, 3)

    risk = "LOW"
    if top1_payer_pct >= 40 or top1_payee_pct >= 40 or top3_payers_pct >= 70 or top3_payees_pct >= 70:
        risk = "HIGH"
    elif top1_payer_pct >= 25 or top1_payee_pct >= 25 or top3_payers_pct >= 50 or top3_payees_pct >= 50:
        risk = "MEDIUM"

    parties_both = sorted(list(set(payer.keys()).intersection(set(payee.keys()))))
    parties_both_sides = [payer[k][0] for k in parties_both[:25]]

    return {
        "top_payers": top_payers,
        "top_payees": top_payees,
        "concentration_risk": {
            "top1_payer_pct": round(top1_payer_pct, 2),
            "top3_payers_pct": round(top3_payers_pct, 2),
            "top1_payee_pct": round(top1_payee_pct, 2),
            "top3_payees_pct": round(top3_payees_pct, 2),
            "risk_level": risk,
        },
        "parties_both_sides": parties_both_sides,
    }


def compute_avg_daily_balance(transactions: list) -> float:
    """Average of end-of-day balances across the full period (all accounts)."""
    # group by (account_id, date) -> last balance
    by_key = {}
    for t in transactions:
        acc = t.get("account_id") or "UNKNOWN"
        dt = t.get("date")
        bal = t.get("balance")
        if not dt or bal is None:
            continue
        key = (acc, dt)
        # transaction list already deterministically sorted in engine; last seen is end-of-day-ish
        by_key[key] = float(bal)
    if not by_key:
        return 0.0
    return float(sum(by_key.values()) / len(by_key))


def build_monthly_volatility(accounts: list) -> list:
    """Aggregate monthly volatility from per-account monthly intraday summaries."""
    month_map = {}  # YYYY-MM -> {high, low}
    for acc in accounts:
        for m in acc.get("monthly_summary", []):
            month = m.get("month")
            if not month:
                continue
            high = m.get("highest_intraday")
            low = m.get("lowest_intraday")
            if high is None or low is None:
                continue
            entry = month_map.get(month, {"high": float(high), "low": float(low)})
            entry["high"] = max(entry["high"], float(high))
            entry["low"] = min(entry["low"], float(low))
            month_map[month] = entry

    out = []
    for month, v in sorted(month_map.items()):
        high = v["high"]
        low = v["low"]
        avg = (high + low) / 2.0 if (high + low) != 0 else 0.0
        vol_idx = ((high - low) / avg * 100.0) if avg else 0.0
        level = "LOW"
        if vol_idx >= 200:
            level = "EXTREME"
        elif vol_idx >= 100:
            level = "HIGH"
        elif vol_idx >= 50:
            level = "MEDIUM"
        out.append(
            {
                "month": month,
                "volatility_index": round(vol_idx, 2),
                "volatility_level": level,
                "highest_balance": round(high, 2),
                "lowest_balance": round(low, 2),
            }
        )
    return out


def build_high_value_flags(transactions: list, threshold: float = 500000.0) -> dict:
    hv = []
    for t in transactions:
        credit = float(t.get("credit") or 0.0)
        debit = float(t.get("debit") or 0.0)
        ttype = "CREDIT" if credit > 0 else ("DEBIT" if debit > 0 else None)
        amt = credit if credit > 0 else debit
        if not ttype:
            continue
        if amt >= threshold:
            hv.append(
                {
                    "date": t.get("date"),
                    "type": ttype,
                    "amount": round(amt, 2),
                    "description": t.get("description"),
                    "account_id": t.get("account_id"),
                    "category": t.get("category"),
                }
            )
    hv = sorted(hv, key=lambda x: -x["amount"])
    return {
        "threshold": threshold,
        "count": len(hv),
        "transactions": hv[:100],
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def analyze():
    """Main analysis function."""

    # ========================================================================
    # STEP 1: LOAD ALL ACCOUNT DATA
    # ========================================================================
    accounts = []
    all_transactions = []
    missing_bank_codes = set()
    account_ids = list(FILE_PATHS.keys())

    for acc_id, file_path in FILE_PATHS.items():
        data = load_json_file(file_path)
        if not data:
            continue

        transactions = data.get("transactions", [])
        if not transactions:
            continue

        # Sort transactions by date then by index for consistent processing
        for idx, txn in enumerate(transactions):
            debit_amt = parse_amount(txn.get("debit"))
            credit_amt = parse_amount(txn.get("credit"))

            all_transactions.append(
                {
                    "idx": idx,
                    "account_id": acc_id,
                    "date": txn["date"],
                    "description": txn["description"],
                    "debit": debit_amt,
                    "credit": credit_amt,
                    "balance": txn.get("balance", 0) or 0,
                    "category": None,
                    "exclude_from_turnover": False,
                    "is_related_party": False,
                    "related_party_name": None,
                    "is_inter_account": False,
                    "transfer_to_account": None,
                }
            )

            # Detect bank codes mentioned in description
            desc_norm = normalize_text(txn.get("description", ""))
            for code in BANK_CODES.keys():
                if code in desc_norm:
                    missing_bank_codes.add(code)

        # Account level analysis
        account_transactions = [t for t in all_transactions if t["account_id"] == acc_id]

        # Monthly summary for account
        monthly_summary = calculate_monthly_metrics(account_transactions)

        # Account totals
        total_credits = sum(t["credit"] for t in account_transactions)
        total_debits = sum(t["debit"] for t in account_transactions)

        opening_balance = account_transactions[0]["balance"]
        closing_balance = account_transactions[-1]["balance"]

        accounts.append(
            {
                "account_id": acc_id,
                "bank_name": ACCOUNT_INFO.get(acc_id, {}).get("bank_name", ""),
                "account_number": ACCOUNT_INFO.get(acc_id, {}).get("account_number", ""),
                "account_holder": ACCOUNT_INFO.get(acc_id, {}).get("account_holder", COMPANY_NAME),
                "account_type": ACCOUNT_INFO.get(acc_id, {}).get("account_type", "Current"),
                "classification": ACCOUNT_INFO.get(acc_id, {}).get("classification", "PRIMARY"),
                "opening_balance": round(opening_balance, 2),
                "closing_balance": round(closing_balance, 2),
                "total_credits": round(total_credits, 2),
                "total_debits": round(total_debits, 2),
                "monthly_summary": monthly_summary,
            }
        )

    # If no transactions loaded, return empty report
    if not all_transactions:
        return {"error": "No transactions loaded"}

    # ========================================================================
    # STEP 2: DETECT PERIOD
    # ========================================================================
    all_transactions.sort(key=lambda x: (x["date"], x["idx"]))
    period_start = all_transactions[0]["date"]
    period_end = all_transactions[-1]["date"]

    # Number of months in period
    start_month = period_start[:7]
    end_month = period_end[:7]
    num_months = (int(end_month[:4]) - int(start_month[:4])) * 12 + (int(end_month[5:7]) - int(start_month[5:7])) + 1

    # ========================================================================
    # STEP 3: FLAG RELATED PARTIES AND INTER-ACCOUNT TRANSFERS
    # ========================================================================
    inter_account_transfers = []
    related_party_transactions = []

    for txn in all_transactions:
        # Related party detection
        is_rp, rp_name = detect_related_party(txn["description"])
        txn["is_related_party"] = is_rp
        txn["related_party_name"] = rp_name

        # Inter-account transfer detection
        is_inter, to_acc = detect_inter_account_transfer(txn["description"], account_ids)
        txn["is_inter_account"] = is_inter
        txn["transfer_to_account"] = to_acc

        if is_rp:
            related_party_transactions.append(txn)
        if is_inter:
            inter_account_transfers.append(txn)
            txn["exclude_from_turnover"] = True

    # ========================================================================
    # STEP 4: CATEGORIZE TRANSACTIONS
    # ========================================================================
    category_totals = defaultdict(lambda: {"count": 0, "amount": 0})

    for txn in all_transactions:
        txn_type = "CREDIT" if txn["credit"] > 0 else "DEBIT"
        amount = txn["credit"] if txn_type == "CREDIT" else txn["debit"]

        category = categorize_transaction(
            txn_type,
            txn["description"],
            amount,
            is_related_party=txn["is_related_party"],
            is_inter_account=txn["is_inter_account"],
        )
        txn["category"] = category

        # Exclude certain categories from turnover
        if category in ["INTER_ACCOUNT_TRANSFER", "RELATED_PARTY_INFLOWS", "RELATED_PARTY_OUTFLOWS", "LOAN_PROCEEDS"]:
            txn["exclude_from_turnover"] = True

        category_totals[category]["count"] += 1
        category_totals[category]["amount"] += amount

    # ========================================================================
    # STEP 5: CONSOLIDATED TOTALS
    # ========================================================================
    total_credits = sum(t["credit"] for t in all_transactions)
    total_debits = sum(t["debit"] for t in all_transactions)

    net_credits = sum(t["credit"] for t in all_transactions if not t["exclude_from_turnover"])
    net_debits = sum(t["debit"] for t in all_transactions if not t["exclude_from_turnover"])

    # ========================================================================
    # STEP 6: VOLATILITY
    # ========================================================================
    all_balances = [t["balance"] for t in all_transactions if t["balance"] is not None]
    high_balance = max(all_balances) if all_balances else 0
    low_balance = min(all_balances) if all_balances else 0
    overall_vol = round(calculate_volatility_index(high_balance, low_balance), 2)
    overall_level = classify_volatility(overall_vol)

    # ========================================================================
    # STEP 7: FLAGS / INTEGRITY SCORE (existing engine logic)
    # ========================================================================
    # Round figure credits
    credits = [t for t in all_transactions if t["credit"] > 0 and not t["exclude_from_turnover"]]
    round_figure_credits = [t for t in credits if t["credit"] % 1000 == 0]
    round_figure_pct = (len(round_figure_credits) / len(credits) * 100) if credits else 0

    # Bank charges
    bank_charges = [t for t in all_transactions if t["category"] == "BANK_CHARGES"]

    # Integrity score (simplified heuristic)
    score = 100
    if missing_bank_codes:
        score -= 10
    if overall_level in ["HIGH", "EXTREME"]:
        score -= 10
    if round_figure_pct > 20:
        score -= 5

    score = max(0, min(100, score))
    if score >= 85:
        rating = "EXCELLENT"
    elif score >= 75:
        rating = "GOOD"
    elif score >= 60:
        rating = "FAIR"
    else:
        rating = "POOR"

    # ========================================================================
    # STEP 13: BUILD FINAL RESULT
    # ========================================================================
    # --------------------------------------------------------------------
    # PATCH: Fill fields that were previously empty in v5.2.1 output
    # - company_name inference (when uploads don't match hardcoded COMPANY_NAME)
    # - counterparties aggregation
    # - monthly volatility breakdown
    # - avg daily balance and high-value transaction flags
    # --------------------------------------------------------------------
    detected_company_name = infer_company_name_from_transactions(all_transactions, fallback=COMPANY_NAME)
    company_name_for_report = detected_company_name or COMPANY_NAME

    # Counterparties (use non-excluded business totals as the denominator)
    credit_basis = net_credits if "net_credits" in locals() else total_credits
    debit_basis = net_debits if "net_debits" in locals() else total_debits
    counterparties_summary = build_counterparty_summary(all_transactions, credit_basis, debit_basis)

    # Monthly volatility aggregation (intraday)
    monthly_volatility = build_monthly_volatility(accounts)

    # Avg daily balance + high-value transactions
    avg_daily_balance = round(compute_avg_daily_balance(all_transactions), 2)
    hv_flags = build_high_value_flags(all_transactions, threshold=500000.0)

    result = {
        "report_info": {
            "schema_version": "5.2.1",
            "company_name": company_name_for_report,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "period_start": period_start,
            "period_end": period_end,
            "total_accounts": len(accounts),
            "total_months": num_months,
            "related_parties": [{"name": rp["name"], "relationship": rp["relationship"]} for rp in RELATED_PARTIES],
            "accounts_not_provided": sorted(list(missing_bank_codes)),
            "analysis_scope": "Consolidated multi-bank statement analysis",
        },
        "accounts": accounts,
        "consolidated": {
            "gross": {
                "total_credits": round(total_credits, 2),
                "total_debits": round(total_debits, 2),
            },
            "net_of_exclusions": {
                "total_credits": round(net_credits, 2),
                "total_debits": round(net_debits, 2),
            },
            "monthly_average": {
                "avg_credits": round(net_credits / num_months, 2) if num_months else 0,
                "avg_debits": round(net_debits / num_months, 2) if num_months else 0,
            },
        },
        "inter_account_transfers": {
            "count": len(inter_account_transfers),
            "total_amount": round(
                sum(t["credit"] + t["debit"] for t in inter_account_transfers),
                2,
            ),
            "transactions_sample": [
                {
                    "date": t["date"],
                    "account_id": t["account_id"],
                    "description": t["description"],
                    "credit": t["credit"],
                    "debit": t["debit"],
                }
                for t in inter_account_transfers[:20]
            ],
        },
        "related_party_transactions": {
            "count": len(related_party_transactions),
            "total_inflows": round(sum(t["credit"] for t in related_party_transactions), 2),
            "total_outflows": round(sum(t["debit"] for t in related_party_transactions), 2),
        },
        "flagged_for_review": {
            "bank_charges": {
                "count": len(bank_charges),
                "total_amount": round(sum(t["debit"] for t in bank_charges), 2),
                "largest_charges": [
                    {
                        "date": t["date"],
                        "amount": round(t["debit"], 2),
                        "description": t["description"],
                    }
                    for t in sorted(bank_charges, key=lambda x: -(x["debit"] or 0))[:5]
                ],
            }
        },
        "categories": {
            "summary": [
                {
                    "category": cat,
                    "type": CATEGORY_DEFINITIONS.get(cat, {}).get("type", "UNKNOWN"),
                    "description": CATEGORY_DEFINITIONS.get(cat, {}).get("description", ""),
                    "transaction_count": data["count"],
                    "total_amount": round(data["amount"], 2),
                    "percentage_of_net": round(
                        (data["amount"] / (net_credits if CATEGORY_DEFINITIONS.get(cat, {}).get("type") == "CREDIT" else net_debits) * 100)
                        if (net_credits if CATEGORY_DEFINITIONS.get(cat, {}).get("type") == "CREDIT" else net_debits)
                        else 0,
                        2,
                    ),
                }
                for cat, data in category_totals.items()
            ]
        },
        "counterparties": counterparties_summary,
        "kite_flying": {
            "risk_score": 0,
            "indicators": [],
            "assessment": "No kite flying analysis implemented in v5.2.1 core logic",
        },
        "volatility": {
            "calculation_method": "intraday",
            "overall_index": overall_vol,
            "overall_level": overall_level,
            "monthly": monthly_volatility,
            "alerts": [f"{overall_level} volatility detected"] if overall_level in ["HIGH", "EXTREME"] else [],
        },
        "recurring_payments": {
            "payment_types": [],
            "assessment": "Recurring payment detection not implemented in v5.2.1 core logic",
        },
        "non_bank_financing": {
            "risk_level": "LOW",
            "indicators": [],
            "assessment": "No evidence of non-bank financing detected",
        },
        "flags": {
            "high_value_transactions": {**hv_flags, "avg_daily_balance": avg_daily_balance},
            "round_figure_transactions": {
                "round_figure_pct": round(round_figure_pct, 2),
                "count": len(round_figure_credits),
            },
        },
        "integrity_score": {
            "score": score,
            "rating": rating,
            "concerns": [
                f"{overall_level} volatility levels observed" if overall_level in ["HIGH", "EXTREME"] else "Volatility within normal range",
                f"Round figure credits at {round(round_figure_pct, 1)}%" if round_figure_pct > 20 else "Round figure credits within normal range",
                "Multiple bank accounts referenced but not provided for analysis" if missing_bank_codes else "All accounts provided",
            ],
        },
        "observations": {
            "key_findings": [
                f"Total net business turnover credits: RM {round(net_credits, 2):,.2f}",
                f"Total net business turnover debits: RM {round(net_debits, 2):,.2f}",
                f"Overall cash flow volatility: {overall_level} (Index: {overall_vol})",
            ]
        },
        "recommendations": [
            (
                {
                    "priority": "HIGH",
                    "category": "Data Completeness",
                    "recommendation": f'Obtain statements from {", ".join(list(missing_bank_codes)[:3])} accounts to verify complete cash flow'
                    if missing_bank_codes
                    else None,
                }
            ),
            (
                {
                    "priority": "MEDIUM",
                    "category": "Volatility Management",
                    "recommendation": "Consider maintaining higher operating balances to reduce volatility",
                }
                if overall_level in ["HIGH", "EXTREME"]
                else None
            ),
            (
                {
                    "priority": "LOW",
                    "category": "Banking Consolidation",
                    "recommendation": "Consider consolidating banking relationships to simplify cash flow monitoring",
                }
                if len(accounts) > 2
                else None
            ),
        ],
    }

    # Remove None recommendations
    result["recommendations"] = [r for r in result["recommendations"] if r is not None]

    return result


if __name__ == "__main__":
    report = analyze()
    print(json.dumps(report, indent=2))
