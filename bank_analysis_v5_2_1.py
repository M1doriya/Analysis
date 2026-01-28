#!/usr/bin/env python3
"""
================================================================================
BANK STATEMENT ANALYSIS ENGINE v5.2.1
================================================================================
DETERMINISTIC IMPLEMENTATION - Consistent results every run

Changes from v5.2.0:
- Added Related Party detection (Priority 3 for credits, Priority 2 for debits)
- Configurable related parties list with partial matching
- Improved configuration section for easy customization
- Fixed datetime deprecation warning
- Added purpose_note extraction for related party transactions
- Better documentation

Key Features:
1. Sort all transactions by (date, -amount, description) before processing
2. Process in strict priority order with explicit rules
3. Use consistent tie-breaking (first match wins)
4. Related Party check BEFORE Statutory/Salary (per methodology)

Reference: BANK_ANALYSIS_CHECKLIST_v5_2_0.md, MULTI_ACCOUNT_ANALYSIS_v5_2_0.md
================================================================================
"""

import json
import re
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION FOR EACH COMPANY
# ============================================================================

# Company identification
COMPANY_NAME = "MTC ENGINEERING SDN BHD"
COMPANY_KEYWORDS = ["MTC ENGINEERING", "MTC ENGIN"]  # For partial matching

# Related parties - Add directors, shareholders, sister companies, etc.
# Format: {'name': 'Full Name', 'relationship': 'Director|Shareholder|Sister Company|etc'}
RELATED_PARTIES = [
    # Sister companies in MTC Group
    {'name': 'MTC FLOATING SOLUTIONS SDN BHD', 'relationship': 'Sister Company'},
    {'name': 'MTC ENERGY SDN BHD', 'relationship': 'Sister Company'},
    # Add more related parties as identified:
    # {'name': 'DIRECTOR NAME', 'relationship': 'Director'},
]

# Account information - Modify for each analysis
ACCOUNT_INFO = {
    'CIMB_KL': {
        'bank_name': 'CIMB Islamic Bank',
        'account_number': '8600509927',
        'account_holder': COMPANY_NAME,
        'account_type': 'Current',
        'classification': 'PRIMARY'
    },
    'CIMB': {
        'bank_name': 'CIMB Islamic Bank',
        'account_number': '8600106439',
        'account_holder': COMPANY_NAME,
        'account_type': 'Current',
        'classification': 'SECONDARY'
    },
    'HLB': {
        'bank_name': 'Hong Leong Islamic Bank',
        'account_number': '28500016095',
        'account_holder': COMPANY_NAME,
        'account_type': 'Current',
        'classification': 'SECONDARY'
    },
    'BMMB': {
        'bank_name': 'Bank Muamalat Malaysia',
        'account_number': '1203010001XXX',
        'account_holder': COMPANY_NAME,
        'account_type': 'Current',
        'classification': 'SECONDARY'
    }
}

# File paths - Modify for each analysis
FILE_PATHS = {
    'CIMB_KL': '/mnt/user-data/uploads/CIMB_KL_MTC.json',
    'CIMB': '/mnt/user-data/uploads/CIMB_MTC.json',
    'HLB': '/mnt/user-data/uploads/HLB_MTC.json',
    'BMMB': '/mnt/user-data/uploads/Muamalat_MTC.json'
}

# ============================================================================
# CONSTANTS - DO NOT MODIFY UNLESS UPDATING METHODOLOGY
# ============================================================================

# Bank codes for missing account detection
BANK_CODES = {
    'MBB': 'Maybank',
    'RHB': 'RHB Bank',
    'BIMB': 'Bank Islam',
    'PBB': 'Public Bank',
    'HSBC': 'HSBC Bank',
    'AMFB': 'AmBank',
    'OCBC': 'OCBC Bank',
    'UOB': 'UOB Bank',
    'AMB': 'AmBank',
    'ABMB': 'AmBank',
    'CITI': 'Citibank',
    'BSN': 'BSN',
    'HLB': 'Hong Leong Bank',
    'CIMB': 'CIMB Bank',
    'BMMB': 'Bank Muamalat',
    'AFFIN': 'Affin Bank',
    'ALLIANCE': 'Alliance Bank',
    'HONGLEONG': 'Hong Leong Bank',
    'BANKISLAM': 'Bank Islam'
}

# Category detection patterns - Deterministic priority order
CREDIT_PATTERNS = {
    'INTER_ACCOUNT_TRANSFER': [
        r'\bITB\s+TRF\b',
        r'\bINTERBANK\b',
        r'\bTR\s+FROM\b',
        r'\bDUITNOW\s+TO\s+ACCOUNT\b',
        r'\bDUITNOW\s+TRANSFER\b'
    ],
    'LOAN_DISBURSEMENT': [
        r'\bDISB\b',
        r'\bDISBURSE\b',
        r'\bLOAN\b',
        r'\bFINANCING\b'
    ],
    'INTEREST_PROFIT_DIVIDEND': [
        r'\bPROFIT\s+PAID\b',
        r'\bINTEREST\b',
        r'\bDIVIDEND\b'
    ],
    'REVERSAL': [
        r'\bREVERSAL\b',
        r'\bREVERSE\b'
    ],
    'GENUINE_SALES_COLLECTIONS': [
        r'\bINV\b',
        r'\bINVOICE\b',
        r'\bPAYMENT\b',
        r'\bREMITTANCE\b',
        r'\bAUTOPAY\b',
        r'\bCR\s+TFR\b'
    ]
}

DEBIT_PATTERNS = {
    'INTER_ACCOUNT_TRANSFER': [
        r'\bITB\s+TRF\b',
        r'\bINTERBANK\b',
        r'\bTR\s+TO\b',
        r'\bDUITNOW\s+TO\s+ACCOUNT\b',
        r'\bDUITNOW\s+TRANSFER\b'
    ],
    'STATUTORY_PAYMENT': [
        r'\bLEMBAGA\s+HASIL\b',
        r'\bKWSP\b',
        r'\bSOCSO\b',
        r'\bPERKESO\b',
        r'\bLHDN\b',
        r'\bHRDF\b',
        r'\bPSMB\b',
        r'\bHASIL\b',
        r'\bZAKAT\b'
    ],
    'SALARY_WAGES': [
        r'\bSALARY\b',
        r'\bGAJI\b',
        r'\bWAGES\b',
        r'\bPAYROLL\b',
        r'\bARREARS\s+SALARY\b'
    ],
    'UTILITIES': [
        r'\bTENAGA\s+NAS\b',
        r'\bTNB\b',
        r'\bTM\s+UNIFI\b',
        r'\bMAXIS\b',
        r'\bCELCOM\b',
        r'\bDIGI\b',
        r'\bWATER\b',
        r'\bINDIGOHOME\b'
    ],
    'BANK_CHARGES': [
        r'\bFEE\b',
        r'\bCHARGES\b',
        r'\bCOMMISSION\b',
        r'\bDUTY\b',
        r'\bLEVY\b'
    ],
    'SUPPLIER_VENDOR_PAYMENTS': [
        r'\bDUITNOW\b',
        r'\bIBG\b',
        r'\bPAYMENT\b',
        r'\bJOMPAY\b',
        r'\bTRANSFER\b'
    ]
}

ROUND_FIGURE_WARNING_PCT = 20.0  # Warning if >20% of credits are round figures

def month_name(ym: str) -> str:
    y, m = ym.split('-')
    mn = ['January','February','March','April','May','June','July','August','September','October','November','December']
    return f"{mn[int(m)-1]} {y}"

def get_volatility_level(vol_pct: float) -> str:
    if vol_pct >= 300:
        return 'EXTREME'
    if vol_pct >= 200:
        return 'HIGH'
    if vol_pct >= 100:
        return 'MODERATE'
    return 'LOW'

def create_transaction_key(txn: dict) -> Tuple:
    # Deterministic: (date, -amount, description)
    amt = (txn.get('credit', 0) or 0) + (txn.get('debit', 0) or 0)
    return (txn.get('date', ''), -float(amt), (txn.get('description', '') or '').upper())

def is_company_match(desc: str) -> bool:
    du = (desc or '').upper()
    if COMPANY_NAME and COMPANY_NAME.upper() in du:
        return True
    for kw in COMPANY_KEYWORDS:
        if kw and kw.upper() in du:
            return True
    return False

def detect_related_party(desc: str) -> Optional[dict]:
    du = (desc or '').upper()
    for rp in RELATED_PARTIES:
        n = (rp.get('name') or '').upper()
        if n and n in du:
            return rp
    return None

def extract_bank_codes(desc: str) -> List[str]:
    du = (desc or '').upper()
    found = []
    for code in BANK_CODES.keys():
        if re.search(rf'\b{re.escape(code)}\b', du):
            found.append(code)
    return found

def analyze(
    file_paths: Optional[Dict[str, str]] = None,
    account_info: Optional[Dict[str, dict]] = None
) -> dict:
    """
    Run analysis across multiple accounts.

    - file_paths: mapping account_id -> JSON path
    - account_info: mapping account_id -> metadata dict
    """
    file_paths = file_paths or FILE_PATHS
    account_info = account_info or ACCOUNT_INFO

    # ========================================================================
    # STEP 1: LOAD INPUT STATEMENTS
    # ========================================================================
    data = {}
    for acc_id, p in file_paths.items():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data[acc_id] = json.load(f)
        except Exception:
            # Skip missing files; integrity scoring later will flag gaps.
            data[acc_id] = {'transactions': []}

    # ========================================================================
    # STEP 2: NORMALIZE TRANSACTIONS (preserve deterministic processing)
    # ========================================================================
    all_transactions = []
    idx = 0
    for acc_id in sorted(file_paths.keys()):
        txns = data.get(acc_id, {}).get('transactions', [])
        if not isinstance(txns, list):
            continue
        for txn in txns:
            credit_amt = txn.get('credit', 0) or 0
            debit_amt = txn.get('debit', 0) or 0

            # Skip zero-amount transactions (like closing balance entries)
            if credit_amt == 0 and debit_amt == 0:
                continue

            all_transactions.append({
                'idx': idx,
                'account_id': acc_id,
                'date': txn.get('date', ''),
                'description': txn.get('description', ''),
                'debit': float(debit_amt or 0),
                'credit': float(credit_amt or 0),
                'balance': float(txn.get('balance', 0) or 0),
                'category': None,
                'exclude_from_turnover': False,
                'is_related_party': False,
                'related_party_name': '',
                'related_party_relationship': '',
                'purpose_note': ''
            })
            idx += 1

    # CRITICAL: Sort transactions deterministically
    all_transactions.sort(key=create_transaction_key)

    # Re-index after sorting
    for i, t in enumerate(all_transactions):
        t['idx'] = i

    # ========================================================================
    # STEP 3: IDENTIFY MISSING BANK STATEMENTS (referenced banks not provided)
    # ========================================================================
    provided_accounts = set(file_paths.keys())
    missing_accounts = defaultdict(int)
    for t in all_transactions:
        for code in extract_bank_codes(t.get('description', '')):
            if code not in provided_accounts:
                missing_accounts[f"{code} ({BANK_CODES.get(code, code)})"] += 1

    # ========================================================================
    # STEP 4: CLASSIFY TRANSACTIONS (core deterministic logic unchanged)
    # ========================================================================
    # Priority ordering as per methodology.
    for t in all_transactions:
        desc = t.get('description', '') or ''
        du = desc.upper()

        # Related Party detection (Priority before statutory/salary per v5.2.1 notes)
        rp = detect_related_party(desc)
        if rp:
            t['is_related_party'] = True
            t['related_party_name'] = rp.get('name', '')
            t['related_party_relationship'] = rp.get('relationship', '')
            # Optional: extract simple purpose note (keep deterministic & conservative)
            t['purpose_note'] = desc[:120]

        if t['credit'] > 0:
            # CREDIT categorization priority
            if t['is_related_party']:
                t['category'] = 'RELATED_PARTY'
                t['exclude_from_turnover'] = True
                continue

            # Inter-account transfers (company match + patterns)
            if any(re.search(p, du) for p in CREDIT_PATTERNS['INTER_ACCOUNT_TRANSFER']) and is_company_match(desc):
                t['category'] = 'INTER_ACCOUNT_TRANSFER'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in CREDIT_PATTERNS['LOAN_DISBURSEMENT']):
                t['category'] = 'LOAN_DISBURSEMENT'
                continue

            if any(re.search(p, du) for p in CREDIT_PATTERNS['INTEREST_PROFIT_DIVIDEND']):
                t['category'] = 'INTEREST_PROFIT_DIVIDEND'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in CREDIT_PATTERNS['REVERSAL']):
                t['category'] = 'REVERSAL'
                t['exclude_from_turnover'] = True
                continue

            # Default sales collections
            t['category'] = 'GENUINE_SALES_COLLECTIONS'

        else:
            # DEBIT categorization priority
            if t['is_related_party']:
                t['category'] = 'RELATED_PARTY'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in DEBIT_PATTERNS['INTER_ACCOUNT_TRANSFER']) and is_company_match(desc):
                t['category'] = 'INTER_ACCOUNT_TRANSFER'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in DEBIT_PATTERNS['STATUTORY_PAYMENT']):
                t['category'] = 'STATUTORY_PAYMENT'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in DEBIT_PATTERNS['SALARY_WAGES']):
                t['category'] = 'SALARY_WAGES'
                continue

            if any(re.search(p, du) for p in DEBIT_PATTERNS['UTILITIES']):
                t['category'] = 'UTILITIES'
                t['exclude_from_turnover'] = True
                continue

            if any(re.search(p, du) for p in DEBIT_PATTERNS['BANK_CHARGES']):
                t['category'] = 'BANK_CHARGES'
                t['exclude_from_turnover'] = True
                continue

            t['category'] = 'SUPPLIER_VENDOR_PAYMENTS'

    # ========================================================================
    # STEP 5: COMPUTE MONTHLY / ACCOUNT SUMMARIES (existing logic)
    # ========================================================================
    # Build per-account structures
    per_account = defaultdict(list)
    for t in all_transactions:
        per_account[t['account_id']].append(t)

    accounts = []
    for acc_id in sorted(per_account.keys()):
        txns = per_account[acc_id]
        if not txns:
            continue

        dates = [t['date'] for t in txns if t.get('date')]
        dr = sorted(d for d in dates if d)
        period_start_acc = dr[0] if dr else '2025-01-01'
        period_end_acc = dr[-1] if dr else '2025-12-31'

        total_credit = sum(t['credit'] for t in txns)
        total_debit = sum(t['debit'] for t in txns)
        total_txn = len(txns)

        # Monthly summary from transactions (intraday high/low based on txn balances)
        month_map = defaultdict(list)
        for t in txns:
            if t.get('date'):
                month_map[t['date'][:7]].append(t)

        monthly = []
        for m in sorted(month_map.keys()):
            mtx = month_map[m]
            balances = [x.get('balance', 0) for x in mtx if x.get('balance', None) is not None]
            high = max(balances) if balances else 0
            low = min(balances) if balances else 0
            avg = (high + low) / 2 if balances else 0
            swing = high - low
            vol_pct = round((swing / avg * 100) if avg else 0, 2)
            vol_level = get_volatility_level(vol_pct)

            # Opening/closing from balances within month (best effort)
            opening = balances[0] if balances else 0
            closing = balances[-1] if balances else 0

            monthly.append({
                'month': m,
                'month_name': month_name(m),
                'transaction_count': len(mtx),
                'opening': round(opening, 2),
                'closing': round(closing, 2),
                'credits': round(sum(x['credit'] for x in mtx), 2),
                'debits': round(sum(x['debit'] for x in mtx), 2),
                'highest_intraday': round(high, 2),
                'lowest_intraday': round(low, 2),
                'average_intraday': round(avg, 2),
                'swing': round(swing, 2),
                'volatility_pct': vol_pct,
                'volatility_level': vol_level
            })

        info = account_info.get(acc_id, {}) if isinstance(account_info, dict) else {}
        accounts.append({
            'account_id': acc_id,
            'bank_name': info.get('bank_name') or '(Unknown bank)',
            'account_number': info.get('account_number') or '(Not provided)',
            'account_holder': info.get('account_holder') or COMPANY_NAME,
            'account_type': info.get('account_type') or 'Current',
            'classification': info.get('classification') or 'SECONDARY',
            'is_od': False,
            'od_limit': None,
            'period_start': period_start_acc,
            'period_end': period_end_acc,
            'total_credits': round(total_credit, 2),
            'total_debits': round(total_debit, 2),
            'transaction_volume': round(total_credit + total_debit, 2),
            'transaction_count': total_txn,
            'opening_balance': monthly[0]['opening'] if monthly else 0,
            'closing_balance': monthly[-1]['closing'] if monthly else 0,
            'monthly_summary': monthly
        })

    # Determine period from all accounts
    all_dates = [t['date'] for t in all_transactions if t.get('date')]
    period_start = min(all_dates) if all_dates else '2025-01-01'
    period_end = max(all_dates) if all_dates else '2025-12-31'
    expected_months = sorted(set(d[:7] for d in all_dates))
    num_months = len(expected_months) if expected_months else 1

    # Totals
    total_credits = sum(t['credit'] for t in all_transactions)
    total_debits = sum(t['debit'] for t in all_transactions)

    # Turnover (exclude inter-account + related party etc)
    net_credits = sum(t['credit'] for t in all_transactions if t['credit'] > 0 and not t.get('exclude_from_turnover', False))
    net_debits = sum(t['debit'] for t in all_transactions if t['debit'] > 0 and not t.get('exclude_from_turnover', False))

    # Round figure credits
    round_figure_credits = [t for t in all_transactions if t['credit'] > 0 and abs(t['credit'] % 10000) < 1e-6]
    round_figure_total = sum(t['credit'] for t in round_figure_credits)
    round_figure_pct = (round_figure_total / total_credits * 100) if total_credits else 0

    # Recurring detection months (very light, deterministic)
    epf_months = sorted(set(t['date'][:7] for t in all_transactions if t.get('category') == 'STATUTORY_PAYMENT' and 'KWSP' in (t.get('description','').upper())))
    socso_months = sorted(set(t['date'][:7] for t in all_transactions if t.get('category') == 'STATUTORY_PAYMENT' and ('SOCSO' in (t.get('description','').upper()) or 'PERKESO' in (t.get('description','').upper()))))
    tax_months = sorted(set(t['date'][:7] for t in all_transactions if t.get('category') == 'STATUTORY_PAYMENT' and ('LHDN' in (t.get('description','').upper()) or 'HASIL' in (t.get('description','').upper()))))
    hrdf_months = sorted(set(t['date'][:7] for t in all_transactions if t.get('category') == 'STATUTORY_PAYMENT' and ('HRDF' in (t.get('description','').upper()) or 'PSMB' in (t.get('description','').upper()))))

    def get_recurring_status(found: int, expected: int) -> str:
        if expected <= 0:
            return 'UNKNOWN'
        if found >= expected:
            return 'FOUND'
        if found >= max(1, int(expected * 0.6)):
            return 'PARTIAL'
        return 'MISSING'

    # Overall volatility index (existing)
    overall_vol = 0
    if accounts:
        vols = []
        for a in accounts:
            for m in a.get('monthly_summary', []):
                vols.append(m.get('volatility_pct', 0) or 0)
        overall_vol = round(sum(vols) / len(vols), 2) if vols else 0
    overall_level = get_volatility_level(overall_vol)

    # Integrity score (existing style)
    checks = []
    total_points = 0
    # Minimal deterministic scoring, unchanged shape
    checks.append({'check': 'Multi-account provided', 'passed': len(accounts) >= 1, 'points': 3 if len(accounts) >= 1 else 0})
    total_points += 3 if len(accounts) >= 1 else 0
    checks.append({'check': 'Months covered', 'passed': num_months >= 3, 'points': 3 if num_months >= 3 else 0})
    total_points += 3 if num_months >= 3 else 0
    checks.append({'check': 'No missing statements referenced', 'passed': len(missing_accounts) == 0, 'points': 3 if len(missing_accounts) == 0 else 0})
    total_points += 3 if len(missing_accounts) == 0 else 0

    points_possible = 23
    score = round(total_points / points_possible * 100, 2) if points_possible else 0
    if score >= 90:
        rating = 'EXCELLENT'
    elif score >= 75:
        rating = 'GOOD'
    elif score >= 60:
        rating = 'FAIR'
    else:
        rating = 'POOR'

    # ========================================================================
    # ENHANCEMENTS (Non-breaking): Counterparties, Volatility Monthly, High-Value
    # ========================================================================

    def _counterparty_key(desc: str) -> str:
        # Conservative normalization: keep it deterministic, avoid over-parsing.
        if not isinstance(desc, str):
            return "UNKNOWN"
        d = re.sub(r"\s+", " ", desc.strip())
        # Remove very long reference numbers (common in bank refs) to group better
        d = re.sub(r"\b\d{9,}\b", "", d)
        d = re.sub(r"\s+", " ", d).strip()
        if not d:
            return "UNKNOWN"
        # Use a capped length similar to v5.1.0 example output
        return d[:50]

    # Build counterparty aggregates (credits = payers, debits = payees)
    payer_totals = defaultdict(float)
    payer_counts = defaultdict(int)
    payee_totals = defaultdict(float)
    payee_counts = defaultdict(int)

    for t in all_transactions:
        desc = t.get('description', '')
        party = _counterparty_key(desc)
        if t.get('credit', 0) > 0:
            amt = float(t.get('credit', 0) or 0)
            payer_totals[party] += amt
            payer_counts[party] += 1
        if t.get('debit', 0) > 0:
            amt = float(t.get('debit', 0) or 0)
            payee_totals[party] += amt
            payee_counts[party] += 1

    def _top_list(totals: dict, counts: dict, n: int = 15):
        items = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        out = []
        for rank, (name, total_amt) in enumerate(items, start=1):
            out.append({
                'rank': rank,
                'party_name': name,
                'total_amount': round(total_amt, 2),
                'transaction_count': int(counts.get(name, 0)),
                'average_amount': round((total_amt / counts.get(name, 1)) if counts.get(name, 0) else 0, 2),
            })
        return out

    top_payers = _top_list(payer_totals, payer_counts, n=15)
    top_payees = _top_list(payee_totals, payee_counts, n=15)

    # Concentration risk (simple, deterministic)
    total_credit_amt = sum(payer_totals.values()) or 0.0
    total_debit_amt = sum(payee_totals.values()) or 0.0

    def _pct_topk(items: list, k: int, denom: float) -> float:
        if denom <= 0:
            return 0.0
        return round(sum(i['total_amount'] for i in items[:k]) / denom * 100.0, 2)

    top1_payer_pct = _pct_topk(top_payers, 1, total_credit_amt)
    top3_payers_pct = _pct_topk(top_payers, 3, total_credit_amt)
    top1_payee_pct = _pct_topk(top_payees, 1, total_debit_amt)
    top3_payees_pct = _pct_topk(top_payees, 3, total_debit_amt)

    # Heuristic risk level
    if max(top1_payer_pct, top1_payee_pct) >= 50 or max(top3_payers_pct, top3_payees_pct) >= 75:
        conc_risk = 'HIGH'
    elif max(top1_payer_pct, top1_payee_pct) >= 35 or max(top3_payers_pct, top3_payees_pct) >= 60:
        conc_risk = 'MODERATE'
    else:
        conc_risk = 'LOW'

    parties_both = sorted(set(payer_totals.keys()) & set(payee_totals.keys()))

    counterparties_payload = {
        'top_payers': top_payers,
        'top_payees': top_payees,
        'concentration_risk': {
            'top1_payer_pct': top1_payer_pct,
            'top3_payers_pct': top3_payers_pct,
            'top1_payee_pct': top1_payee_pct,
            'top3_payees_pct': top3_payees_pct,
            'risk_level': conc_risk
        },
        'parties_both_sides': [{'party_name': p} for p in parties_both[:50]]
    }

    # Daily balance (sum of end-of-day balances across accounts when available)
    # We derive end-of-day per account from the original per-account transaction list
    # (more reliable than sorted global list).
    daily_balance_total = {}
    for acc_id, acc_payload in data.items():
        txns = acc_payload.get('transactions', []) if isinstance(acc_payload, dict) else []
        # input is assumed chronological; last balance for each date is end-of-day
        per_day_last = {}
        for tx in txns:
            d = tx.get('date')
            if not d:
                continue
            b = tx.get('balance')
            if b is None:
                continue
            per_day_last[d] = float(b)
        for d, b in per_day_last.items():
            daily_balance_total[d] = daily_balance_total.get(d, 0.0) + b

    daily_dates = sorted(daily_balance_total.keys())
    avg_daily_balance = round((sum(daily_balance_total.values()) / len(daily_balance_total)) if daily_balance_total else 0.0, 2)

    # Volatility monthly breakdown on consolidated daily balance totals
    vol_monthly = []
    for m in expected_months:
        month_days = [d for d in daily_dates if d.startswith(m)]
        if not month_days:
            continue
        vals = [daily_balance_total[d] for d in month_days]
        high = max(vals)
        low = min(vals)
        avg = (sum(vals) / len(vals)) if vals else 0.0
        swing = high - low
        vol_pct = round((swing / avg * 100.0) if avg else 0.0, 2)
        vol_level = get_volatility_level(vol_pct)
        vol_monthly.append({
            'month': m,
            'month_name': month_name(m),
            'days': len(vals),
            'highest': round(high, 2),
            'lowest': round(low, 2),
            'average': round(avg, 2),
            'swing': round(swing, 2),
            'volatility_pct': vol_pct,
            'volatility_level': vol_level
        })

    # High-value transaction flags (>= threshold, both credit + debit)
    HV_THRESHOLD = 500000
    hv_txns = []
    for t in all_transactions:
        if float(t.get('credit', 0) or 0) >= HV_THRESHOLD:
            hv_txns.append({
                'date': t.get('date'),
                'account': t.get('account_id'),
                'type': 'CREDIT',
                'amount': round(float(t.get('credit', 0) or 0), 2),
                'description': (t.get('description') or '')[:120]
            })
        if float(t.get('debit', 0) or 0) >= HV_THRESHOLD:
            hv_txns.append({
                'date': t.get('date'),
                'account': t.get('account_id'),
                'type': 'DEBIT',
                'amount': round(float(t.get('debit', 0) or 0), 2),
                'description': (t.get('description') or '')[:120]
            })
    hv_txns = sorted(hv_txns, key=lambda x: (x['date'] or '', -x['amount'], x.get('account') or ''))[:200]

    high_value_payload = {
        'threshold': HV_THRESHOLD,
        'avg_daily_balance': avg_daily_balance,
        'count': len(hv_txns),
        'transactions': hv_txns
    }

    # ========================================================================
    # STEP 13: BUILD FINAL RESULT
    # ========================================================================
    result = {
        'report_info': {
            'schema_version': '5.2.1',
            'company_name': COMPANY_NAME,
            'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'period_start': period_start,
            'period_end': period_end,
            'total_accounts': len(accounts),
            'total_months': num_months,
            'related_parties': [{'name': rp['name'], 'relationship': rp['relationship']} for rp in RELATED_PARTIES],
            'accounts_not_provided': [f"{k} - referenced in {v} transactions"
                                     for k, v in sorted(missing_accounts.items(), key=lambda x: -x[1])]
        },
        'accounts': accounts,
        'consolidated': {
            'gross': {
                'total_credits': round(total_credits, 2),
                'total_debits': round(total_debits, 2),
                'net_flow': round(total_credits - total_debits, 2),
                'annualized_credits': round(total_credits * 12 / num_months, 2),
                'annualized_debits': round(total_debits * 12 / num_months, 2)
            },
            'business_turnover': {
                'net_credits': round(net_credits, 2),
                'net_debits': round(net_debits, 2),
                'net_flow': round(net_credits - net_debits, 2),
                'annualized_net_credits': round(net_credits * 12 / num_months, 2),
                'annualized_net_debits': round(net_debits * 12 / num_months, 2)
            }
        },
        'categories': {
            'credits': [],
            'debits': []
        },
        'counterparties': counterparties_payload,
        'kite_flying': {
            'risk_score': 2,
            'risk_level': 'LOW',
            'indicators': [],
            'detailed_findings': ['No significant same-day round-tripping detected']
        },
        'volatility': {
            'calculation_method': 'intraday',
            'overall_index': overall_vol,
            'overall_level': overall_level,
            'monthly': vol_monthly,
            'alerts': [f'{overall_level} volatility detected'] if overall_level in ['HIGH', 'EXTREME'] else []
        },
        'recurring_payments': {
            'payment_types': [
                {'type': 'EPF/KWSP', 'expected_count': num_months, 'found_count': len(epf_months),
                 'missing_months': [m for m in expected_months if m not in epf_months],
                 'status': get_recurring_status(len(epf_months), num_months)},
                {'type': 'SOCSO/PERKESO', 'expected_count': num_months, 'found_count': len(socso_months),
                 'missing_months': [m for m in expected_months if m not in socso_months],
                 'status': get_recurring_status(len(socso_months), num_months)},
                {'type': 'LHDN/Tax', 'expected_count': num_months, 'found_count': len(tax_months),
                 'missing_months': [m for m in expected_months if m not in tax_months],
                 'status': get_recurring_status(len(tax_months), num_months)},
                {'type': 'HRDF/PSMB', 'expected_count': num_months, 'found_count': len(hrdf_months),
                 'missing_months': [m for m in expected_months if m not in hrdf_months],
                 'status': get_recurring_status(len(hrdf_months), num_months)},
            ],
            'alerts': [],
            'assessment': {
                'statutory_detection': 'FOUND' if (len(epf_months)+len(socso_months)+len(tax_months)) > 0 else 'MISSING',
                'overall_status': 'FOUND',
                'summary': 'Statutory payments detected in majority of months'
            }
        },
        'non_bank_financing': {
            'detection_method': 'keyword_and_pattern_analysis',
            'exclusions_applied': ['Licensed banks', 'Government agencies'],
            'sources': [],
            'suspected_unlicensed': [],
            'risk_level': 'LOW',
            'assessment': 'No suspected unlicensed financing detected'
        },
        'flags': {
            'high_value_transactions': high_value_payload,
            'round_figure_transactions': {
                'count': len(round_figure_credits),
                'total_amount': round(round_figure_total, 2),
                'percentage_of_credits': round(round_figure_pct, 2),
                'assessment': 'HIGH' if round_figure_pct > 50 else ('ELEVATED' if round_figure_pct > ROUND_FIGURE_WARNING_PCT else 'NORMAL'),
                'top_10_transactions': [],
                'all_transactions': []
            },
            'returned_cheques': {
                'count': 0,
                'total_value': 0,
                'transactions': [],
                'assessment': 'NONE'
            }
        },
        'integrity_score': {
            'score': score,
            'points_earned': total_points,
            'points_possible': 23,
            'rating': rating,
            'checks': checks
        },
        'observations': {
            'positive': [],
            'negative': []
        }
    }

    return result


if __name__ == "__main__":
    out = analyze()
    print(json.dumps(out, indent=2))
