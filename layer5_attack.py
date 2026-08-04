"""
Layer 5: Informed Date-Shift Attack — Per-Event-Type Weekday Probabilities
==========================================================================
Extends Layer 4 by giving the attacker knowledge of event types.

Instead of a single p_weekday=0.98 for all dates, the informed attacker
assigns each date a weekday probability based on the most constraining
event type observed on that date:

  - Office Visit:       p=0.996
  - Ancillary Procedure: p=1.000
  - ED visit:           p=0.735
  - etc.

This is realistic because event types (ENC_TYPE, CPT codes, LOINC codes)
are NOT protected health information and appear in deidentified datasets.

The script:
  1. Computes empirical weekday rates per event type from the extracts
  2. For each patient-date, assigns p_weekday = max across all events
  3. Runs the Bayesian attack with per-date likelihoods
  4. Outputs both Layer 4 (naive) and Layer 5 (informed) metrics for comparison

Usage:
  python layer5_attack.py [data_dir] --shift 30 365 --workers 16
  python layer5_attack.py . --shift 30 --sample 10000 --workers 8

Reads: extract_outpatient_encounters.csv, extract_outpatient_labs.csv,
       extract_outpatient_procedures.csv
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
import os
import sys
import argparse
import time
from collections import defaultdict


# ── Holiday calendar (same as Layer 4) ─────────────────────────────────────

def _nth_weekday(year, month, n, weekday):
    if n > 0:
        d = date(year, month, 1)
        delta = (weekday - d.weekday()) % 7
        first = d + timedelta(days=delta)
        return first + timedelta(weeks=n - 1)
    else:
        if month == 12:
            d = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            d = date(year, month + 1, 1) - timedelta(days=1)
        delta = (d.weekday() - weekday) % 7
        return d - timedelta(days=delta) + timedelta(weeks=n + 1 if n < -1 else 0)


def generate_holiday_ordinals(year_start, year_end):
    holidays = set()
    for yr in range(year_start, year_end + 1):
        dates = [
            date(yr, 1, 1),
            _nth_weekday(yr, 5, -1, 0),
            date(yr, 7, 4),
            _nth_weekday(yr, 9, 1, 0),
            _nth_weekday(yr, 11, 4, 3),
            _nth_weekday(yr, 11, 4, 3) + timedelta(1),
            date(yr, 12, 25),
        ]
        for d in dates:
            holidays.add(d.toordinal())
    return holidays


# ── Step 1: Compute empirical weekday rates ────────────────────────────────

def compute_weekday_rates(data_dir):
    """
    Compute empirical weekday fraction per event type from extract CSVs.
    Returns dict: event_type_string -> p_weekday (float in [0, 1])
    """
    rates = {}

    # ── Encounters: by RAW_ENC_TYPE (most granular)
    enc_file = os.path.join(data_dir, 'extract_outpatient_encounters.csv')
    if os.path.exists(enc_file):
        print("  Computing encounter weekday rates...")
        df = pd.read_csv(enc_file, usecols=['event_date', 'ENC_TYPE', 'RAW_ENC_TYPE'],
                         parse_dates=['event_date'])
        df['dow'] = df['event_date'].dt.dayofweek  # 0=Mon, 6=Sun
        df['is_weekday'] = df['dow'] < 5

        # RAW_ENC_TYPE level (most informative)
        raw_rates = df.groupby('RAW_ENC_TYPE')['is_weekday'].agg(['mean', 'count'])
        for raw_type, row in raw_rates.iterrows():
            if row['count'] >= 100:  # only trust rates with enough data
                key = f"RAW:{raw_type}"
                rates[key] = max(0.50, min(0.9999, row['mean']))

        # ENC_TYPE level (fallback)
        enc_rates = df.groupby('ENC_TYPE')['is_weekday'].agg(['mean', 'count'])
        for enc_type, row in enc_rates.iterrows():
            if row['count'] >= 100:
                key = f"ENC:{enc_type}"
                rates[key] = max(0.50, min(0.9999, row['mean']))

        print(f"    {len([k for k in rates if k.startswith('RAW:')])} RAW_ENC_TYPEs, "
              f"{len([k for k in rates if k.startswith('ENC:')])} ENC_TYPEs")
        del df

    # ── Labs: by lab_name (or LOINC if available)
    lab_file = os.path.join(data_dir, 'extract_outpatient_labs.csv')
    if os.path.exists(lab_file):
        print("  Computing lab weekday rates...")
        # Try to load with different possible column names
        try:
            df = pd.read_csv(lab_file, usecols=['order_date', 'lab_name'],
                             parse_dates=['order_date'])
            date_col = 'order_date'
            type_col = 'lab_name'
        except (ValueError, KeyError):
            try:
                df = pd.read_csv(lab_file, usecols=['order_date', 'LAB_LOINC'],
                                 parse_dates=['order_date'])
                date_col = 'order_date'
                type_col = 'LAB_LOINC'
            except (ValueError, KeyError):
                df = pd.read_csv(lab_file, usecols=['order_date'],
                                 parse_dates=['order_date'])
                date_col = 'order_date'
                type_col = None

        df['dow'] = df[date_col].dt.dayofweek
        df['is_weekday'] = df['dow'] < 5

        if type_col and type_col in df.columns:
            lab_rates = df.groupby(type_col)['is_weekday'].agg(['mean', 'count'])
            n_labs = 0
            for lab_type, row in lab_rates.iterrows():
                if row['count'] >= 100:
                    key = f"LAB:{lab_type}"
                    rates[key] = max(0.50, min(0.9999, row['mean']))
                    n_labs += 1
            print(f"    {n_labs} lab types")
        else:
            # Global lab rate
            rates['LAB:_global'] = max(0.50, min(0.9999, df['is_weekday'].mean()))
            print(f"    Global lab rate: {rates['LAB:_global']:.4f}")
        del df

    # ── Procedures: global rate (could be extended per PX code)
    px_file = os.path.join(data_dir, 'extract_outpatient_procedures.csv')
    if os.path.exists(px_file):
        print("  Computing procedure weekday rates...")
        try:
            df = pd.read_csv(px_file, usecols=['event_date', 'RAW_PX'],
                             parse_dates=['event_date'])
            type_col = 'RAW_PX'
        except (ValueError, KeyError):
            try:
                df = pd.read_csv(px_file, usecols=['event_date', 'PX'],
                                 parse_dates=['event_date'])
                type_col = 'PX'
            except (ValueError, KeyError):
                df = pd.read_csv(px_file, usecols=['event_date'],
                                 parse_dates=['event_date'])
                type_col = None

        df['dow'] = df['event_date'].dt.dayofweek
        df['is_weekday'] = df['dow'] < 5

        if type_col and type_col in df.columns:
            px_rates = df.groupby(type_col)['is_weekday'].agg(['mean', 'count'])
            n_px = 0
            for px_type, row in px_rates.iterrows():
                if row['count'] >= 100:
                    key = f"PX:{px_type}"
                    rates[key] = max(0.50, min(0.9999, row['mean']))
                    n_px += 1
            print(f"    {n_px} procedure types")
        else:
            rates['PX:_global'] = max(0.50, min(0.9999, df['is_weekday'].mean()))
            print(f"    Global procedure rate: {rates['PX:_global']:.4f}")
        del df

    print(f"  Total event types with rates: {len(rates)}")

    # Summary statistics
    if rates:
        vals = list(rates.values())
        print(f"  p_weekday range: {min(vals):.4f} – {max(vals):.4f}")
        print(f"  p_weekday median: {np.median(vals):.4f}")
        n_perfect = sum(1 for v in vals if v >= 0.999)
        n_high = sum(1 for v in vals if v >= 0.95)
        n_low = sum(1 for v in vals if v < 0.85)
        print(f"  ≥99.9% weekday: {n_perfect},  ≥95%: {n_high},  <85%: {n_low}")

    return rates


# ── Step 2: Build per-patient (date, p_weekday) arrays ─────────────────────

def build_patient_date_arrays_informed(data_dir, rates, scenario='combined_order'):
    """
    Load extracts, build per-patient arrays of (date_ordinal, p_weekday).

    For each patient-date, p_weekday is the MAX across all event types
    observed on that date (most constraining signal for the attacker).

    Returns dict: { PATID: (ordinals_array, p_weekday_array) }
    Also returns a "naive" version with uniform p_weekday for comparison.
    """
    print(f"\nBuilding informed patient arrays (scenario: {scenario})...")

    # Store: patid -> { ordinal -> max_p_weekday }
    patient_dates = defaultdict(dict)  # patid -> {ordinal: best_p_weekday}

    # Default p_weekday for types not in the rates dict
    DEFAULT_P = 0.98

    # ── Encounters
    enc_file = os.path.join(data_dir, 'extract_outpatient_encounters.csv')
    if os.path.exists(enc_file):
        print("  Loading encounters with types...", end=' ', flush=True)
        df = pd.read_csv(enc_file,
                         usecols=['PATID', 'event_date', 'ENC_TYPE', 'RAW_ENC_TYPE'],
                         parse_dates=['event_date'])
        df = df.dropna(subset=['event_date'])

        # Map each row to its p_weekday via vectorized lookup
        df['raw_key'] = 'RAW:' + df['RAW_ENC_TYPE'].astype(str)
        df['enc_key'] = 'ENC:' + df['ENC_TYPE'].astype(str)
        df['p'] = df['raw_key'].map(rates).fillna(df['enc_key'].map(rates)).fillna(DEFAULT_P)
        df['ordinal'] = df['event_date'].apply(lambda x: x.toordinal())

        # For each (PATID, ordinal), keep max p
        best = df.groupby(['PATID', 'ordinal'])['p'].max().reset_index()
        for patid, ordinal, p in zip(best['PATID'].values, best['ordinal'].values, best['p'].values):
            if ordinal not in patient_dates[patid] or p > patient_dates[patid].get(ordinal, 0):
                patient_dates[patid][ordinal] = p

        print(f"{len(df):,} events → {len(best):,} patient-dates")
        del df, best

    # ── Labs
    lab_file = os.path.join(data_dir, 'extract_outpatient_labs.csv')
    if os.path.exists(lab_file):
        date_col = 'order_date' if 'order' in scenario else 'result_date'
        print(f"  Loading labs with types ({date_col})...", end=' ', flush=True)

        sample = pd.read_csv(lab_file, nrows=1)
        lab_cols = ['PATID', date_col]
        type_col = None
        for candidate in ['lab_name', 'LAB_LOINC']:
            if candidate in sample.columns:
                type_col = candidate
                lab_cols.append(type_col)
                break

        df = pd.read_csv(lab_file, usecols=lab_cols, parse_dates=[date_col])
        df = df.dropna(subset=[date_col])
        df['ordinal'] = df[date_col].apply(lambda x: x.toordinal())

        if type_col and type_col in df.columns:
            df['lab_key'] = 'LAB:' + df[type_col].astype(str)
            global_lab = rates.get('LAB:_global', DEFAULT_P)
            df['p'] = df['lab_key'].map(rates).fillna(global_lab)
        else:
            df['p'] = rates.get('LAB:_global', DEFAULT_P)

        best = df.groupby(['PATID', 'ordinal'])['p'].max().reset_index()
        for patid, ordinal, p in zip(best['PATID'].values, best['ordinal'].values, best['p'].values):
            if ordinal not in patient_dates[patid] or p > patient_dates[patid].get(ordinal, 0):
                patient_dates[patid][ordinal] = p

        print(f"{len(df):,} events → {len(best):,} patient-dates")
        del df, best

    # ── Procedures
    px_file = os.path.join(data_dir, 'extract_outpatient_procedures.csv')
    if os.path.exists(px_file):
        print(f"  Loading procedures with types...", end=' ', flush=True)

        sample = pd.read_csv(px_file, nrows=1)
        px_cols = ['PATID', 'event_date']
        type_col = None
        for candidate in ['RAW_PX', 'PX']:
            if candidate in sample.columns:
                type_col = candidate
                px_cols.append(type_col)
                break

        df = pd.read_csv(px_file, usecols=px_cols, parse_dates=['event_date'])
        df = df.dropna(subset=['event_date'])
        df['ordinal'] = df['event_date'].apply(lambda x: x.toordinal())

        if type_col and type_col in df.columns:
            df['px_key'] = 'PX:' + df[type_col].astype(str)
            global_px = rates.get('PX:_global', DEFAULT_P)
            df['p'] = df['px_key'].map(rates).fillna(global_px)
        else:
            df['p'] = rates.get('PX:_global', DEFAULT_P)

        best = df.groupby(['PATID', 'ordinal'])['p'].max().reset_index()
        for patid, ordinal, p in zip(best['PATID'].values, best['ordinal'].values, best['p'].values):
            if ordinal not in patient_dates[patid] or p > patient_dates[patid].get(ordinal, 0):
                patient_dates[patid][ordinal] = p

        print(f"{len(df):,} events → {len(best):,} patient-dates")
        del df, best

    # ── Convert to arrays, keep patients with >= 2 dates
    patient_arrays = {}
    for patid, date_dict in patient_dates.items():
        if len(date_dict) >= 2:
            sorted_ords = sorted(date_dict.keys())
            ordinals = np.array(sorted_ords, dtype=np.int32)
            p_weekdays = np.array([date_dict[o] for o in sorted_ords], dtype=np.float64)
            patient_arrays[patid] = (ordinals, p_weekdays)

    print(f"  Patients with >=2 distinct dates: {len(patient_arrays):,}")

    # ── Summary of per-date p_weekday distribution
    all_ps = np.concatenate([pa[1] for pa in patient_arrays.values()])
    print(f"  Per-date p_weekday: mean={all_ps.mean():.4f}, "
          f"median={np.median(all_ps):.4f}, "
          f"min={all_ps.min():.4f}, max={all_ps.max():.4f}")
    print(f"  Dates at ≥99%: {(all_ps >= 0.99).sum():,} ({100*(all_ps >= 0.99).mean():.1f}%)")
    print(f"  Dates at ≥99.9%: {(all_ps >= 0.999).sum():,} ({100*(all_ps >= 0.999).mean():.1f}%)")
    print(f"  Dates at <90%: {(all_ps < 0.90).sum():,} ({100*(all_ps < 0.90).mean():.1f}%)")

    return patient_arrays


# ── Core attack — informed (per-date p_weekday) ───────────────────────────

_holiday_arr = None
_p_holiday = None
_global_p_weekday = None   # for naive comparison

def _init_worker_l5(holiday_ordinals_list, p_holiday, global_p_weekday):
    global _holiday_arr, _p_holiday, _global_p_weekday
    _holiday_arr = np.array(holiday_ordinals_list, dtype=np.int32)
    _p_holiday = p_holiday
    _global_p_weekday = global_p_weekday


def attack_patient_informed(args):
    """
    Run BOTH naive and informed attacks for one patient.

    args: (patid, date_ordinals, p_weekday_per_date, shift_range)

    Returns dict with both naive and informed vulnerability metrics.
    """
    patid, date_ordinals, p_weekdays, shift_range = args

    n_dates = len(date_ordinals)
    span_days = int(date_ordinals[-1] - date_ordinals[0])

    # ── 1. Apply random backward shift (same seed as Layer 4 for comparability)
    rng = np.random.default_rng(hash(patid) & 0xFFFFFFFF)
    s_true = int(rng.integers(1, shift_range + 1))
    shifted = date_ordinals - s_true

    # ── 2. Candidate shifts
    candidates = np.arange(1, shift_range + 1, dtype=np.int32)
    candidate_originals = shifted[None, :] + candidates[:, None]  # (S, k)

    # Day-of-week
    dow = (candidate_originals - 1) % 7
    is_weekend = dow >= 5  # (S, k)

    # Holiday check
    if len(_holiday_arr) > 0:
        is_holiday = np.isin(candidate_originals, _holiday_arr)  # (S, k)
    else:
        is_holiday = np.zeros_like(is_weekend)

    # ── 3. Hard model (same for both — uses perfect weekday assumption)
    n_weekend_per_candidate = is_weekend.sum(axis=1)
    survives_dow = n_weekend_per_candidate == 0
    n_survivors_dow = int(survives_dow.sum())

    survives_all = survives_dow.copy()
    if len(_holiday_arr) > 0:
        for idx in np.where(survives_dow)[0]:
            if np.any(is_holiday[idx]):
                survives_all[idx] = False
    n_survivors_hol = int(survives_all.sum())

    max_entropy = np.log2(shift_range)
    entropy_hard = np.log2(n_survivors_hol) if n_survivors_hol > 0 else 0.0
    p_true_hard = 1.0 / n_survivors_hol if n_survivors_hol > 0 else 0.0

    true_idx = s_true - 1
    true_survived = bool(survives_all[true_idx])
    shift_recovered = (n_survivors_hol == 1) and true_survived

    # ── 4. Soft model — NAIVE (uniform p_weekday)
    log_pwd_naive = np.log(_global_p_weekday)
    log_pwe_naive = np.log(1.0 - _global_p_weekday)
    log_phol = np.log(_p_holiday)

    ll_naive = np.where(is_weekend, log_pwe_naive, log_pwd_naive).sum(axis=1)
    if is_holiday.any():
        current_naive = np.where(is_weekend, log_pwe_naive, log_pwd_naive)
        ll_naive += ((log_phol - current_naive) * is_holiday).sum(axis=1)

    ll_max = ll_naive.max()
    post_naive = np.exp(ll_naive - ll_max)
    post_naive /= post_naive.sum()

    mask_pos = post_naive > 0
    entropy_naive = float(-np.sum(post_naive[mask_pos] * np.log2(post_naive[mask_pos])))
    p_true_naive = float(post_naive[true_idx])
    p_max_naive = float(post_naive.max())
    argmax_naive = int(np.argmax(post_naive)) + 1
    max_is_true_naive = (argmax_naive == s_true)

    # ── 5. Soft model — INFORMED (per-date p_weekday)
    # log P(date_i | m) depends on event type via p_weekdays[i]
    # Shape: p_weekdays is (k,), we need (S, k) log-likelihoods

    log_pwd_informed = np.log(p_weekdays)           # (k,)
    log_pwe_informed = np.log(1.0 - p_weekdays)     # (k,)

    # Base likelihood: weekday or weekend, per-date
    ll_informed = np.where(is_weekend,
                           log_pwe_informed[None, :],   # broadcast (S, k)
                           log_pwd_informed[None, :]).sum(axis=1)  # sum over dates -> (S,)

    # Holiday adjustment
    if is_holiday.any():
        current_informed = np.where(is_weekend,
                                    log_pwe_informed[None, :],
                                    log_pwd_informed[None, :])
        ll_informed += ((log_phol - current_informed) * is_holiday).sum(axis=1)

    ll_max = ll_informed.max()
    post_informed = np.exp(ll_informed - ll_max)
    post_informed /= post_informed.sum()

    mask_pos = post_informed > 0
    entropy_informed = float(-np.sum(post_informed[mask_pos] * np.log2(post_informed[mask_pos])))
    p_true_informed = float(post_informed[true_idx])
    p_max_informed = float(post_informed.max())
    argmax_informed = int(np.argmax(post_informed)) + 1
    max_is_true_informed = (argmax_informed == s_true)

    effective_naive = 2.0 ** entropy_naive
    effective_informed = 2.0 ** entropy_informed

    # ── 6. Per-patient p_weekday summary
    mean_p = float(p_weekdays.mean())
    min_p = float(p_weekdays.min())
    max_p = float(p_weekdays.max())

    return {
        'PATID': patid,
        'shift_range': shift_range,
        's_true': s_true,
        'n_dates': n_dates,
        'span_days': span_days,
        'mean_p_weekday': round(mean_p, 4),
        'min_p_weekday': round(min_p, 4),
        'max_p_weekday': round(max_p, 4),
        # Hard model (shared)
        'n_survivors_dow': n_survivors_dow,
        'n_survivors_holiday': n_survivors_hol,
        'entropy_hard': round(entropy_hard, 4),
        'p_true_hard': round(p_true_hard, 6),
        'shift_recovered': shift_recovered,
        'true_survived': true_survived,
        # Naive soft model (Layer 4 equivalent)
        'entropy_naive': round(entropy_naive, 4),
        'p_true_naive': round(p_true_naive, 6),
        'p_max_naive': round(p_max_naive, 6),
        'max_is_true_naive': max_is_true_naive,
        'effective_naive': round(effective_naive, 2),
        # Informed soft model (Layer 5)
        'entropy_informed': round(entropy_informed, 4),
        'p_true_informed': round(p_true_informed, 6),
        'p_max_informed': round(p_max_informed, 6),
        'max_is_true_informed': max_is_true_informed,
        'effective_informed': round(effective_informed, 2),
        # Improvement
        'entropy_reduction': round(entropy_naive - entropy_informed, 4),
        'p_max_lift': round(p_max_informed - p_max_naive, 6),
        # Baseline
        'entropy_before': round(max_entropy, 4),
    }


# ── Summary ───────────────────────────────────────────────────────────────

def summarize_results(df, shift_range):
    n = len(df)
    max_ent = np.log2(shift_range)

    print(f"\n{'='*80}")
    print(f"SHIFT RANGE: {shift_range} days  |  Patients: {n:,}  |  Max entropy: {max_ent:.2f} bits")
    print(f"{'='*80}")

    # ── Comparison table
    print(f"\n  {'Metric':<42} {'Naive':>12} {'Informed':>12} {'Delta':>10}")
    print(f"  {'-'*78}")

    metrics = [
        ('Mean entropy (bits)',
         df['entropy_naive'].mean(), df['entropy_informed'].mean()),
        ('Mean P(true shift)',
         df['p_true_naive'].mean(), df['p_true_informed'].mean()),
        ('Mean P(max)',
         df['p_max_naive'].mean(), df['p_max_informed'].mean()),
        ('Argmax = true (%)',
         100*df['max_is_true_naive'].mean(), 100*df['max_is_true_informed'].mean()),
        ('Mean effective candidates',
         df['effective_naive'].mean(), df['effective_informed'].mean()),
    ]

    for name, naive, informed in metrics:
        delta = informed - naive
        sign = '+' if delta >= 0 else ''
        print(f"  {name:<42} {naive:>12.4f} {informed:>12.4f} {sign}{delta:>9.4f}")

    # ── Selective attacker comparison
    print(f"\n  SELECTIVE ATTACKER (threshold → precision, yield)")
    print(f"  {'Threshold':<12} {'--- Naive ---':>28} {'--- Informed ---':>28}")
    print(f"  {'':12} {'Targeted':>9} {'Correct':>9} {'Prec%':>7}  {'Targeted':>9} {'Correct':>9} {'Prec%':>7}")
    print(f"  {'-'*78}")

    for tau in [0.50, 0.75, 0.90, 0.95, 0.99]:
        for model, pm_col, mt_col in [('naive', 'p_max_naive', 'max_is_true_naive'),
                                       ('informed', 'p_max_informed', 'max_is_true_informed')]:
            tgt = df[df[pm_col] >= tau]
            n_tgt = len(tgt)
            n_corr = tgt[mt_col].sum()
            prec = 100*n_corr/n_tgt if n_tgt > 0 else 0
            if model == 'naive':
                print(f"  ≥{tau:<10.2f} {n_tgt:>9,} {n_corr:>9,} {prec:>6.1f}%", end='')
            else:
                print(f"  {n_tgt:>9,} {n_corr:>9,} {prec:>6.1f}%")

    # ── Vulnerability tiers comparison
    print(f"\n  VULNERABILITY TIERS")
    print(f"  {'Tier':<35} {'Naive':>12} {'Informed':>12}")
    print(f"  {'-'*60}")
    for name, lo, hi in [('CRITICAL (≥0.50)', 0.50, 1.01),
                          ('HIGH (0.25–0.50)', 0.25, 0.50),
                          ('MODERATE (0.10–0.25)', 0.10, 0.25),
                          ('LOW (<0.10)', 0.00, 0.10)]:
        n_naive = ((df['p_max_naive'] >= lo) & (df['p_max_naive'] < hi)).sum()
        n_informed = ((df['p_max_informed'] >= lo) & (df['p_max_informed'] < hi)).sum()
        print(f"  {name:<35} {n_naive:>8,} ({100*n_naive/n:>5.1f}%) "
              f"{n_informed:>8,} ({100*n_informed/n:>5.1f}%)")

    # ── Stratify by date count
    bins = [2, 4, 6, 10, 17, 20, 30, 60, 100, 100000]
    labels = ['2-3', '4-5', '6-9', '10-16', '17-19', '20-29', '30-59', '60-99', '100+']
    df_c = df.copy()
    df_c['date_bin'] = pd.cut(df_c['n_dates'], bins=bins, labels=labels, right=False)

    print(f"\n  BY DATE COUNT: Informed attacker advantage")
    print(f"  {'Dates':<8} {'N':>9} {'P(max)N':>9} {'P(max)I':>9} "
          f"{'Max=T% N':>9} {'Max=T% I':>9} {'Lift':>7}")
    print(f"  {'-'*62}")

    for label in labels:
        g = df_c[df_c['date_bin'] == label]
        if len(g) == 0:
            continue
        pm_n = g['p_max_naive'].mean()
        pm_i = g['p_max_informed'].mean()
        mt_n = 100*g['max_is_true_naive'].mean()
        mt_i = 100*g['max_is_true_informed'].mean()
        lift = mt_i - mt_n
        print(f"  {label:<8} {len(g):>9,} {pm_n:>9.4f} {pm_i:>9.4f} "
              f"{mt_n:>8.1f}% {mt_i:>8.1f}% {lift:>+6.1f}%")

    # ── Patients where informed attacker succeeds but naive fails
    informed_only = (df['max_is_true_informed'] & ~df['max_is_true_naive']).sum()
    naive_only = (df['max_is_true_naive'] & ~df['max_is_true_informed']).sum()
    both = (df['max_is_true_naive'] & df['max_is_true_informed']).sum()
    neither = (~df['max_is_true_naive'] & ~df['max_is_true_informed']).sum()
    print(f"\n  RECOVERY OVERLAP")
    print(f"    Both correct:          {both:>10,} ({100*both/n:.2f}%)")
    print(f"    Informed only:         {informed_only:>10,} ({100*informed_only/n:.2f}%)")
    print(f"    Naive only:            {naive_only:>10,} ({100*naive_only/n:.2f}%)")
    print(f"    Neither:               {neither:>10,} ({100*neither/n:.2f}%)")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Layer 5: Informed date-shift attack (per-event-type)')
    parser.add_argument('data_dir', nargs='?',
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Directory with extract_*.csv files')
    parser.add_argument('--shift', nargs='+', type=int,
                        default=[30, 365],
                        help='Shift ranges to test (days)')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()),
                        help='Parallel workers')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample N patients (0=all)')
    parser.add_argument('--scenario', default='combined_order',
                        choices=['combined_order', 'combined_result'],
                        help='Lab date column')
    parser.add_argument('--p-holiday', type=float, default=0.001,
                        help='P(event on institutional holiday)')
    parser.add_argument('--p-weekday-naive', type=float, default=0.98,
                        help='Naive model: uniform P(weekday)')
    args = parser.parse_args()

    output_dir = args.data_dir

    print("=" * 60)
    print("LAYER 5: INFORMED DATE-SHIFT ATTACK")
    print("=" * 60)
    print(f"  Naive baseline: p_weekday={args.p_weekday_naive}")
    print(f"  Holiday: p_holiday={args.p_holiday}")

    # Step 1: Compute empirical weekday rates
    print(f"\n── Step 1: Empirical weekday rates ──")
    rates = compute_weekday_rates(args.data_dir)

    # Save rates for reference
    rates_df = pd.DataFrame([
        {'event_type': k, 'p_weekday': v} for k, v in sorted(rates.items())
    ])
    rates_file = os.path.join(output_dir, 'layer5_weekday_rates.csv')
    rates_df.to_csv(rates_file, index=False)
    print(f"  Saved: {rates_file}")

    # Step 2: Build informed patient arrays
    print(f"\n── Step 2: Build patient arrays ──")
    patient_arrays = build_patient_date_arrays_informed(
        args.data_dir, rates, args.scenario)

    if args.sample > 0 and args.sample < len(patient_arrays):
        rng = np.random.default_rng(42)
        keys = rng.choice(list(patient_arrays.keys()), args.sample, replace=False)
        patient_arrays = {k: patient_arrays[k] for k in keys}
        print(f"  Sampled {args.sample:,} patients")

    # Holiday calendar
    all_ord = np.concatenate([pa[0] for pa in patient_arrays.values()])
    min_year = date.fromordinal(int(all_ord.min())).year - 6
    max_year = date.fromordinal(int(all_ord.max())).year + 1
    holiday_ordinals = generate_holiday_ordinals(min_year, max_year)
    holiday_list = sorted(holiday_ordinals)
    print(f"  Holiday calendar: {min_year}–{max_year} ({len(holiday_list)} dates)")

    patid_list = list(patient_arrays.keys())

    # Step 3: Run attacks
    for shift_range in args.shift:
        print(f"\n{'>'*60}")
        print(f"  Shift range: {shift_range} days | "
              f"{len(patid_list):,} patients | {args.workers} workers")
        print(f"{'>'*60}")

        work_items = [
            (pid, patient_arrays[pid][0], patient_arrays[pid][1], shift_range)
            for pid in patid_list
        ]

        t0 = time.time()
        with Pool(
            processes=args.workers,
            initializer=_init_worker_l5,
            initargs=(holiday_list, args.p_holiday, args.p_weekday_naive),
        ) as pool:
            results = pool.map(attack_patient_informed, work_items, chunksize=500)

        elapsed = time.time() - t0
        rate = len(results) / elapsed
        print(f"  Completed in {elapsed:.1f}s ({rate:,.0f} patients/sec)")

        df_res = pd.DataFrame(results)

        outfile = os.path.join(output_dir, f'layer5_attack_{shift_range}d.csv')
        df_res.to_csv(outfile, index=False)
        print(f"  Saved: {outfile}")

        summarize_results(df_res, shift_range)

        del df_res, results

    print("\n=== Done. ===")


if __name__ == '__main__':
    main()
