"""
Layer 4: Simulated Date-Shift Attack with Probabilistic Vulnerability
=====================================================================
For each patient, computes a full posterior distribution over candidate
shifts, yielding per-patient vulnerability metrics.

Two complementary models:
  HARD: Binary elimination — any candidate mapping a date to weekend or
        holiday is completely eliminated.  Survivors are equally likely.
  SOFT: Bayesian likelihood — each candidate gets a probability based on
        how well reconstructed dates match expected weekday/holiday patterns.
        Candidates mapping to weekends are penalized but not eliminated.

Per-patient metrics:
  - n_survivors: candidates surviving hard elimination
  - entropy_hard / entropy_soft: attacker's remaining uncertainty (bits)
  - info_gain: bits learned by the attack
  - p_true_hard / p_true_soft: probability assigned to the true shift
  - p_max_soft: probability of the most likely candidate (may != true)
  - max_is_true: whether the argmax of the posterior IS the true shift
  - shift_recovered: unique recovery under hard model

Usage:
  python layer4_attack.py [data_dir] [--shift 14 30 365 1826] [--workers 16]
  python layer4_attack.py . --shift 30 --sample 10000 --workers 8   # quick test

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


# ── US Holiday Calendar ────────────────────────────────────────────────────

def _nth_weekday(year, month, n, weekday):
    """Find the nth occurrence of a weekday (0=Mon) in a given month."""
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
    """Generate set of date ordinals for US institutional closures.
    Matches the holidays detected empirically in Layer 1."""
    holidays = set()
    for yr in range(year_start, year_end + 1):
        dates = [
            date(yr, 1, 1),                          # New Year's Day
            _nth_weekday(yr, 5, -1, 0),               # Memorial Day
            date(yr, 7, 4),                            # Independence Day
            _nth_weekday(yr, 9, 1, 0),                 # Labor Day
            _nth_weekday(yr, 11, 4, 3),                # Thanksgiving
            _nth_weekday(yr, 11, 4, 3) + timedelta(1), # Day after Thanksgiving
            date(yr, 12, 25),                          # Christmas Day
        ]
        for d in dates:
            holidays.add(d.toordinal())
    return holidays


# ── Build patient date arrays ──────────────────────────────────────────────

def build_patient_date_arrays(data_dir, scenario='combined_order'):
    """
    Load extracts, build per-patient arrays of distinct date ordinals.
    Returns dict: { PATID: np.array([ordinal1, ordinal2, ...]) }
    """
    print(f"Loading data for scenario: {scenario}")

    all_dates = {}

    enc_file = os.path.join(data_dir, 'extract_outpatient_encounters.csv')
    if os.path.exists(enc_file):
        print(f"  Loading encounters...", end=' ', flush=True)
        df = pd.read_csv(enc_file, usecols=['PATID', 'event_date'],
                         parse_dates=['event_date'])
        n = 0
        for patid, d in zip(df['PATID'].values, df['event_date'].values):
            if pd.isna(d):
                continue
            dt = pd.Timestamp(d).to_pydatetime().date()
            if patid not in all_dates:
                all_dates[patid] = set()
            all_dates[patid].add(dt.toordinal())
            n += 1
        print(f"{n:,} dates")
        del df

    lab_file = os.path.join(data_dir, 'extract_outpatient_labs.csv')
    if os.path.exists(lab_file):
        date_col = 'order_date' if 'order' in scenario else 'result_date'
        print(f"  Loading labs ({date_col})...", end=' ', flush=True)
        df = pd.read_csv(lab_file, usecols=['PATID', date_col],
                         parse_dates=[date_col])
        n = 0
        for patid, d in zip(df['PATID'].values, df[date_col].values):
            if pd.isna(d):
                continue
            dt = pd.Timestamp(d).to_pydatetime().date()
            if patid not in all_dates:
                all_dates[patid] = set()
            all_dates[patid].add(dt.toordinal())
            n += 1
        print(f"{n:,} dates")
        del df

    px_file = os.path.join(data_dir, 'extract_outpatient_procedures.csv')
    if os.path.exists(px_file):
        print(f"  Loading procedures...", end=' ', flush=True)
        df = pd.read_csv(px_file, usecols=['PATID', 'event_date'],
                         parse_dates=['event_date'])
        n = 0
        for patid, d in zip(df['PATID'].values, df['event_date'].values):
            if pd.isna(d):
                continue
            dt = pd.Timestamp(d).to_pydatetime().date()
            if patid not in all_dates:
                all_dates[patid] = set()
            all_dates[patid].add(dt.toordinal())
            n += 1
        print(f"{n:,} dates")
        del df

    # Convert to sorted arrays, keep patients with >= 2 dates
    patient_arrays = {}
    for patid, ordinals in all_dates.items():
        if len(ordinals) >= 2:
            patient_arrays[patid] = np.array(sorted(ordinals), dtype=np.int32)

    print(f"  Patients with >=2 distinct dates: {len(patient_arrays):,}")
    return patient_arrays


# ── Core attack — per patient ──────────────────────────────────────────────
#
#  Global lookup arrays set once per worker via initializer.

_holiday_set = None
_holiday_arr = None
_p_weekday = None
_p_holiday = None

def _init_worker(holiday_ordinals_list, p_weekday, p_holiday):
    """Called once per worker process to set up shared read-only data."""
    global _holiday_set, _holiday_arr, _p_weekday, _p_holiday
    _holiday_set = set(holiday_ordinals_list)
    _holiday_arr = np.array(holiday_ordinals_list, dtype=np.int32)
    _p_weekday = p_weekday
    _p_holiday = p_holiday


def attack_patient(args):
    """
    Run the full attack for one patient.

    args: (patid, date_ordinals, shift_range)

    Uses global _holiday_set, _holiday_arr, _p_weekday, _p_holiday
    set by the pool initializer.

    Returns dict with vulnerability metrics.
    """
    patid, date_ordinals, shift_range = args

    n_dates = len(date_ordinals)
    span_days = int(date_ordinals[-1] - date_ordinals[0])

    # ── 1. Apply random backward shift ────────────────────────────────
    rng = np.random.default_rng(hash(patid) & 0xFFFFFFFF)
    s_true = int(rng.integers(1, shift_range + 1))
    shifted = date_ordinals - s_true

    # ── 2. Candidate shifts: m in [1 .. shift_range] ─────────────────
    candidates = np.arange(1, shift_range + 1, dtype=np.int32)

    # Reconstruct candidate originals: shape (shift_range, n_dates)
    candidate_originals = shifted[None, :] + candidates[:, None]

    # Day-of-week: (ordinal - 1) % 7 -> 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    dow = (candidate_originals - 1) % 7
    is_weekend = dow >= 5                     # bool (shift_range, n_dates)
    n_weekend_per_candidate = is_weekend.sum(axis=1)

    # ── 3. Hard elimination: weekday ──────────────────────────────────
    survives_dow = n_weekend_per_candidate == 0
    n_survivors_dow = int(survives_dow.sum())

    # ── 4. Hard elimination: holidays (among DOW survivors) ───────────
    survives_all = survives_dow.copy()
    if len(_holiday_arr) > 0:
        for idx in np.where(survives_dow)[0]:
            if np.any(np.isin(candidate_originals[idx], _holiday_arr)):
                survives_all[idx] = False
    n_survivors_hol = int(survives_all.sum())

    # ── 5. Soft probabilistic model ───────────────────────────────────
    #
    #  log-likelihood for candidate m:
    #    ll(m) = sum_i  log P(date_i | m)
    #
    #  P(date_i | m):
    #    - holiday:  p_holiday           (overrides weekday check)
    #    - weekday:  p_weekday
    #    - weekend:  1 - p_weekday

    log_pwd  = np.float64(np.log(_p_weekday))
    log_pwe  = np.float64(np.log(1.0 - _p_weekday))
    log_phol = np.float64(np.log(_p_holiday))

    # Base: weekday/weekend
    ll = np.where(is_weekend, log_pwe, log_pwd).sum(axis=1)

    # Override for holidays
    if len(_holiday_arr) > 0:
        is_holiday = np.isin(candidate_originals, _holiday_arr)
        current_contrib = np.where(is_weekend, log_pwe, log_pwd)
        adjustment = (log_phol - current_contrib) * is_holiday
        ll += adjustment.sum(axis=1)

    # Numerically stable softmax -> posterior
    ll_max = ll.max()
    posterior_unnorm = np.exp(ll - ll_max)
    Z = posterior_unnorm.sum()
    posterior = posterior_unnorm / Z

    # ── 6. Metrics ────────────────────────────────────────────────────
    max_entropy = np.log2(shift_range)

    # Hard model
    entropy_hard = np.log2(n_survivors_hol) if n_survivors_hol > 0 else 0.0
    p_true_hard = 1.0 / n_survivors_hol if n_survivors_hol > 0 else 0.0

    # Soft model
    mask_pos = posterior > 0
    entropy_soft = float(-np.sum(posterior[mask_pos] * np.log2(posterior[mask_pos])))
    p_true_soft = float(posterior[s_true - 1])
    p_max_soft = float(posterior.max())
    argmax_shift = int(np.argmax(posterior)) + 1
    max_is_true = (argmax_shift == s_true)

    info_gain_hard = max_entropy - entropy_hard
    info_gain_soft = max_entropy - entropy_soft

    # Recovery
    true_idx = s_true - 1
    true_survived = bool(survives_all[true_idx])
    shift_recovered = (n_survivors_hol == 1) and true_survived
    shift_in_top3 = (n_survivors_hol <= 3) and true_survived

    effective_candidates = 2.0 ** entropy_soft

    # Mod-7 residue classes
    if n_survivors_dow > 0:
        n_residue_classes = len(np.unique(candidates[survives_dow] % 7))
    else:
        n_residue_classes = 0

    return {
        'PATID': patid,
        'shift_range': shift_range,
        's_true': s_true,
        'n_dates': n_dates,
        'span_days': span_days,
        # Hard model
        'n_survivors_dow': n_survivors_dow,
        'n_survivors_holiday': n_survivors_hol,
        'entropy_hard': round(entropy_hard, 4),
        'p_true_hard': round(p_true_hard, 6),
        'info_gain_hard': round(info_gain_hard, 4),
        'shift_recovered': shift_recovered,
        'shift_in_top3': shift_in_top3,
        'n_residue_classes': n_residue_classes,
        'true_survived': true_survived,
        # Soft model
        'entropy_soft': round(entropy_soft, 4),
        'p_true_soft': round(p_true_soft, 6),
        'p_max_soft': round(p_max_soft, 6),
        'max_is_true': max_is_true,
        'info_gain_soft': round(info_gain_soft, 4),
        'effective_candidates': round(effective_candidates, 2),
        # Baseline
        'entropy_before': round(max_entropy, 4),
    }


# ── Summary statistics ────────────────────────────────────────────────────

def summarize_results(df, shift_range):
    """Print and return summary statistics for a given shift range."""
    n = len(df)
    max_ent = np.log2(shift_range)

    print(f"\n{'='*78}")
    print(f"SHIFT RANGE: {shift_range} days   |   "
          f"Patients: {n:,}   |   Max entropy: {max_ent:.2f} bits")
    print(f"{'='*78}")

    # Sanity
    n_bad = n - df['true_survived'].sum()
    if n_bad > 0:
        print(f"  *** WARNING: true shift eliminated for {n_bad:,} patients ***")

    # ── Hard model overview ───────────────────────────────────────────
    print(f"\n  HARD MODEL (binary elimination)")
    print(f"  {'After DOW elimination:':<40} "
          f"median survivors = {df['n_survivors_dow'].median():.0f}, "
          f"mean = {df['n_survivors_dow'].mean():.1f}")
    print(f"  {'After DOW + holiday elimination:':<40} "
          f"median survivors = {df['n_survivors_holiday'].median():.0f}, "
          f"mean = {df['n_survivors_holiday'].mean():.1f}")
    print(f"  {'Shift uniquely recovered:':<40} "
          f"{df['shift_recovered'].sum():>10,}  "
          f"({100*df['shift_recovered'].mean():.2f}%)")
    print(f"  {'Shift in top 3:':<40} "
          f"{df['shift_in_top3'].sum():>10,}  "
          f"({100*df['shift_in_top3'].mean():.2f}%)")
    print(f"  {'DOW uniquely determined (1 residue):':<40} "
          f"{(df['n_residue_classes']==1).sum():>10,}  "
          f"({100*(df['n_residue_classes']==1).mean():.2f}%)")

    # ── Soft model overview ───────────────────────────────────────────
    print(f"\n  SOFT MODEL (Bayesian posterior)")
    print(f"  {'Mean entropy (bits):':<40} {df['entropy_soft'].mean():.2f}")
    print(f"  {'Mean info gain (bits):':<40} {df['info_gain_soft'].mean():.2f}")
    print(f"  {'Mean P(true shift):':<40} {df['p_true_soft'].mean():.4f}")
    print(f"  {'Mean P(max shift):':<40} {df['p_max_soft'].mean():.4f}")
    print(f"  {'Argmax = true shift:':<40} "
          f"{df['max_is_true'].sum():>10,}  "
          f"({100*df['max_is_true'].mean():.2f}%)")
    print(f"  {'Mean effective candidates:':<40} "
          f"{df['effective_candidates'].mean():.1f}")

    # ── Stratify by date count ────────────────────────────────────────
    bins = [2, 4, 6, 10, 17, 20, 30, 60, 100, 100000]
    labels = ['2-3', '4-5', '6-9', '10-16', '17-19',
              '20-29', '30-59', '60-99', '100+']
    df_c = df.copy()
    df_c['date_bin'] = pd.cut(df_c['n_dates'], bins=bins, labels=labels, right=False)

    hdr = (f"  {'Dates':<8} {'N':>9} {'Med surv':>9} {'Ent hard':>9} "
           f"{'Ent soft':>9} {'P(true)':>8} {'P(max)':>8} "
           f"{'Recov%':>7} {'Max=T%':>7}")
    print(f"\n  BY DATE COUNT")
    print(hdr)
    print(f"  {'-'*84}")

    summary_rows = []
    for label in labels:
        g = df_c[df_c['date_bin'] == label]
        if len(g) == 0:
            continue
        row = {
            'shift_range': shift_range,
            'stratum': 'dates',
            'bin': label,
            'n_patients': len(g),
            'median_survivors': g['n_survivors_holiday'].median(),
            'mean_entropy_hard': round(g['entropy_hard'].mean(), 3),
            'mean_entropy_soft': round(g['entropy_soft'].mean(), 3),
            'mean_p_true_soft': round(g['p_true_soft'].mean(), 5),
            'mean_p_max_soft': round(g['p_max_soft'].mean(), 5),
            'pct_recovered': round(100 * g['shift_recovered'].mean(), 2),
            'pct_max_is_true': round(100 * g['max_is_true'].mean(), 2),
        }
        summary_rows.append(row)
        print(f"  {label:<8} {row['n_patients']:>9,} {row['median_survivors']:>9.0f} "
              f"{row['mean_entropy_hard']:>9.2f} {row['mean_entropy_soft']:>9.2f} "
              f"{row['mean_p_true_soft']:>8.4f} {row['mean_p_max_soft']:>8.4f} "
              f"{row['pct_recovered']:>6.1f}% {row['pct_max_is_true']:>6.1f}%")

    # ── Stratify by record span ───────────────────────────────────────
    span_bins = [0, 180, 365, 730, 1095, 100000]
    span_labels = ['<6mo', '6-12mo', '1-2yr', '2-3yr', '3yr+']
    df_c['span_bin'] = pd.cut(df_c['span_days'], bins=span_bins,
                               labels=span_labels, right=False)

    print(f"\n  BY RECORD SPAN")
    print(hdr.replace('Dates', 'Span '))
    print(f"  {'-'*84}")

    for label in span_labels:
        g = df_c[df_c['span_bin'] == label]
        if len(g) == 0:
            continue
        row = {
            'shift_range': shift_range,
            'stratum': 'span',
            'bin': label,
            'n_patients': len(g),
            'median_survivors': g['n_survivors_holiday'].median(),
            'mean_entropy_hard': round(g['entropy_hard'].mean(), 3),
            'mean_entropy_soft': round(g['entropy_soft'].mean(), 3),
            'mean_p_true_soft': round(g['p_true_soft'].mean(), 5),
            'mean_p_max_soft': round(g['p_max_soft'].mean(), 5),
            'pct_recovered': round(100 * g['shift_recovered'].mean(), 2),
            'pct_max_is_true': round(100 * g['max_is_true'].mean(), 2),
        }
        summary_rows.append(row)
        print(f"  {label:<8} {row['n_patients']:>9,} {row['median_survivors']:>9.0f} "
              f"{row['mean_entropy_hard']:>9.2f} {row['mean_entropy_soft']:>9.2f} "
              f"{row['mean_p_true_soft']:>8.4f} {row['mean_p_max_soft']:>8.4f} "
              f"{row['pct_recovered']:>6.1f}% {row['pct_max_is_true']:>6.1f}%")

    # ── Vulnerability tier summary ────────────────────────────────────
    print(f"\n  VULNERABILITY TIERS (soft model, P(max) thresholds)")
    tiers = [
        (0.50, 1.01, 'CRITICAL: P(max) >= 0.50  — likely uniquely identified'),
        (0.25, 0.50, 'HIGH:     P(max) 0.25-0.50 — strong partial recovery'),
        (0.10, 0.25, 'MODERATE: P(max) 0.10-0.25 — meaningful info leak'),
        (0.00, 0.10, 'LOW:      P(max) < 0.10    — limited vulnerability'),
    ]
    for lo, hi, desc in tiers:
        mask = (df['p_max_soft'] >= lo) & (df['p_max_soft'] < hi)
        cnt = mask.sum()
        print(f"    {desc}: {cnt:>10,} ({100*cnt/n:.1f}%)")

    return summary_rows


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Layer 4: Simulated date-shift attack')
    parser.add_argument('data_dir', nargs='?',
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Directory with extract_*.csv files')
    parser.add_argument('--shift', nargs='+', type=int,
                        default=[14, 30, 90, 365, 1826],
                        help='Shift ranges to test (days)')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()),
                        help='Parallel workers')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample N patients (0=all)')
    parser.add_argument('--scenario', default='combined_order',
                        choices=['combined_order', 'combined_result'],
                        help='Lab date column')
    parser.add_argument('--p-weekday', type=float, default=0.98,
                        help='Soft model: P(event on weekday) for outpatient')
    parser.add_argument('--p-holiday', type=float, default=0.001,
                        help='Soft model: P(event on institutional holiday)')
    args = parser.parse_args()

    output_dir = args.data_dir

    print("=" * 60)
    print("LAYER 4: SIMULATED DATE-SHIFT ATTACK")
    print("=" * 60)
    print(f"  Soft model: p_weekday={args.p_weekday}, p_holiday={args.p_holiday}")

    # Load data
    patient_arrays = build_patient_date_arrays(args.data_dir, args.scenario)

    if args.sample > 0 and args.sample < len(patient_arrays):
        rng = np.random.default_rng(42)
        keys = rng.choice(list(patient_arrays.keys()), args.sample, replace=False)
        patient_arrays = {k: patient_arrays[k] for k in keys}
        print(f"  Sampled {args.sample:,} patients")

    # Holiday calendar
    all_ord = np.concatenate(list(patient_arrays.values()))
    min_year = date.fromordinal(int(all_ord.min())).year - 6
    max_year = date.fromordinal(int(all_ord.max())).year + 1
    holiday_ordinals = generate_holiday_ordinals(min_year, max_year)
    holiday_list = sorted(holiday_ordinals)
    print(f"  Holiday calendar: {min_year}-{max_year} ({len(holiday_list)} dates)")

    patid_list = list(patient_arrays.keys())
    all_summaries = []

    for shift_range in args.shift:
        print(f"\n{'>'*60}")
        print(f"  Shift range: {shift_range} days | "
              f"{len(patid_list):,} patients | {args.workers} workers")
        print(f"{'>'*60}")

        work_items = [
            (pid, patient_arrays[pid], shift_range) for pid in patid_list
        ]

        t0 = time.time()

        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(holiday_list, args.p_weekday, args.p_holiday),
        ) as pool:
            results = pool.map(attack_patient, work_items, chunksize=500)

        elapsed = time.time() - t0
        rate = len(results) / elapsed
        print(f"  Completed in {elapsed:.1f}s ({rate:,.0f} patients/sec)")

        df_res = pd.DataFrame(results)

        outfile = os.path.join(output_dir, f'layer4_attack_{shift_range}d.csv')
        df_res.to_csv(outfile, index=False)
        print(f"  Saved: {outfile}")

        summary_rows = summarize_results(df_res, shift_range)
        all_summaries.extend(summary_rows)

        del df_res, results

    if all_summaries:
        df_sum = pd.DataFrame(all_summaries)
        sf = os.path.join(output_dir, 'layer4_attack_summary.csv')
        df_sum.to_csv(sf, index=False)
        print(f"\nSaved summary: {sf}")

    print("\n=== Done. ===")


if __name__ == '__main__':
    main()
