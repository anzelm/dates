#!/usr/bin/env python3
"""
make_synthetic_cohort.py — synthetic outpatient date cohort for end-to-end demos
================================================================================

Generates a fully synthetic set of extract_*.csv files in exactly the format
consumed by layer1_patient_vulnerability.py, layer4_attack.py,
layer4_selective_attack.py and layer5_attack.py, so that any reader can run a
complete demonstration without access to an institutional CDW.

No real patient data, no derivative of real patient data, and no record-level
information of any kind is used or reproduced.  The generator is driven by a
small *profile* of aggregate parameters (a date-count survival table, per-stratum
record-span quantiles, day-of-week weights and holiday suppression factors).  The
default profile is transcribed from the aggregate statistics published in the
accompanying manuscript.  A site can regenerate the profile from its own data
with --calibrate (see below); the resulting JSON still contains only aggregate
numbers.

What the synthetic cohort reproduces
------------------------------------
  * the distribution of distinct event dates per patient (heavy-tailed; anchored
    on the published survival values at k = 2, 4, 10, 17, 60, 100)
  * the joint distribution of record span with date count
  * weekday concentration of outpatient events (default 98% weekday, with the
    observed Mon-Fri profile, not a flat weekday distribution)
  * suppression of events on the seven institutional closure days, at the
    *empirically observed* residual rates (~2-17% of a normal day), which is a
    weaker signal than the p_holiday = 0.001 the attack model assumes
  * the split of a patient's dates across encounters / labs / procedures, and
    the order-date vs result-date turnaround that makes RESULT_DATE less
    weekday-skewed than LAB_ORDER_DATE

What it deliberately does NOT reproduce
---------------------------------------
Clinical content.  Diagnoses, values, demographics and code assignments are
placeholders: only the temporal structure is modelled, because only the
temporal structure drives the unshifting risk.

Dates written to the CSVs are TRUE (unshifted) dates, exactly as an
institutional extract would be.  The shift is applied by layer4_attack.py.

Usage
-----
  # default demo cohort (10,000 patients, 5-year window)
  python make_synthetic_cohort.py --out-dir synthetic_demo

  # larger cohort, fixed seed, alternative weekend rate (cf. Fig. 3 sweep)
  python make_synthetic_cohort.py --out-dir sim_10pct -n 50000 \
         --weekend-rate 0.10 --seed 7

  # derive an institution-matched profile from real per-patient counts
  # (aggregate output only — no record-level data leaves the site)
  python make_synthetic_cohort.py --calibrate layer1_patient_date_counts.csv \
         --calibrate-scenario combined_outpatient_order \
         --write-profile site_profile.json --no-generate

  # then generate from it
  python make_synthetic_cohort.py --profile site_profile.json -n 20000 \
         --out-dir synthetic_site

Outputs (in --out-dir)
----------------------
  extract_outpatient_encounters.csv   PATID, event_date, ENC_TYPE, RAW_ENC_TYPE
  extract_outpatient_labs.csv         PATID, order_date, result_date, lab_name, LAB_LOINC
  extract_outpatient_procedures.csv   PATID, event_date, PX, PX_TYPE
  extract_demographics.csv            PATID, BIRTH_DATE, SEX
  synthetic_cohort_profile.json       the profile actually used (provenance)

Author: A. Kudlicki et al.  Released under the repository license.
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default profile: aggregate statistics only, transcribed from the manuscript
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROFILE = {
    "profile_name": "published_combined_outpatient",
    "source": (
        "Aggregate statistics published in the accompanying manuscript "
        "(combined outpatient scenario, labs = LAB_ORDER_DATE, "
        "N = 1,617,988 patients with >=2 distinct dates)."
    ),

    # P(K >= k) among patients with K >= 2 distinct dates.
    # k = 2, 4, 10, 17, 60 and 100 are the published values (Table 2, combined
    # outpatient scenario); beyond k = 100 the tail is extrapolated by the same
    # piecewise power law, with no published anchor.
    "date_count_ccdf": [
        [2,    1.000],
        [4,    0.717],
        [10,   0.410],
        [17,   0.269],
        [60,   0.055],
        [100,  0.014],
        [200,  0.0035],
        [400,  0.0009],
        [800,  0.0002],
        [1300, 0.0],
    ],

    # Record span (days between first and last event) by date-count stratum,
    # given as [p10, p25, p50, p75, p90].  These are illustrative values
    # consistent with the published median spans; use --calibrate to replace
    # them with site-specific aggregates.
    "span_quantiles": {
        "2-3":    [1,   14,  120,  500,  1100],
        "4-5":    [7,   60,  380,  900,  1450],
        "6-9":    [21,  130, 600,  1150, 1600],
        "10-16":  [60,  280, 830,  1350, 1700],
        "17-19":  [110, 420, 1000, 1470, 1740],
        "20-29":  [140, 480, 1080, 1520, 1760],
        "30-59":  [210, 600, 1220, 1620, 1780],
        "60-99":  [330, 780, 1400, 1700, 1800],
        "100+":   [450, 950, 1550, 1760, 1815],
    },

    # Relative event volume by day of week (observed / mean daily count),
    # from the ambulatory-visit row of the day-of-week heatmap.
    "dow_ratios": {"Mon": 1.38, "Tue": 1.48, "Wed": 1.42,
                   "Thu": 1.37, "Fri": 1.21, "Sat": 0.10, "Sun": 0.04},

    # Residual event volume on institutional closure days, as a fraction of an
    # average day.  Blended across encounter, laboratory and procedure subtypes,
    # whose observed holiday ratios span <0.01 (office visits) to 0.77 (refills);
    # a patient's distinct dates are a union over those subtypes, so the
    # effective per-date rate is several times the ambulatory-visit value.
    # Note this is far larger than the p_holiday = 0.001 the attack model
    # assumes: the synthetic cohort does not hand the attack a cleaner holiday
    # signal than was actually observed.
    "holiday_suppression": {
        "new_years_day": 0.09, "memorial_day": 0.12, "independence_day": 0.09,
        "labor_day": 0.12, "thanksgiving": 0.09, "day_after_thanksgiving": 0.50,
        "christmas": 0.06,
    },

    # Weekend use is not uniform across patients: a minority of patients account
    # for most weekend outpatient activity.  frac_high patients are given
    # high_multiple times the cohort weekend rate; the rest are rescaled so the
    # cohort-level weekend rate is preserved.
    "weekend_mixture": {"frac_high": 0.10, "high_multiple": 8.0},

    # Fraction of the cohort appearing in each source table, and the per-date
    # probability that a given event date is represented in that table.
    "source_membership": {"encounters": 0.975, "labs": 0.129, "procedures": 0.720},
    "source_date_share": {"encounters": 0.70, "labs": 0.35, "procedures": 0.50},

    # Lab turnaround: P(result_date - order_date = d days), d = 0..7, plus a
    # send-out tail.  Reproduces the lower weekday concentration of RESULT_DATE.
    "lab_turnaround": {"probs": [0.62, 0.17, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01],
                       "sendout_frac": 0.03, "sendout_min": 8, "sendout_max": 21},

    "window_start": "2021-02-01",
    "window_end": "2026-02-01",
}

DATE_BINS = [2, 4, 6, 10, 17, 20, 30, 60, 100, 10 ** 9]
DATE_LABELS = ["2-3", "4-5", "6-9", "10-16", "17-19", "20-29", "30-59", "60-99", "100+"]

RAW_ENC_TYPES = [
    ("AV", "Office Visit", 0.22), ("AV", "Appointment", 0.26),
    ("AV", "Postpartum Visit", 0.01), ("TH", "Telemedicine", 0.05),
    ("OA", "Ancillary Procedure", 0.09), ("OA", "Routine Prenatal", 0.02),
    ("OA", "Image Encounter", 0.06), ("OA", "Nurse Triage", 0.04),
    ("AV", "Consult", 0.05), ("AV", "Nurse Only", 0.04),
    ("AV", "Clinical Support", 0.03), ("OA", "Transcribe Orders", 0.04),
    ("OA", "Chart Update", 0.04), ("OA", "Refill", 0.05),
]

LAB_NAMES = [
    ("Hemoglobin A1C", "4548-4", 0.10), ("Glucose", "2345-7", 0.12),
    ("Creatinine", "2160-0", 0.11), ("Lipid Panel", "57698-3", 0.08),
    ("TSH", "3016-3", 0.07), ("Hemoglobin", "718-7", 0.09),
    ("Sodium", "2951-2", 0.08), ("Potassium", "2823-3", 0.08),
    ("INR", "6301-6", 0.03), ("Vitamin D", "49054-0", 0.04),
    ("Urinalysis", "24357-6", 0.06), ("Hepatic Function Panel", "24325-3", 0.05),
    ("CBC with Differential", "57021-8", 0.06), ("Ferritin", "2276-4", 0.03),
]

PX_CODES = [
    ("99213", "CH"), ("99214", "CH"), ("80053", "CH"), ("85025", "CH"),
    ("71046", "CH"), ("93000", "CH"), ("36415", "CH"), ("90471", "CH"),
    ("76700", "CH"), ("45378", "CH"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Holiday calendar — same construction as layer4_attack.py
# ─────────────────────────────────────────────────────────────────────────────

def _nth_weekday(year, month, n, weekday):
    """nth occurrence of a weekday (0=Mon) in a month; n<0 counts from the end."""
    if n > 0:
        d = date(year, month, 1)
        first = d + timedelta(days=(weekday - d.weekday()) % 7)
        return first + timedelta(weeks=n - 1)
    d = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
    delta = (d.weekday() - weekday) % 7
    return d - timedelta(days=delta) + timedelta(weeks=n + 1 if n < -1 else 0)


def holiday_table(year_start, year_end):
    """{ordinal: holiday_key} for the seven detected institutional closures."""
    out = {}
    for yr in range(year_start, year_end + 1):
        thx = _nth_weekday(yr, 11, 4, 3)
        for key, d in [
            ("new_years_day", date(yr, 1, 1)),
            ("memorial_day", _nth_weekday(yr, 5, -1, 0)),
            ("independence_day", date(yr, 7, 4)),
            ("labor_day", _nth_weekday(yr, 9, 1, 0)),
            ("thanksgiving", thx),
            ("day_after_thanksgiving", thx + timedelta(days=1)),
            ("christmas", date(yr, 12, 25)),
        ]:
            out[d.toordinal()] = key
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Profile handling
# ─────────────────────────────────────────────────────────────────────────────

def build_count_pmf(ccdf_anchors, k_max=None):
    """
    Turn a sparse survival table [[k, P(K>=k)], ...] into a pmf over k.
    Between anchors the survival function is interpolated log-linearly in
    log(k) (i.e. a piecewise power law), which reproduces the anchor values
    exactly and keeps the tail monotone.
    """
    anchors = sorted((int(k), float(s)) for k, s in ccdf_anchors)
    k_top = anchors[-1][0] if k_max is None else int(k_max)
    ks = np.arange(2, k_top + 1)

    xs = np.log([a[0] for a in anchors])
    ys = np.log([max(a[1], 1e-12) for a in anchors])
    surv = np.exp(np.interp(np.log(ks), xs, ys))
    surv[0] = 1.0
    surv = np.minimum.accumulate(surv)
    if anchors[-1][1] <= 0:
        surv[-1] = 0.0

    pmf = np.diff(np.append(surv, 0.0)) * -1.0
    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    return ks, pmf


def day_weights(profile, weekend_rate=None, holiday_scale=1.0):
    """
    Per-day sampling weights: index 0..6 = Mon..Sun, plus holiday multipliers.
    If weekend_rate is given, the Sat/Sun mass is rescaled to that fraction
    while preserving the relative Sat:Sun and Mon..Fri shapes.
    """
    r = profile["dow_ratios"]
    wd = np.array([r["Mon"], r["Tue"], r["Wed"], r["Thu"], r["Fri"]], dtype=float)
    we = np.array([r["Sat"], r["Sun"]], dtype=float)
    wd /= wd.sum()
    we = we / we.sum() if we.sum() > 0 else np.array([0.5, 0.5])

    if weekend_rate is None:
        total_we = (r["Sat"] + r["Sun"]) / sum(r.values())
    else:
        total_we = float(weekend_rate)

    w = np.concatenate([wd * (1.0 - total_we), we * total_we])
    hol = {k: min(1.0, v * holiday_scale)
           for k, v in profile["holiday_suppression"].items()}
    return w, hol, total_we


# ─────────────────────────────────────────────────────────────────────────────
# Calibration from an existing per-patient count file (aggregates only)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(path, scenario, base_profile):
    """
    Fit a profile from layer1_patient_date_counts.csv (columns: PATID,
    n_distinct_dates, span_days, scenario).  Only aggregate quantities —
    a survival table and per-stratum span quantiles — are retained.
    """
    df = pd.read_csv(path, usecols=["n_distinct_dates", "span_days", "scenario"])
    if scenario:
        df = df[df["scenario"] == scenario]
    if df.empty:
        sys.exit(f"ERROR: no rows for scenario '{scenario}' in {path}")

    df = df[df["n_distinct_dates"] >= 2]
    n = len(df)
    k = df["n_distinct_dates"].to_numpy()

    grid = [2, 3, 4, 5, 6, 8, 10, 13, 17, 20, 25, 30, 40, 60, 80, 100,
            150, 200, 300, 400, 600, 800, 1000]
    grid = [g for g in grid if g <= k.max()]
    ccdf = [[int(g), round(float((k >= g).mean()), 6)] for g in grid]
    if ccdf[-1][1] > 0:
        ccdf.append([int(k.max()) + 1, 0.0])

    df = df.copy()
    df["bin"] = pd.cut(df["n_distinct_dates"], bins=DATE_BINS,
                       labels=DATE_LABELS, right=False)
    spans = {}
    for lab in DATE_LABELS:
        g = df.loc[df["bin"] == lab, "span_days"].dropna()
        if len(g) >= 20:
            spans[lab] = [int(round(v)) for v in
                          np.percentile(g, [10, 25, 50, 75, 90])]
        else:
            spans[lab] = base_profile["span_quantiles"][lab]

    prof = dict(base_profile)
    prof["profile_name"] = f"calibrated:{os.path.basename(path)}:{scenario or 'all'}"
    prof["source"] = (f"Aggregate survival table and span quantiles fitted from "
                      f"{n:,} patients in {os.path.basename(path)} "
                      f"(scenario={scenario or 'all'}). No record-level data retained.")
    prof["date_count_ccdf"] = ccdf
    prof["span_quantiles"] = spans
    prof["calibration_n_patients"] = int(n)
    print(f"  Calibrated on {n:,} patients; median dates = {np.median(k):.0f}, "
          f"median span = {df['span_days'].median():.0f} d")
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def sample_span(rng, k_arr, span_q):
    """Draw a record span per patient by interpolating its stratum's quantiles."""
    labels = pd.cut(k_arr, bins=DATE_BINS, labels=DATE_LABELS, right=False).astype(str)
    q_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    out = np.empty(len(k_arr), dtype=np.float64)
    u = rng.random(len(k_arr))
    for lab in DATE_LABELS:
        m = labels == lab
        if not m.any():
            continue
        qs = np.asarray(span_q[lab], dtype=float)
        out[m] = np.interp(u[m], q_levels, qs)
        # extrapolate the tails so the full range of spans is represented
        lo, hi = u[m] < 0.10, u[m] > 0.90
        idx = np.where(m)[0]
        out[idx[lo]] = qs[0] * (u[m][lo] / 0.10) ** 1.5
        out[idx[hi]] = qs[-1] + (qs[-1] - qs[-2]) * ((u[m][hi] - 0.90) / 0.10) ** 1.5
    return np.maximum(out, 1.0)


def gumbel_top_k(rng, log_w, k):
    """Weighted sampling of k items without replacement (Gumbel top-k trick)."""
    g = rng.gumbel(size=log_w.shape[0])
    return np.argpartition(-(log_w + g), k - 1)[:k]


def generate(profile, n_patients, seed, weekend_rate, holiday_scale, patid_prefix):
    rng = np.random.default_rng(seed)

    start = date.fromisoformat(profile["window_start"])
    end = date.fromisoformat(profile["window_end"])
    o0, o1 = start.toordinal(), end.toordinal()
    n_days = o1 - o0 + 1
    ordinals = np.arange(o0, o1 + 1)

    w_dow, hol_mult, realized_we = day_weights(profile, weekend_rate, holiday_scale)
    hol = holiday_table(start.year - 1, end.year + 1)

    # per-day weight vector across the whole window
    dow = (ordinals - 1) % 7                     # 0 = Monday
    weights = w_dow[dow].astype(np.float64)
    mean_day = w_dow.mean()                      # the "1.0 = average day" reference
    for i, o in enumerate(ordinals):
        key = hol.get(int(o))
        if key is not None:
            # holiday suppression is expressed as observed / mean daily count,
            # so set the absolute weight rather than scaling the weekday weight
            weights[i] = min(weights[i], hol_mult.get(key, 0.05) * mean_day)
    weights_base = weights
    log_w_base = np.log(np.maximum(weights_base, 1e-12))

    # date counts and spans
    ks_grid, pmf = build_count_pmf(profile["date_count_ccdf"])
    k_arr = rng.choice(ks_grid, size=n_patients, p=pmf)
    k_arr = np.minimum(k_arr, int(n_days * 0.72))          # cannot exceed weekdays available
    span_arr = sample_span(rng, k_arr, profile["span_quantiles"])
    # a span must be long enough to hold k weekday events
    span_arr = np.clip(np.maximum(span_arr, np.ceil(k_arr * 1.8)), 1, n_days - 1).astype(int)

    memb = profile["source_membership"]
    share = profile["source_date_share"]
    turn = profile["lab_turnaround"]
    tp = np.asarray(turn["probs"], dtype=float)
    tp = tp / tp.sum()

    enc_rows, lab_rows, px_rows, demo_rows = [], [], [], []

    enc_types = np.array([t[0] for t in RAW_ENC_TYPES])
    enc_raw = np.array([t[1] for t in RAW_ENC_TYPES])
    enc_p = np.array([t[2] for t in RAW_ENC_TYPES], dtype=float)
    enc_p /= enc_p.sum()
    lab_nm = np.array([l[0] for l in LAB_NAMES])
    lab_lo = np.array([l[1] for l in LAB_NAMES])
    lab_p = np.array([l[2] for l in LAB_NAMES], dtype=float)
    lab_p /= lab_p.sum()
    px_cd = np.array([p[0] for p in PX_CODES])
    px_ty = np.array([p[1] for p in PX_CODES])

    # per-patient weekend propensity (two-component mixture, cohort rate preserved)
    mix = profile.get("weekend_mixture", {"frac_high": 0.0, "high_multiple": 1.0})
    f_hi, mult = float(mix["frac_high"]), float(mix["high_multiple"])
    r_hi = min(0.9, realized_we * mult)
    r_lo = max(0.0, (realized_we - f_hi * r_hi) / (1.0 - f_hi)) if f_hi < 1 else realized_we
    is_hi = rng.random(n_patients) < f_hi
    is_wknd = dow >= 5
    base_we = max(realized_we, 1e-9)

    for i in range(n_patients):
        patid = f"{patid_prefix}{i + 1:08d}"
        k = int(k_arr[i])
        span = int(span_arr[i])

        r_i = r_hi if is_hi[i] else r_lo
        if abs(r_i - realized_we) < 1e-12:
            weights, log_w = weights_base, log_w_base
        else:
            adj = np.where(is_wknd, r_i / base_we,
                           (1.0 - r_i) / max(1.0 - base_we, 1e-9))
            weights = weights_base * adj
            log_w = np.log(np.maximum(weights, 1e-12))

        # each patient has a dominant encounter subtype, as in real scheduling
        j_primary = rng.choice(len(enc_p), p=enc_p)

        t0 = o0 + int(rng.integers(0, n_days - span))
        a, b = t0 - o0, t0 - o0 + span               # indices into the window
        avail = b - a + 1

        if avail <= k or span <= 10:
            # short or saturated interval: weighted draw over the whole interval
            sel = gumbel_top_k(rng, log_w[a:b + 1], min(k, avail))
            picks = (sel + a).tolist()
        else:
            # anchor the two endpoints near the ends of the interval so the
            # realized span matches the target instead of shrinking inward;
            # both anchors are drawn with the same day-type weights
            e = max(3, min(7, span // 4))   # >=4-day anchor window, so a
            #                                  zero-weekend profile stays zero
            first = a + int(rng.choice(e + 1, p=_norm(weights[a:a + e + 1])))
            lo = max(b - e, first + 1)
            last = lo + int(rng.choice(b - lo + 1, p=_norm(weights[lo:b + 1])))
            picks = [first, last]
            need = k - 2
            if need > 0:
                interior = np.arange(first + 1, last)
                if len(interior) <= need:
                    picks.extend(interior.tolist())
                else:
                    sel = gumbel_top_k(rng, log_w[first + 1:last], need)
                    picks.extend(interior[sel].tolist())

        dates = np.unique(np.asarray(picks, dtype=np.int64)) + o0

        in_enc = rng.random() < memb["encounters"]
        in_lab = rng.random() < memb["labs"]
        in_px = rng.random() < memb["procedures"]
        if not (in_enc or in_lab or in_px):
            in_enc = True

        for o in dates:
            d = date.fromordinal(int(o)).isoformat()
            placed = False
            if in_enc and rng.random() < share["encounters"]:
                n_rows = 1 + int(rng.random() < 0.20) + int(rng.random() < 0.05)
                for _ in range(n_rows):
                    j = j_primary if rng.random() < 0.80 else rng.choice(len(enc_p), p=enc_p)
                    enc_rows.append((patid, d, enc_types[j], enc_raw[j]))
                placed = True
            if in_lab and rng.random() < share["labs"]:
                j = rng.choice(len(lab_p), p=lab_p)
                if rng.random() < turn["sendout_frac"]:
                    lag = int(rng.integers(turn["sendout_min"], turn["sendout_max"] + 1))
                else:
                    lag = int(rng.choice(len(tp), p=tp))
                res = date.fromordinal(int(o) + lag).isoformat()
                lab_rows.append((patid, d, res, lab_nm[j], lab_lo[j]))
                placed = True
            if in_px and rng.random() < share["procedures"]:
                j = rng.integers(len(px_cd))
                px_rows.append((patid, d, px_cd[j], px_ty[j]))
                placed = True
            if not placed:                      # every date must appear somewhere
                if in_enc:
                    enc_rows.append((patid, d, enc_types[j_primary], enc_raw[j_primary]))
                elif in_px:
                    j = rng.integers(len(px_cd))
                    px_rows.append((patid, d, px_cd[j], px_ty[j]))
                else:
                    j = rng.choice(len(lab_p), p=lab_p)
                    lab_rows.append((patid, d, d, lab_nm[j], lab_lo[j]))

        byr = int(rng.integers(1930, 2020))
        demo_rows.append((patid, date(byr, int(rng.integers(1, 13)),
                                      int(rng.integers(1, 29))).isoformat(),
                          rng.choice(["F", "M"])))

    dfs = {
        "extract_outpatient_encounters.csv":
            pd.DataFrame(enc_rows, columns=["PATID", "event_date", "ENC_TYPE", "RAW_ENC_TYPE"]),
        "extract_outpatient_labs.csv":
            pd.DataFrame(lab_rows, columns=["PATID", "order_date", "result_date",
                                            "lab_name", "LAB_LOINC"]),
        "extract_outpatient_procedures.csv":
            pd.DataFrame(px_rows, columns=["PATID", "event_date", "PX", "PX_TYPE"]),
        "extract_demographics.csv":
            pd.DataFrame(demo_rows, columns=["PATID", "BIRTH_DATE", "SEX"]),
    }
    return dfs, realized_we


def _norm(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s > 0 else np.full(len(v), 1.0 / len(v))


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def report(dfs, profile):
    """Print the properties that matter for the attack, next to the targets."""
    frames = [dfs["extract_outpatient_encounters.csv"][["PATID", "event_date"]],
              dfs["extract_outpatient_procedures.csv"][["PATID", "event_date"]],
              dfs["extract_outpatient_labs.csv"][["PATID", "order_date"]]
              .rename(columns={"order_date": "event_date"})]
    comb = pd.concat(frames, ignore_index=True).drop_duplicates()
    comb["event_date"] = pd.to_datetime(comb["event_date"])

    g = comb.groupby("PATID")["event_date"]
    k = g.nunique()
    span = (g.max() - g.min()).dt.days
    k = k[k >= 2]
    n = len(k)

    dow = comb["event_date"].dt.dayofweek
    wknd = float((dow >= 5).mean())

    target = dict(profile.get("date_count_ccdf", []))
    print("\n  SYNTHETIC COHORT PROPERTIES")
    print(f"  {'Patients with >=2 distinct dates:':<42} {n:,}")
    print(f"  {'Median / mean distinct dates:':<42} "
          f"{k.median():.0f} / {k.mean():.1f}")
    print(f"  {'Median record span (days):':<42} {span.median():.0f}")
    print(f"  {'Weekend event rate:':<42} {100 * wknd:.2f}%")
    print(f"\n  {'Threshold':<12} {'synthetic':>12} {'profile target':>16}")
    print(f"  {'-' * 42}")
    for thr in [4, 10, 17, 60, 100]:
        got = 100.0 * (k >= thr).mean()
        tgt = target.get(thr)
        tgt_s = f"{100 * tgt:.1f}%" if tgt is not None else "—"
        print(f"  >= {thr:<9} {got:>11.1f}% {tgt_s:>16}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate a synthetic outpatient date cohort in extract_*.csv format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-n", "--n-patients", type=int, default=10000)
    p.add_argument("-o", "--out-dir", default="synthetic_cohort")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--profile", help="JSON profile to generate from")
    p.add_argument("--calibrate", help="layer1_patient_date_counts.csv to fit a profile from")
    p.add_argument("--calibrate-scenario", default="combined_outpatient_order")
    p.add_argument("--write-profile", help="write the profile actually used to this path")
    p.add_argument("--no-generate", action="store_true",
                   help="only calibrate / write the profile")
    p.add_argument("--weekend-rate", type=float, default=None,
                   help="override the fraction of events on Sat/Sun "
                        "(e.g. 0.05, 0.10, 0.20 for the weekend-rate sweep)")
    p.add_argument("--holiday-scale", type=float, default=1.0,
                   help="multiply the residual holiday rates (0 = hard closure)")
    p.add_argument("--window-start", default=None)
    p.add_argument("--window-end", default=None)
    p.add_argument("--patid-prefix", default="SYN")
    args = p.parse_args()

    print("=" * 66)
    print("SYNTHETIC OUTPATIENT COHORT GENERATOR")
    print("=" * 66)

    profile = dict(DEFAULT_PROFILE)
    if args.profile:
        with open(args.profile) as fh:
            profile = json.load(fh)
        print(f"  Profile: {args.profile} ({profile.get('profile_name', '?')})")
    elif args.calibrate:
        profile = calibrate(args.calibrate, args.calibrate_scenario, DEFAULT_PROFILE)
    else:
        print(f"  Profile: built-in ({profile['profile_name']})")

    if args.window_start:
        profile["window_start"] = args.window_start
    if args.window_end:
        profile["window_end"] = args.window_end

    if args.write_profile:
        with open(args.write_profile, "w") as fh:
            json.dump(profile, fh, indent=2)
        print(f"  Wrote profile: {args.write_profile}")

    if args.no_generate:
        return

    print(f"  Patients: {args.n_patients:,} | seed: {args.seed} | "
          f"window: {profile['window_start']} .. {profile['window_end']}")
    if args.weekend_rate is not None:
        print(f"  Weekend event rate overridden to {100 * args.weekend_rate:.1f}%")

    dfs, we = generate(profile, args.n_patients, args.seed,
                       args.weekend_rate, args.holiday_scale, args.patid_prefix)

    os.makedirs(args.out_dir, exist_ok=True)
    for fname, df in dfs.items():
        path = os.path.join(args.out_dir, fname)
        df.to_csv(path, index=False)
        print(f"  Wrote {path}  ({len(df):,} rows)")

    prof_out = os.path.join(args.out_dir, "synthetic_cohort_profile.json")
    used = dict(profile)
    used["generated_with"] = {"n_patients": args.n_patients, "seed": args.seed,
                              "weekend_rate": we, "holiday_scale": args.holiday_scale,
                              "script": os.path.basename(__file__)}
    with open(prof_out, "w") as fh:
        json.dump(used, fh, indent=2)
    print(f"  Wrote {prof_out}")

    report(dfs, profile)
    print("\n  Next:")
    print(f"    python layer1_patient_vulnerability.py {args.out_dir}")
    print(f"    python layer4_attack.py {args.out_dir} --shift 30 365 --workers 4")
    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
