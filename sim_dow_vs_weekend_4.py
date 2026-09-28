"""
Simulated Enterprise: DOW Recovery vs. Weekend Event Rate
==========================================================
Output: fig_dow_vs_weekend_rate.svg

Usage:
  python3 sim_dow_vs_weekend.py [--patients 5000] [--shift 365]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import time


def simulate_patient_dates_v2(n_dates, weekend_frac, rng):
    base_ordinal = 738000
    window = 1826
    ordinals = set()
    max_attempts = n_dates * 100
    attempts = 0
    while len(ordinals) < n_dates and attempts < max_attempts:
        attempts += 1
        want_weekend = (rng.random() < weekend_frac)
        if want_weekend:
            week_num = int(rng.integers(0, window // 7))
            base_dow = (base_ordinal - 1) % 7
            days_to_sat = (5 - base_dow) % 7
            sat = base_ordinal + days_to_sat + week_num * 7
            candidate = sat if rng.random() < 0.5 else sat + 1
        else:
            week_num = int(rng.integers(0, window // 7))
            base_dow = (base_ordinal - 1) % 7
            days_to_mon = (0 - base_dow) % 7
            mon = base_ordinal + days_to_mon + week_num * 7
            candidate = mon + int(rng.integers(0, 5))
        if base_ordinal <= candidate < base_ordinal + window:
            ordinals.add(candidate)
    return np.array(sorted(ordinals), dtype=np.int32)


def attack_dow(date_ordinals, shift_range, p_weekday, rng):
    s_true = int(rng.integers(1, shift_range + 1))
    shifted = date_ordinals - s_true
    candidates = np.arange(1, shift_range + 1, dtype=np.int32)
    candidate_originals = shifted[None, :] + candidates[:, None]
    dow = (candidate_originals - 1) % 7
    is_weekend = dow >= 5

    log_pwd = np.log(p_weekday) if p_weekday > 0 else -1e10
    log_pwe = np.log(1.0 - p_weekday) if p_weekday < 1.0 else -1e10
    ll = np.where(is_weekend, log_pwe, log_pwd).sum(axis=1)

    ll_max = ll.max()
    posterior = np.exp(ll - ll_max)
    posterior /= posterior.sum()

    true_residue = s_true % 7
    residue_classes = candidates % 7
    class_mass = np.zeros(7)
    for c in range(7):
        class_mass[c] = posterior[residue_classes == c].sum()

    best_class = int(np.argmax(class_mass))
    dow_correct = (best_class == true_residue)
    p_best_class = float(class_mass[best_class])
    return dow_correct, p_best_class


def run_simulation(weekend_fracs, date_counts, n_patients, shift_range):
    results = {}
    total_combos = len(weekend_fracs) * len(date_counts)
    combo = 0
    selective_thresholds = [0.50, 0.75, 0.90, 0.95]

    for n_dates in date_counts:
        for wf in weekend_fracs:
            combo += 1
            p_weekday = 1.0 - wf
            dow_correct_count = 0
            p_bests = []
            corrects = []

            rng = np.random.default_rng(hash((n_dates, int(wf * 10000))) & 0xFFFFFFFF)

            for i in range(n_patients):
                dates = simulate_patient_dates_v2(n_dates, wf, rng)
                if len(dates) < 2:
                    continue
                dc, pb = attack_dow(dates, shift_range, p_weekday, rng)
                dow_correct_count += dc
                p_bests.append(pb)
                corrects.append(dc)

            p_bests = np.array(p_bests)
            corrects = np.array(corrects, dtype=bool)

            selective = {}
            for tau in selective_thresholds:
                targeted = p_bests >= tau
                n_targeted = targeted.sum()
                n_correct = (targeted & corrects).sum()
                selective[tau] = {
                    'targeted_pct': 100.0 * n_targeted / n_patients,
                    'correct_pct': 100.0 * n_correct / n_patients,
                    'precision': 100.0 * n_correct / n_targeted if n_targeted > 0 else 0.0,
                }

            results[(wf, n_dates)] = {
                'dow_correct_pct': 100.0 * dow_correct_count / n_patients,
                'selective': selective,
            }

            if combo % 10 == 0 or combo == total_combos:
                s90 = selective[0.90]
                print(f"  [{combo}/{total_combos}] wkend={wf:.1%} dates={n_dates}: "
                      f"DOW correct={results[(wf, n_dates)]['dow_correct_pct']:.1f}%, "
                      f"@90%: {s90['correct_pct']:.1f}% compromised")

    return results


def compute_dow_profile(profile_rates, n_dates, n_patients):
    profiles = {}
    for wf in profile_rates:
        dow_counts = np.zeros(7, dtype=int)
        rng = np.random.default_rng(hash((n_dates, int(wf * 10000), 999)) & 0xFFFFFFFF)
        for _ in range(n_patients):
            dates = simulate_patient_dates_v2(n_dates, wf, rng)
            for d in dates:
                dow_counts[(d - 1) % 7] += 1
        profiles[wf] = dow_counts / dow_counts.sum()
    return profiles


def make_plots(results, weekend_fracs, date_counts, dow_profiles, output_dir):
    wf_pct = [w * 100 for w in weekend_fracs]
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(date_counts) - 1)) for i in range(len(date_counts))]

    chance_kw = dict(color='red', linestyle='--', alpha=0.6, linewidth=2.0)

    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.7, 1, 1, 1], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # ── Panel 0: DOW distribution ──
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x_days = np.arange(7)
    profile_colors = {0.0: '#2166ac', 0.05: '#66bd63', 0.10: '#fdae61', 0.20: '#d73027'}
    for wf, profile in sorted(dow_profiles.items()):
        ax0.plot(x_days, profile * 100, 'o-', color=profile_colors.get(wf, 'gray'),
                 label=f'{wf*100:.0f}%', markersize=5, linewidth=2)
    ax0.axhline(100/7, **chance_kw)
    ax0.set_xticks(x_days)
    ax0.set_xticklabels(day_labels, fontsize=9)
    ax0.set_ylabel('% of Events', fontsize=11)
    ax0.set_title('Simulated DOW\nDistribution', fontsize=12)
    ax0.legend(title='Weekend\nrate', fontsize=8, title_fontsize=9)
    ax0.set_ylim(0, 25)
    ax0.grid(True, alpha=0.3)

    # ── Panel 1: Argmax DOW recovery ──
    for idx, nd in enumerate(date_counts):
        y = [results[(wf, nd)]['dow_correct_pct'] for wf in weekend_fracs]
        ax1.plot(wf_pct, y, 'o-', color=colors[idx], label=f'{nd}',
                 markersize=3, linewidth=2)
    ax1.set_xlabel('Weekend Event Rate (%)', fontsize=11)
    ax1.set_ylabel('DOW Correctly Recovered (%)', fontsize=11)
    ax1.set_title('All Patients:\nArgmax DOW Recovery', fontsize=12)
    ax1.legend(title='Dates per\npatient', fontsize=8, title_fontsize=9,
               loc='upper right')
    ax1.set_xlim(-0.5, max(wf_pct) + 0.5)
    ax1.set_ylim(0, 105)
    ax1.axhline(100/7, **chance_kw)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Selective @90% ──
    for idx, nd in enumerate(date_counts):
        y = [results[(wf, nd)]['selective'][0.90]['correct_pct']
             for wf in weekend_fracs]
        ax2.plot(wf_pct, y, 'o-', color=colors[idx], label=f'{nd}',
                 markersize=3, linewidth=2)
    ax2.set_xlabel('Weekend Event Rate (%)', fontsize=11)
    ax2.set_ylabel('Patients Correctly Identified (%)', fontsize=11)
    ax2.set_title('Selective Attacker (≥90%):\nPatients Compromised', fontsize=12)
    ax2.legend(title='Dates per\npatient', fontsize=8, title_fontsize=9,
               loc='upper right')
    ax2.set_xlim(-0.5, max(wf_pct) + 0.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Selective @95% ──
    for idx, nd in enumerate(date_counts):
        y = [results[(wf, nd)]['selective'][0.95]['correct_pct']
             for wf in weekend_fracs]
        ax3.plot(wf_pct, y, 'o-', color=colors[idx], label=f'{nd}',
                 markersize=3, linewidth=2)
    ax3.set_xlabel('Weekend Event Rate (%)', fontsize=11)
    ax3.set_ylabel('Patients Correctly Identified (%)', fontsize=11)
    ax3.set_title('Selective Attacker (≥95%):\nPatients Compromised', fontsize=12)
    ax3.legend(title='Dates per\npatient', fontsize=8, title_fontsize=9,
               loc='upper right')
    ax3.set_xlim(-0.5, max(wf_pct) + 0.5)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['svg', 'pdf']:
        outfile = f'{output_dir}/fig_dow_vs_weekend_rate.{ext}'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"  Saved: {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Simulate DOW recovery vs weekend event rate')
    parser.add_argument('--patients', type=int, default=5000,
                        help='Patients per (weekend_frac, n_dates) cell')
    parser.add_argument('--shift', type=int, default=365,
                        help='Shift range in days')
    parser.add_argument('--output', default='.',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("SIMULATED ENTERPRISE: DOW RECOVERY vs WEEKEND RATE")
    print("=" * 60)
    print(f"  Patients per cell: {args.patients}")
    print(f"  Shift range: {args.shift} days")

    weekend_fracs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
                     0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                     0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.20]
    date_counts = [3, 5, 9, 15, 25, 50, 100]

    print(f"  Weekend rates: {len(weekend_fracs)} values, "
          f"0%–{max(weekend_fracs)*100:.0f}%")
    print(f"  Date counts: {date_counts}")
    print(f"  Total cells: {len(weekend_fracs) * len(date_counts)}")
    print()

    print("  Computing DOW profiles...")
    dow_profiles = compute_dow_profile([0.0, 0.05, 0.10, 0.20], 25,
                                        args.patients)

    t0 = time.time()
    results = run_simulation(weekend_fracs, date_counts,
                             args.patients, args.shift)
    elapsed = time.time() - t0
    print(f"\n  Simulation completed in {elapsed:.1f}s")

    make_plots(results, weekend_fracs, date_counts, dow_profiles, args.output)

    # Summary tables
    header = f"{'Wkend%':>7}"
    for nd in date_counts:
        header += f" {nd:>6}d"

    print(f"\n{'='*80}")
    print("SUMMARY: Bayesian DOW correct recovery (%)")
    print(f"{'='*80}")
    print(header)
    print("-" * len(header))
    for wf in weekend_fracs:
        row = f"{wf*100:>6.1f}%"
        for nd in date_counts:
            row += f" {results[(wf, nd)]['dow_correct_pct']:>6.1f}"
        print(row)

    for tau_label, tau in [('@90%', 0.90), ('@95%', 0.95)]:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Selective attacker {tau_label} — patients correctly compromised (%)")
        print(f"{'='*80}")
        print(header)
        print("-" * len(header))
        for wf in weekend_fracs:
            row = f"{wf*100:>6.1f}%"
            for nd in date_counts:
                row += f" {results[(wf, nd)]['selective'][tau]['correct_pct']:>6.1f}"
            print(row)

    print("\n=== Done. ===")


if __name__ == '__main__':
    main()
