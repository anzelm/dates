# Date-Shift Reidentification: Attack Simulation and Safety Assessment

Reference implementation accompanying *"Leveraging Routine Scheduling Patterns to
Unshift Date-Shifted Clinical Data."* The code quantifies the reidentification
risk that weekday and holiday scheduling patterns create for date-shifted clinical
data, using a Bayesian inference model over candidate shifts.

All four scripts operate **entirely on local CSV extracts** — no database access,
no credentials, and no PHI are required or included. The scripts assume the extract
files already exist in the working directory.

## Contents

| Script | Role |
|--------|------|
| `layer1_patient_vulnerability.py` | Structural screen: distinct-date counts per patient across nine dataset scenarios; reports the fraction crossing day-of-week and full-recovery thresholds. |
| `layer4_attack.py` | Core Bayesian attack simulation. Computes a posterior over candidate shifts per patient and reports per-patient vulnerability metrics (entropy, information gain, recovery probability). |
| `layer4_selective_attack.py` | Selective-attacker variant. Same posterior, evaluated as a targeting strategy: precision and population fraction compromised as a function of a confidence threshold. |
| `layer5_attack.py` | Defensive safety assessment. Derives weekday/holiday rates empirically from the data, then runs the attack to classify patients into risk tiers for a proposed shift range. |

A natural progression is `layer1` (is there a signal to exploit?) → `layer4` (how
much can an attacker recover?) → `layer4_selective` (how many can a rational
attacker recover confidently?) → `layer5` (how safe is a given shift range before
release?).

## Requirements

- Python 3.8+
- `pandas`, `numpy` (standard library otherwise: `multiprocessing`, `argparse`,
  `datetime`, `collections`, `os`, `sys`, `time`)

```bash
pip install pandas numpy
```

## Input files

Place these CSV extracts in the working directory (or pass its path as the first
argument). The scripts read only the columns listed; extra columns are ignored.

| File | Columns used |
|------|--------------|
| `extract_outpatient_encounters.csv` | `PATID`, `event_date`; plus `ENC_TYPE`, `RAW_ENC_TYPE` (Layer 1 office-visit scenario and Layer 5 rate estimation) |
| `extract_outpatient_labs.csv` | `PATID`, `order_date`, `result_date`; plus `lab_name` or `LAB_LOINC` (Layer 1 chronic-disease scenario and Layer 5 rate estimation) |
| `extract_outpatient_procedures.csv` | `PATID`, `event_date`; plus `RAW_PX` or `PX` (Layer 5 rate estimation) |
| `extract_demographics.csv` | `BIRTH_DATE` (Layer 1 only; optional — skipped if absent) |

Date columns are parsed as dates on load. `PATID` is used only as a grouping key
and carries no meaning outside the extract. `layer5_attack.py` sniffs each labs and
procedures file for the richer columns and falls back gracefully (to date-only rate
estimation) when they are not present, so the attack still runs on minimal extracts.

## Running the scripts

Each script takes the extract directory as an optional first positional argument
(defaults to the script's own directory) and writes its outputs there.

### Layer 1 — structural vulnerability screen

```bash
python layer1_patient_vulnerability.py [data_dir]
```

No further flags. Internally iterates nine fixed dataset scenarios (encounters,
office visits, labs by order/result date, chronic-disease labs, procedures, and two
combined scenarios). Writes `layer1_patient_vulnerability_summary.csv` (per-scenario
threshold-crossing percentages) and `layer1_patient_date_counts.csv` (per-patient
distinct-date counts and record span, tagged by scenario).

### Layer 4 — Bayesian attack simulation

```bash
python layer4_attack.py [data_dir] [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--shift D [D ...]` | several | One or more shift ranges to evaluate, in days (e.g. `--shift 30 365 1826`). |
| `--workers N` | `min(32, cpu_count())` | Parallel worker processes. |
| `--sample N` | `0` (all) | Randomly sample N patients for a quick run. |
| `--scenario {combined_order,combined_result}` | `combined_order` | Which lab date column to fold into the combined date set. |
| `--p-weekday F` | `0.98` | Soft-model probability an outpatient event falls on a weekday. |
| `--p-holiday F` | `0.001` | Soft-model probability an outpatient event falls on an institutional closure day. |

Writes one `layer4_attack_<shift>d.csv` per shift range (per-patient metrics) and a
combined `layer4_attack_summary.csv` (stratified by date count and record span).

Quick smoke test:

```bash
python layer4_attack.py . --shift 30 --sample 10000 --workers 8
```

### Layer 4 (selective) — targeting strategy

```bash
python layer4_selective_attack.py [data_dir] [options]
```

Same flags and defaults as `layer4_attack.py` (`--shift`, `--workers`, `--sample`,
`--scenario`, `--p-weekday`, `--p-holiday`). Evaluates the posterior as a selective
attack: for a range of confidence thresholds it reports precision (of targeted
patients, how many are correctly reidentified) and the fraction of the total
population compromised. Writes per-shift result CSVs and a summary.

### Layer 5 — pre-release safety assessment

```bash
python layer5_attack.py [data_dir] [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--shift D [D ...]` | several | Shift range(s) to assess, in days. |
| `--workers N` | `min(32, cpu_count())` | Parallel worker processes. |
| `--sample N` | `0` (all) | Sample N patients for a quick run. |
| `--scenario {combined_order,combined_result}` | `combined_order` | Lab date column folded into the combined date set. |
| `--p-holiday F` | `0.001` | Probability of an event on an institutional closure day. |
| `--p-weekday-naive F` | `0.98` | Fallback weekday probability, used where an empirical per-type rate cannot be estimated from the data. |

Note the flag is `--p-weekday-naive` here, not `--p-weekday`: Layer 5 first derives
weekday and holiday rates empirically from the supplied extracts (writing
`layer5_weekday_rates.csv`), and only falls back to the naive value where a rate
cannot be computed. It then runs the attack and classifies each patient into a risk
tier (Low / Moderate / High / Critical) by posterior confidence, writing one
`layer5_attack_<shift>d.csv` per shift range with the per-patient tier assignment.

## Method summary

The custodian applies a constant backward shift *s\** drawn uniformly from
{1, …, *S*}. The attacker observes only the shifted dates and *S*. For each
candidate shift *m*, reconstructed dates are scored by how well they match expected
weekday/holiday patterns; a uniform prior yields a posterior over candidates. The
day-of-week constraint alone confines survivors to a single residue class modulo 7
(≈ *S*/7 candidates); holidays falling within the record span further disambiguate.
Per-patient confidence is the posterior mass on its most likely candidate. The full
model, including the entropy and information-gain metrics and the holiday-elimination
approximation, is given in the manuscript and its appendix.

The same computation serves both roles: run by an adversary it is an attack; run by
a custodian before release it is a risk-assessment tool that identifies which
patients a given shift range fails to protect.

## Notes and constraints

- Extracts are read-only inputs; the scripts never write back to them.
- No credentials, connection strings, or extraction (database) code are part of this
  release. The scripts begin from the CSV extracts and go no further upstream.
- Runtime scales with patient count × shift range × worker count. Large shift ranges
  (e.g. 1826 days) over the full cohort are the heaviest configurations; use
  `--sample` to size a run before committing to the full population.
- `PATID` values are opaque grouping keys with no external meaning.
