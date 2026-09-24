# Synthetic demonstration cohort

The analysis scripts in this repository read four CSV extracts that, at our
site, are derived from a PCORnet-model clinical data warehouse. Those extracts
are protected health information and cannot be distributed. To allow the
pipeline to be run and inspected without them, `make_synthetic_cohort.py`
generates a fully synthetic cohort in exactly the same format.

```bash
python make_synthetic_cohort.py -o synthetic_demo -n 10000
```

Runs in under a minute and is deterministic: the default seed (`--seed
20260915`) reproduces the numbers below exactly.

## Files

| File | Rows (n = 10,000) | Columns |
|---|---|---|
| `extract_outpatient_encounters.csv` | 165,577 | `PATID`, `event_date`, `ENC_TYPE`, `RAW_ENC_TYPE` |
| `extract_outpatient_procedures.csv` | 58,304 | `PATID`, `event_date`, `PX`, `PX_TYPE` |
| `extract_outpatient_labs.csv` | 7,258 | `PATID`, `order_date`, `result_date`, `lab_name`, `LAB_LOINC` |
| `extract_demographics.csv` | 10,000 | `PATID`, `BIRTH_DATE`, `SEX` |
| `synthetic_cohort_profile.json` | — | the aggregate parameters used for generation |

Dates are **true (unshifted)** dates in `YYYY-MM-DD` format over a five-year
window; each analysis script applies its own shift. If you supply your own
extracts instead, only the columns above are read — any extra columns are
ignored.

## What the cohort is

Patients, encounters, laboratory tests, procedures, codes and demographics are
invented. No record-level data, and no derivative of record-level data, is used
or distributed. The generator is driven entirely by aggregate parameters — a
date-count survival table, record-span quantiles, day-of-week weights and
holiday suppression factors — transcribed from the statistics reported in the
paper.

It reproduces the three properties that drive unshifting risk:

| Property | Synthetic | Case-study CDW |
|---|---|---|
| Patients with ≥ 4 distinct dates | 71.5 % | 71.7 % |
| ≥ 10 distinct dates | 40.5 % | 41.0 % |
| ≥ 17 distinct dates | 26.5 % | 26.9 % |
| ≥ 60 distinct dates | 5.7 % | 5.5 % |
| ≥ 100 distinct dates | 1.4 % | 1.4 % |
| Weekend event rate | 2.2 % | ~2 % |
| Event density on closure days | 0.14 × a normal day | 0.01–0.35 × by event type |

It does **not** reproduce clinical content, coding realism, or the correlation
structure of real scheduling (fixed-interval follow-up, clinic-specific day
preferences). Results from it demonstrate that the pipeline runs and behaves as
described; they are not a substitute for the case-study results and not a risk
assessment of any real dataset.

## Testing the tools

```bash
python layer1_patient_vulnerability.py synthetic_demo
PYTHONHASHSEED=0 python layer4_attack.py synthetic_demo --shift 30 365 --workers 8
python layer4_selective_attack.py synthetic_demo --shift 30 365
python layer5_attack.py synthetic_demo
```

`PYTHONHASHSEED=0` is required for reproducibility: the per-patient true shift
is seeded from `hash(PATID)`, and Python salts string hashing per process, so
results otherwise vary by about one percentage point between runs.

### Expected output

`layer1_patient_vulnerability.py` — all nine scenarios run; the combined
outpatient rows are:

```
 Combined Outpatient (labs=ORDER_DATE)   10000  71.51  55.57  26.48  40.52  21.49  5.71  1.42
Combined Outpatient (labs=RESULT_DATE)   10000  71.75  55.98  26.66  40.78  21.84  5.80  1.49
```

`layer4_attack.py`:

```
30 days    Argmax = true shift: 2,235 (22.35%)   uniquely recovered 295 (2.95%)   Tier 1: 7.2%
           100+ dates: P(max) = 0.93, argmax correct 85.2%
365 days   Argmax = true shift:   257 ( 2.57%)   uniquely recovered   3 (0.03%)   Tier 1: 0.5%
           100+ dates: P(max) = 0.54, argmax correct 40.1%
```

These fall within a point or two of the corresponding case-study values
(22.4 % and 2.7 %), which is the intended check: the synthetic cohort is close
enough to exercise the code and show the expected behaviour, not to reproduce
the published risk estimates.

The line `*** WARNING: true shift eliminated for N patients ***` printed by the
hard-elimination model is expected, not an error. It counts patients with a
genuine weekend or holiday event, which the hard model cannot accommodate; the
Bayesian model handles them correctly.

## Other uses

```bash
# organizations with more weekend activity (cf. Fig. 3)
python make_synthetic_cohort.py -o sim_w10 --weekend-rate 0.10

# derive an institution-matched profile from your own per-patient date counts.
# The profile holds aggregate quantities only (survival table, span quantiles);
# no record-level data leaves the site.
python make_synthetic_cohort.py --calibrate layer1_patient_date_counts.csv \
       --write-profile site_profile.json --no-generate
python make_synthetic_cohort.py --profile site_profile.json -n 20000 -o syn_site
```
