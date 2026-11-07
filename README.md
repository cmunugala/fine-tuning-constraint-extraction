# Scheduling Constraint Synthetic Data

This repository includes a deterministic synthetic-data pipeline for training a
model to turn manager requests into labels defined by `SchedulingOutput` in
`models.py`. The existing Pydantic models are imported without modification.

## Review the preview first

Generate the 20 representative examples:

```bash
python -m synthetic_data preview --seed 20261102 --output-dir data
```

Inspect `data/preview.jsonl`. Each line is independently loadable JSON with this
shape:

```json
{"text":"Jordan can't work every Monday after 4 pm.","reference_date":"2026-11-02","label":{"constraints":[{"employee":"Jordan","type":"unavailable","priority":"hard","time_scope":{"days_of_week":["monday"],"start_time":"16:00:00","recurring":true}}]}}
```

The `reference_date` is the interpretation anchor for relative phrases such as
“tomorrow” or “in three days.” “Next Tuesday” means the first Tuesday strictly
after that date. Optional null label fields are omitted.

## Generate the full dataset

After manually approving the preview, run:

```bash
python -m synthetic_data generate --seed 20261102 --output-dir data --approved-preview
```

This writes `train.jsonl` (1,000 rows), `val.jsonl` (150 rows),
`test.jsonl` (200 rows), and `dataset_manifest.json`. The command refuses to run
without both the approval flag and an existing valid 20-row preview.

Validate existing outputs with:

```bash
python -m synthetic_data validate --input-dir data
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The final splits use disjoint phrasing-template banks. Thirty percent of test
examples use constraint combinations held out of train and validation, and the
pipeline rejects duplicate normalized requests or canonical labels across
splits.
