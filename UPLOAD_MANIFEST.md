# Upload Manifest

Use this folder for GitHub/submission. It contains only the final submission files, reproducible code, final model/training artifacts, and concise documentation.

Most important file: `final/team_redrob_ranker_v2_final.csv`

Validation command:

```bash
python3 code/validate_submission.py final/team_redrob_ranker_v2_final.csv
```

Optional QA command:

```bash
python3 code/v2_qa.py --submission final/team_redrob_ranker_v2_final.csv --audit final/audit_top300_v2_final.csv --validator code/validate_submission.py --out final/qa_report_final.json
```

Notes:
- `candidates.jsonl` is not included because it is challenge-provided data. See `DATASET_NOTE.md`.
- Old experiments, duplicate CSVs, preview PNGs, and local system files are intentionally excluded.
