#!/usr/bin/env python3
"""
QA checks for V2 submission and audit files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True


NON_TARGET_TITLES = {
    "HR Manager",
    "Marketing Manager",
    "Sales Executive",
    "Accountant",
    "Civil Engineer",
    "Mechanical Engineer",
    "Graphic Designer",
    "Content Writer",
    "Operations Manager",
    "Customer Support",
    "Project Manager",
    "Business Analyst",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_validator(validator: Path, submission: Path) -> dict[str, str | int]:
    proc = subprocess.run(
        [sys.executable, str(validator), str(submission)],
        text=True,
        capture_output=True,
        check=False,
    )
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="QA V2 submission.")
    parser.add_argument("--submission", default=Path("Project/v2/team_redrob_ranker_v2.csv"), type=Path)
    parser.add_argument("--audit", default=Path("Project/v2/audit_top300_v2.csv"), type=Path)
    parser.add_argument("--validator", default=Path("India_runs_data_and_ai_challenge/validate_submission.py"), type=Path)
    parser.add_argument("--out", default=Path("Project/v2/qa_report.json"), type=Path)
    args = parser.parse_args()

    submission = read_csv(args.submission)
    audit = read_csv(args.audit)
    top100_audit = audit[:100]

    years = [float(r["profile_years"]) for r in top100_audit if r.get("profile_years")]
    risks = [float(r["fake_risk"]) for r in top100_audit if r.get("fake_risk")]
    titles = Counter(r["current_title"] for r in top100_audit)
    non_target = [r for r in top100_audit if r["current_title"] in NON_TARGET_TITLES]
    high_risk = [r for r in top100_audit if float(r.get("fake_risk") or 0) > 0.10]
    reasonings = [r.get("reasoning", "") for r in submission]
    unique_reasonings = len(set(reasonings))
    short_reasonings = sum(1 for r in reasonings if len(r) < 60)

    validator_result = run_validator(args.validator, args.submission)

    report = {
        "submission": str(args.submission),
        "audit": str(args.audit),
        "validator": validator_result,
        "row_count": len(submission),
        "audit_top100_count": len(top100_audit),
        "years": {
            "min": min(years) if years else None,
            "median": statistics.median(years) if years else None,
            "max": max(years) if years else None,
            "under_5": sum(y < 5 for y in years),
            "over_10": sum(y > 10 for y in years),
        },
        "fake_risk": {
            "max": max(risks) if risks else None,
            "mean": statistics.mean(risks) if risks else None,
            "count_gt_0_10": len(high_risk),
            "candidate_ids_gt_0_10": [r["candidate_id"] for r in high_risk],
        },
        "titles": titles.most_common(),
        "non_target_titles_in_top100": [r["candidate_id"] for r in non_target],
        "reasoning": {
            "unique_count": unique_reasonings,
            "short_count_lt_60_chars": short_reasonings,
        },
        "manual_review_recommendation": [
            "Read full profiles for ranks 1-30 before final submission.",
            "Reject or down-adjust any candidate whose career history does not prove AI/search/ranking fit.",
            "Check that each top-30 reasoning sentence is factual and rank-consistent.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
