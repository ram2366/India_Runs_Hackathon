#!/usr/bin/env python3
"""
Print a full, human-readable candidate profile by candidate_id.
Useful during manual labeling and top-30 audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import rank
import v2_common as v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Show candidate profile and V1/V2 features.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()

    for scored in v2.stream_scored_candidates(args.candidates):
        if scored["candidate_id"] != args.candidate_id:
            continue
        c = scored["candidate"]
        p = c["profile"]
        print("=" * 90)
        print(f"{scored['candidate_id']} | {p.get('current_title')} | {p.get('years_of_experience')} yrs")
        print(f"{p.get('current_company')} | {p.get('current_industry')} | {p.get('location')} | {p.get('country')}")
        print(f"V1 score={scored['score']:.6f} fake_risk={scored['fake_risk']:.3f}")
        print("=" * 90)
        print("\nHEADLINE")
        print(p.get("headline", ""))
        print("\nSUMMARY")
        print(p.get("summary", ""))
        print("\nCAREER HISTORY")
        for job in c.get("career_history") or []:
            print("-" * 90)
            print(
                f"{job.get('title')} at {job.get('company')} | {job.get('duration_months')} months | "
                f"{job.get('start_date')} to {job.get('end_date') or 'present'}"
            )
            print(job.get("description", ""))
        print("\nSKILLS")
        for skill in c.get("skills") or []:
            print(
                f"- {skill.get('name')} | {skill.get('proficiency')} | "
                f"{skill.get('duration_months', 0)}m | endorsements={skill.get('endorsements', 0)}"
            )
        print("\nREDROB SIGNALS")
        print(json.dumps(c.get("redrob_signals") or {}, indent=2))
        print("\nFEATURES")
        print(json.dumps(v2.feature_dict(scored), indent=2))
        print("\nRISK REASONS")
        print("; ".join(scored.get("risk_reasons") or []) or "none")
        return

    raise SystemExit(f"Candidate not found: {args.candidate_id}")


if __name__ == "__main__":
    main()
