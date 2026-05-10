#!/usr/bin/env python3
"""
Produce a V2 ranked submission.

Inputs:
- candidates.jsonl
- optional trained V2 model from v2_train_reranker.py
- optional manual overrides CSV for reproducible top-rank audit decisions

Output:
- top-100 submission CSV
- optional audit CSV
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import rank
import v2_common as v2


def load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("candidate_id") or "").strip()
            if not cid or cid.startswith("#"):
                continue
            reject_s = (row.get("reject") or "").strip().lower()
            adjustment_s = (row.get("score_adjustment") or "").strip()
            try:
                adjustment = float(adjustment_s) if adjustment_s else 0.0
            except ValueError:
                adjustment = 0.0
            overrides[cid] = {
                "reject": reject_s in {"1", "true", "yes", "y"},
                "score_adjustment": adjustment,
                "note": (row.get("note") or "").strip(),
            }
    return overrides


def apply_override(score: float, cid: str, overrides: dict[str, dict[str, Any]]) -> tuple[float, str]:
    item = overrides.get(cid)
    if not item:
        return score, ""
    if item.get("reject"):
        return -999.0, item.get("note", "manual reject")
    return score + float(item.get("score_adjustment") or 0.0), item.get("note", "")


def push_heap(heap: list, scored: dict[str, Any], keep: int) -> None:
    item = (float(scored["score"]), scored["candidate_id"], scored)
    if len(heap) < keep:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def write_submission(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, row in enumerate(rows[:100], start=1):
            writer.writerow(
                [
                    row["candidate_id"],
                    idx,
                    f"{float(row['score']):.12f}",
                    rank.make_reasoning(row, idx),
                ]
            )


def write_audit(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_headers = [f"feature_{name}" for name in v2.FEATURE_NAMES]
    headers = [
        "rank",
        "candidate_id",
        "v2_score",
        "v1_score",
        "fake_risk",
        "current_title",
        "profile_years",
        "company",
        "location",
        "manual_override_note",
        "risk_reasons",
        "evidence_hits",
    ] + feature_headers

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for idx, row in enumerate(rows[:limit], start=1):
            c = row["candidate"]
            p = c["profile"]
            features = v2.feature_dict(row)
            writer.writerow(
                {
                    "rank": idx,
                    "candidate_id": row["candidate_id"],
                    "v2_score": f"{float(row['score']):.9f}",
                    "v1_score": f"{float(row.get('v1_score', row['score'])):.9f}",
                    "fake_risk": f"{float(row['fake_risk']):.3f}",
                    "current_title": p.get("current_title", ""),
                    "profile_years": p.get("years_of_experience", ""),
                    "company": p.get("current_company", ""),
                    "location": p.get("location", ""),
                    "manual_override_note": row.get("manual_override_note", ""),
                    "risk_reasons": "; ".join(row.get("risk_reasons") or []),
                    "evidence_hits": " | ".join(rank.choose_evidence(row.get("hits") or {}, limit=8)),
                    **{f"feature_{name}": f"{features[name]:.6f}" for name in v2.FEATURE_NAMES},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 ranker.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--manual-overrides", type=Path, default=Path("Project/v2/manual_overrides.csv"))
    parser.add_argument("--out", default=Path("Project/v2/team_redrob_ranker_v2.csv"), type=Path)
    parser.add_argument("--audit", default=Path("Project/v2/audit_top300_v2.csv"), type=Path)
    parser.add_argument("--audit-limit", type=int, default=300)
    parser.add_argument("--keep", type=int, default=1200)
    args = parser.parse_args()

    model = v2.load_model(args.model) if args.model else None
    overrides = load_overrides(args.manual_overrides)
    heap: list = []
    for scored in v2.stream_scored_candidates(args.candidates):
        v1_score = float(scored["score"])
        new_score = v2.model_score(scored, model)
        new_score, note = apply_override(new_score, scored["candidate_id"], overrides)
        scored["v1_score"] = v1_score
        scored["score"] = new_score
        scored["manual_override_note"] = note
        push_heap(heap, scored, keep=max(100, args.keep))

    rows = [item[2] for item in heap]
    rows.sort(key=lambda r: (-float(r["score"]), r["candidate_id"]))
    rows = [row for row in rows if float(row["score"]) > -900.0]

    write_submission(rows, args.out)
    if args.audit:
        write_audit(rows, args.audit, args.audit_limit)
    print(f"Wrote V2 submission: {args.out}")
    print(f"Wrote V2 audit: {args.audit}")
    if model:
        print(f"Model type: {model.get('model_type')} labels={model.get('label_count', 0)}")
    else:
        print("No model supplied; V2 output used V1 scores.")


if __name__ == "__main__":
    main()
