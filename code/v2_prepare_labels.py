#!/usr/bin/env python3
"""
Prepare a manual-label pack for V2 training.

This script samples candidates from useful buckets:
- V1 top candidates, because top-rank ordering matters most.
- Borderline candidates, because the model must learn cutoffs.
- Trap/fake-risk candidates, because honeypots must stay out.
- Random candidates, to avoid overfitting only to V1's worldview.

The user fills the `label` column:
5 = perfect, 4 = strong, 3 = acceptable, 2 = weak, 1 = bad, 0 = fake/trap.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import rank
import v2_common as v2


def heap_push(heap: list, key: float, scored: dict, limit: int) -> None:
    item = (key, scored["candidate_id"], scored)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def add_candidate(selected: dict[str, tuple[str, dict]], scored: dict, bucket: str) -> None:
    cid = scored["candidate_id"]
    if cid not in selected:
        selected[cid] = (bucket, scored)


def write_guidelines(path: Path) -> None:
    text = """# V2 Manual Labeling Guide

Fill `Project/v2/labels_to_fill.csv`.

Only edit these columns:
- `label`
- `label_notes`

Use this label scale:

| Label | Meaning | Simple test |
|---|---|---|
| 5 | Perfect / dream fit | I would defend this candidate in top 10. |
| 4 | Strong fit | Very relevant, maybe one concern. |
| 3 | Acceptable fit | Could be in top 100, but not top 30. |
| 2 | Weak fit | Some relevant signal, but likely below top 100. |
| 1 | Bad fit | Mostly wrong role or too weak. |
| 0 | Fake/trap | Keyword stuffing, impossible profile, or no career proof. |

How to label:
1. Read title, years, company, summary, career_history, evidence_hits, risk_reasons.
2. Trust career history more than skills.
3. Penalize AI keywords that do not appear in actual work.
4. Reward production search/ranking/retrieval/recommender/evaluation work.
5. Use behavior/logistics as tie-breakers, not as the main reason.

Good label examples:
- Senior ML Engineer, 7 years, built hybrid retrieval and learning-to-rank in production: `5`.
- Data Engineer, 6 years, mostly Spark/Airflow with some ML side projects: `2`.
- HR Manager with RAG/FAISS/Pinecone skills but no AI engineering career: `0`.

Minimum useful labels:
- 150 labels: enough for a first V2 sanity check.
- 300 labels: good.
- 500 labels: strong.

After labeling, run:

```bash
python3 Project/v2_train_reranker.py \\
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \\
  --labels Project/v2/labels_to_fill.csv \\
  --model Project/v2/model.json \\
  --metrics Project/v2/training_metrics.json

python3 Project/v2_rank.py \\
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \\
  --model Project/v2/model.json \\
  --out Project/v2/team_redrob_ranker_v2.csv \\
  --audit Project/v2/audit_top300_v2.csv
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V2 manual-label pack.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", default=Path("Project/v2/labels_to_fill.csv"), type=Path)
    parser.add_argument("--profiles-jsonl", default=Path("Project/v2/labeling_profiles.jsonl"), type=Path)
    parser.add_argument("--guidelines", default=Path("Project/v2/labeling_guidelines.md"), type=Path)
    parser.add_argument("--target", type=int, default=520)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    top_heap: list = []
    fake_heap: list = []
    keyword_heap: list = []
    plain_language_heap: list = []
    adjacent_heap: list = []
    random_reservoir: list[dict] = []
    seen_count = 0

    for scored in v2.stream_scored_candidates(args.candidates):
        seen_count += 1
        features = scored["features"]
        profile = scored["candidate"]["profile"]
        title = rank.norm(profile.get("current_title"))
        risk_text = " ".join(scored.get("risk_reasons") or [])

        heap_push(top_heap, float(scored["score"]), scored, 700)
        heap_push(fake_heap, float(scored["fake_risk"]), scored, 160)

        if "keyword stuffing" in risk_text:
            heap_push(keyword_heap, float(scored["fake_risk"]), scored, 120)

        if features["relevance_system"] >= 0.70:
            heap_push(plain_language_heap, float(scored["score"]), scored, 140)

        if any(x in title for x in ["data engineer", "backend", "software engineer", "analytics"]):
            if features["retrieval"] + features["ranking"] + features["production"] >= 1.20:
                heap_push(adjacent_heap, float(scored["score"]), scored, 120)

        if len(random_reservoir) < 120:
            random_reservoir.append(scored)
        else:
            j = rng.randrange(seen_count)
            if j < 120:
                random_reservoir[j] = scored

    top_ranked = [item[2] for item in sorted(top_heap, reverse=True)]
    selected: dict[str, tuple[str, dict]] = {}

    for scored in top_ranked[:260]:
        add_candidate(selected, scored, "v1_top_260")
    for scored in top_ranked[260:620:3]:
        add_candidate(selected, scored, "v1_borderline")
    for _, _, scored in sorted(plain_language_heap, reverse=True)[:80]:
        add_candidate(selected, scored, "plain_language_relevance")
    for _, _, scored in sorted(adjacent_heap, reverse=True)[:70]:
        add_candidate(selected, scored, "adjacent_but_relevant")
    for _, _, scored in sorted(fake_heap, reverse=True)[:80]:
        add_candidate(selected, scored, "high_fake_risk")
    for _, _, scored in sorted(keyword_heap, reverse=True)[:70]:
        add_candidate(selected, scored, "keyword_stuffer")
    for scored in random_reservoir[:90]:
        add_candidate(selected, scored, "random_background")

    rows = list(selected.values())[: args.target]
    headers = [
        "label",
        "label_notes",
        "selection_bucket",
        "candidate_id",
        "v1_score",
        "fake_risk",
        "current_title",
        "years",
        "company",
        "industry",
        "location",
        "headline",
        "summary",
        "career_history",
        "top_skills",
        "evidence_hits",
        "risk_reasons",
        "signals",
    ]

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for bucket, scored in rows:
            summary = v2.candidate_summary(scored)
            writer.writerow(
                {
                    "label": "",
                    "label_notes": "",
                    "selection_bucket": bucket,
                    "candidate_id": scored["candidate_id"],
                    "v1_score": f"{float(scored['score']):.6f}",
                    "fake_risk": f"{float(scored['fake_risk']):.3f}",
                    **summary,
                }
            )

    with args.profiles_jsonl.open("w", encoding="utf-8") as f:
        for bucket, scored in rows:
            c = scored["candidate"].copy()
            c["_selection_bucket"] = bucket
            c["_v1_score"] = scored["score"]
            c["_fake_risk"] = scored["fake_risk"]
            c["_features"] = scored["features"]
            c["_risk_reasons"] = scored["risk_reasons"]
            f.write(rank.json.dumps(c, ensure_ascii=False) + "\n")

    write_guidelines(args.guidelines)
    print(f"Wrote {len(rows)} candidates to label: {args.out}")
    print(f"Wrote full candidate profiles: {args.profiles_jsonl}")
    print(f"Wrote labeling guide: {args.guidelines}")


if __name__ == "__main__":
    main()
