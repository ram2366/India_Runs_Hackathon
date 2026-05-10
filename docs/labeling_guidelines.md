# V2 Manual Labeling Guide

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
python3 Project/v2_train_reranker.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --labels Project/v2/labels_to_fill.csv \
  --model Project/v2/model.json \
  --metrics Project/v2/training_metrics.json

python3 Project/v2_rank.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --model Project/v2/model.json \
  --out Project/v2/team_redrob_ranker_v2.csv \
  --audit Project/v2/audit_top300_v2.csv
```
