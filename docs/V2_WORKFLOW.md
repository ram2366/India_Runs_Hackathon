# Redrob V2 Workflow

V2 turns the V1 rule-based ranker into a calibrated ranking system.

## What V1 Is

V1 is a fast rule-based ranker:

- reads 100,000 candidates
- scores title, experience, career evidence, production ML, ranking/retrieval, behavior, location, company, education
- detects fake/keyword-stuffed profiles
- writes a valid top-100 CSV

V1 is strong, but its weights are hand-written.

## What V2 Adds

V2 adds a learning loop:

1. Generate a candidate labeling pack.
2. You manually label candidates from 0 to 5.
3. Train a small pairwise reranker.
4. Run V2 ranking.
5. QA the output.
6. Manually audit top 30 through `manual_overrides.csv`.
7. Generate final CSV.

## Step 1: Generate Label Pack

```bash
python3 Project/v2_prepare_labels.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --out Project/v2/labels_to_fill.csv
```

This creates:

- `Project/v2/labels_to_fill.csv`
- `Project/v2/labeling_profiles.jsonl`
- `Project/v2/labeling_guidelines.md`

## Step 2: Manual Work You Must Do

Open:

```text
Project/v2/labels_to_fill.csv
```

Only fill:

- `label`
- `label_notes`

Label scale:

```text
5 = perfect / dream candidate
4 = strong candidate
3 = acceptable candidate
2 = weak candidate
1 = bad candidate
0 = fake/trap
```

Minimum:

- 150 labels: usable first V2
- 300 labels: good
- 500 labels: strong

Do not label only good candidates. Label top candidates, borderline candidates, random candidates, and fake-looking profiles.

## Step 3: Train Reranker

```bash
python3 Project/v2_train_reranker.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --labels Project/v2/labels_to_fill.csv \
  --model Project/v2/model.json \
  --metrics Project/v2/training_metrics.json
```

The model is a small offline pairwise linear reranker. It learns which feature patterns should rank above others.

## Step 4: Run V2 Ranking

```bash
python3 Project/v2_rank.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --model Project/v2/model.json \
  --out Project/v2/team_redrob_ranker_v2.csv \
  --audit Project/v2/audit_top300_v2.csv
```

## Step 5: QA

```bash
python3 Project/v2_qa.py \
  --submission Project/v2/team_redrob_ranker_v2.csv \
  --audit Project/v2/audit_top300_v2.csv
```

Check:

- official validator passes
- no non-AI titles in top 100
- fake risk stays low
- top 10 are genuinely excellent
- reasoning is factual

## Step 6: Top-30 Manual Audit

Read full profiles for ranks 1-30 in:

```text
Project/v2/audit_top300_v2.csv
```

If a candidate should be removed or adjusted, edit:

```text
Project/v2/manual_overrides.csv
```

Examples:

```csv
candidate_id,reject,score_adjustment,note
CAND_1234567,true,0.0,"Reject: career history does not prove ranking/retrieval ownership."
CAND_7654321,false,0.05,"Boost: clear ranking/eval ownership and strong logistics."
```

Then rerun V2 ranking and QA.

## Step 7: Submission Packaging

Before upload, fill:

```text
Project/submission_metadata.yaml
```

Important TODO fields:

- team name
- contact details
- GitHub repo
- sandbox link
- compute environment

For a simple hosted sandbox, use:

```text
Project/demo_app.py
Project/requirements-demo.txt
```

On Streamlit Cloud or HuggingFace Spaces, configure the app entry point as:

```bash
streamlit run Project/demo_app.py
```

The sandbox is only for small sample ranking. The official full ranking command
remains `Project/v2_rank.py`.

## Why This Improves Winning Chance

The hackathon score is top-heavy:

- NDCG@10 matters most
- NDCG@50 matters second
- fake profiles can disqualify
- reasoning is manually reviewed
- code must reproduce under CPU/no-network constraints

V2 directly targets those criteria:

- labels improve calibration
- pairwise training improves ordering
- fake-risk layer protects against honeypots
- top-30 audit protects NDCG@10
- structured reasoning helps Stage 4 review
- all decisions are reproducible in code/config
