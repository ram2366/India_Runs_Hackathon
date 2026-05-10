# Redrob Candidate Intelligence Ranker

An offline ML ranking system that finds the best-fit candidates for an AI / ML hiring requirement and returns the top 100 profiles with scores and clear reasoning.

The solution is designed for candidate discovery where ranking quality, evidence, profile safety, and explainability matter.

## What This System Does

The system reads candidate profiles, extracts important hiring signals, detects weak or risky profiles, ranks candidates, and generates a final CSV with the strongest candidates at the top.

It focuses on:

- AI / ML / search / ranking / recommendation experience
- 5 to 10 years of relevant work experience
- production ML system ownership
- retrieval, ranking, relevance, NLP, LLM, and recommendation evidence
- evaluation experience such as NDCG, MRR, recall@K, A/B testing, and offline-online evaluation
- India or relocation fit
- fake-profile and low-quality-profile risk
- deterministic reasoning for every selected candidate

The final ranking does not depend on live LLM calls, paid APIs, GPUs, or cloud services. It runs locally using feature engineering, reviewed labels, a trained reranker, and deterministic reasoning generation.

## Final Output

Main output file:

```text
final/team_redrob_ranker_v2_final.csv
```

Supporting output files:

```text
final/audit_top300_v2_final.csv
final/qa_report_final.json
final/top40_review_report.csv
final/boundary_review_ranks_35_45.csv
final/top40_review_summary.json
```

Documentation PDFs:

```text
final/Redrob_Filled_Idea_Submission_PPT.pdf
final/Redrob_Integrated_Solution_Approach.pdf
final/Redrob_Idea_Submission_PPT_Content.pdf
```

## Quality Snapshot

- Final output rows: 100
- Validator status: passed
- Candidates under 5 years in top 100: 0
- Candidates over 10 years in top 100: 0
- Candidates with fake risk greater than 0.10 in top 100: 0
- Non-target titles in top 100: 0
- Unique reasoning strings: 100
- Top 40 reviewed for high precision
- Top 40 reviewed labels: 17 label-5 candidates and 23 label-4 candidates

Model metrics:

- NDCG@10: 0.904
- NDCG@50: 0.968
- MAP for label >= 3: 0.976
- Pairwise accuracy: 0.961
- Precision@10 for label >= 3: 1.000

## How The Ranking Works

### 1. Candidate Parsing

Each profile is converted into structured signals:

- title
- years of experience
- company
- location
- skills
- summary
- career history
- availability signals
- response rate
- GitHub strength
- evidence phrases from the profile

### 2. Feature Extraction

The system extracts problem-specific features such as:

- title fit
- experience-band fit
- search / retrieval evidence
- ranking / recommendation evidence
- production system evidence
- evaluation evidence
- LLM / NLP relevance
- location and relocation fit
- profile completeness
- fake-risk score
- career depth
- availability strength

### 3. Fake-Profile And Risk Detection

The fake-risk layer penalizes candidates whose profiles look weak, inconsistent, or hard to trust.

It checks signals such as:

- missing career history
- generic profile text
- weak evidence for claimed seniority
- title and experience mismatch
- incomplete profile details
- suspiciously thin AI / ML evidence
- low-quality or inconsistent profile signals

High-risk candidates are pushed down so the top results stay safer and more reliable.

### 4. First-Stage Scoring

`rank.py` creates the first ranking using deterministic scoring.

It rewards candidates who have:

- strong JD relevance
- ranking, retrieval, search, or recommendation ownership
- production ML experience
- measurable evaluation work
- correct experience range
- strong location and availability fit
- low fake-risk score

### 5. Reviewed Training Data

The model uses 520 reviewed candidate examples stored in:

```text
training_data.csv
```

Each example has a label that teaches the system whether the candidate is a strong, acceptable, weak, or risky fit for the role.

### 6. Pairwise Reranking

`v2_train_reranker.py` trains a lightweight pairwise reranker.

The reranker learns ordering decisions like:

```text
Candidate A should rank above Candidate B
```

This is useful because the real objective is not just scoring candidates independently. The important goal is putting the best candidates above the weaker ones.

### 7. Final Ranking

`v2_rank.py` applies the trained model and creates:

- final top 100 CSV
- top 300 audit file
- score for every selected candidate
- deterministic reasoning for every selected candidate

The reasoning is generated from extracted candidate facts and evidence, not from a live LLM.

### 8. QA Checks

`v2_qa.py` checks the final output for:

- correct row count
- validator pass
- experience-band leakage
- fake-risk leakage
- non-target titles
- duplicate reasoning
- weak reasoning
- top-candidate quality

## Repository Structure

```text
.
├── final/
│   ├── team_redrob_ranker_v2_final.csv
│   ├── audit_top300_v2_final.csv
│   ├── qa_report_final.json
│   ├── top40_review_report.csv
│   ├── boundary_review_ranks_35_45.csv
│   ├── top40_review_summary.json
│   ├── Redrob_Filled_Idea_Submission_PPT.pdf
│   ├── Redrob_Integrated_Solution_Approach.pdf
│   └── Redrob_Idea_Submission_PPT_Content.pdf
├── code/
│   ├── rank.py
│   ├── v2_common.py
│   ├── v2_train_reranker.py
│   ├── v2_rank.py
│   ├── v2_qa.py
│   ├── v2_prepare_labels.py
│   ├── v2_show_candidate.py
│   ├── demo_app.py
│   └── validate_submission.py
├── docs/
│   ├── V2_WORKFLOW.md
│   ├── labeling_guidelines.md
│   ├── requirements.txt
│   └── requirements-demo.txt
├── model_consensus_reviewed.json
├── training_data.csv
├── submission_metadata.yaml
├── DATASET_NOTE.md
└── UPLOAD_MANIFEST.md
```

## Important Files

| File | Purpose |
| --- | --- |
| `final/team_redrob_ranker_v2_final.csv` | Final ranked top 100 candidates |
| `final/audit_top300_v2_final.csv` | Audit view with features, evidence, and risk signals |
| `final/qa_report_final.json` | QA summary for the final output |
| `final/top40_review_report.csv` | High-precision review of the top 40 candidates |
| `model_consensus_reviewed.json` | Trained reranker model |
| `training_data.csv` | Reviewed labels used for training |
| `code/rank.py` | Feature extraction, first-stage scoring, fake-risk logic |
| `code/v2_train_reranker.py` | Pairwise reranker training |
| `code/v2_rank.py` | Final ranking generation |
| `code/v2_qa.py` | QA checks |
| `code/validate_submission.py` | Output format validator |

## Validate The Final Output

Run this from the repository root:

```bash
python3 code/validate_submission.py final/team_redrob_ranker_v2_final.csv
```

Expected output:

```text
Submission is valid.
```

## Reproduce The Ranking

The full candidate dataset is not included in this repository. Place `candidates.jsonl` in the repository root before running the pipeline.

Generate the final ranking:

```bash
python3 code/v2_rank.py \
  --candidates candidates.jsonl \
  --model model_consensus_reviewed.json \
  --out final/team_redrob_ranker_v2_final.csv \
  --audit final/audit_top300_v2_final.csv \
  --audit-limit 300
```

Run QA:

```bash
python3 code/v2_qa.py \
  --submission final/team_redrob_ranker_v2_final.csv \
  --audit final/audit_top300_v2_final.csv \
  --validator code/validate_submission.py \
  --out final/qa_report_final.json
```

Retrain the reranker:

```bash
python3 code/v2_train_reranker.py \
  --candidates candidates.jsonl \
  --labels training_data.csv \
  --model model_consensus_reviewed.json \
  --metrics training_metrics.json \
  --predictions labeled_predictions.csv
```

## Why This Solution Is Strong

The solution is focused on ranking quality instead of adding unnecessary features.

Its strongest parts are:

- problem-specific feature engineering
- fake-profile filtering
- reviewed training labels
- pairwise reranking
- top-40 precision audit
- deterministic reasoning
- reproducible offline pipeline
- final validation pass

The goal is simple: put the most relevant, safest, and most explainable candidates at the top.