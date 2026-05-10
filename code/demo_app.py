"""
Small Streamlit sandbox demo for Redrob ranker.

This is for the hackathon sandbox link, not for the official full ranking path.
It accepts a small JSONL candidate sample, ranks it with V1/V2 fallback logic,
and lets reviewers download a CSV.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import streamlit as st

import rank
import v2_common as v2


st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker Sandbox")
st.caption("CPU-only demo for small samples. Full competition ranking uses Project/v2_rank.py.")

model_path = Path("Project/v2/model.json")
model = v2.load_model(model_path) if model_path.exists() else None

uploaded = st.file_uploader("Upload sample candidates JSONL", type=["jsonl", "txt"])
if uploaded:
    rows = []
    for raw in uploaded.getvalue().decode("utf-8").splitlines():
        if not raw.strip():
            continue
        candidate = json.loads(raw)
        scored = rank.score_candidate(candidate)
        scored["v1_score"] = scored["score"]
        scored["score"] = v2.model_score(scored, model)
        rows.append(scored)
    rows.sort(key=lambda r: (-float(r["score"]), r["candidate_id"]))

    table_rows = []
    for idx, row in enumerate(rows[:100], start=1):
        p = row["candidate"]["profile"]
        table_rows.append(
            {
                "rank": idx,
                "candidate_id": row["candidate_id"],
                "score": round(float(row["score"]), 6),
                "title": p.get("current_title", ""),
                "years": p.get("years_of_experience", ""),
                "company": p.get("current_company", ""),
                "location": p.get("location", ""),
                "fake_risk": round(float(row["fake_risk"]), 3),
                "reasoning": rank.make_reasoning(row, idx),
            }
        )

    st.dataframe(table_rows, use_container_width=True)

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for item in table_rows:
        writer.writerow([item["candidate_id"], item["rank"], item["score"], item["reasoning"]])
    st.download_button("Download ranked CSV", out.getvalue(), file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Upload a small JSONL sample to run the ranker.")
