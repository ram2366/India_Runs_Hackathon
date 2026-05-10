#!/usr/bin/env python3
"""
Shared utilities for the V2 Redrob ranker.

V2 keeps the V1 ranker as the feature engine, then optionally applies a small
trained linear reranker learned from manual labels. The final ranking path stays
CPU-only and offline.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rank  # noqa: E402


FEATURE_NAMES = [
    "v1_score",
    "fit_score",
    "fake_risk",
    "title",
    "years",
    "retrieval",
    "ranking",
    "evaluation",
    "production",
    "relevance_system",
    "shipper",
    "llm",
    "skills",
    "company",
    "location",
    "behavior",
    "education",
    "raw_years_ideal",
    "raw_years_senior_band",
    "response_rate",
    "notice_fast",
    "open_to_work",
    "india_or_relocate",
    "target_city",
    "verified_strength",
    "github_strength",
    "top_rank_signal",
    "career_depth_signal",
    "availability_signal",
    "risk_safe",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def feature_dict(scored: dict[str, Any]) -> dict[str, float]:
    candidate = scored["candidate"]
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    f = scored["features"]

    years = float(profile.get("years_of_experience") or 0.0)
    notice_days = float(signals.get("notice_period_days") or 180.0)
    country = rank.norm(profile.get("country"))
    location = rank.norm(profile.get("location"))
    city = location.split(",")[0].strip()
    willing = bool(signals.get("willing_to_relocate"))
    github_raw = float(signals.get("github_activity_score") or -1.0)

    verified_strength = (
        bool_float(signals.get("verified_email"))
        + bool_float(signals.get("verified_phone"))
        + bool_float(signals.get("linkedin_connected"))
    ) / 3.0

    features = {
        "v1_score": float(scored["score"]),
        "fit_score": float(scored["fit_score"]),
        "fake_risk": float(scored["fake_risk"]),
        "title": float(f["title"]),
        "years": float(f["years"]),
        "retrieval": float(f["retrieval"]),
        "ranking": float(f["ranking"]),
        "evaluation": float(f["evaluation"]),
        "production": float(f["production"]),
        "relevance_system": float(f["relevance_system"]),
        "shipper": float(f["shipper"]),
        "llm": float(f["llm"]),
        "skills": float(f["skills"]),
        "company": float(f["company"]),
        "location": float(f["location"]),
        "behavior": float(f["behavior"]),
        "education": float(f["education"]),
        "raw_years_ideal": clamp(1.0 - abs(years - 7.0) / 5.0),
        "raw_years_senior_band": 1.0 if 5.0 <= years <= 9.0 else 0.0,
        "response_rate": clamp(float(signals.get("recruiter_response_rate") or 0.0)),
        "notice_fast": clamp(1.0 - max(0.0, notice_days - 15.0) / 105.0),
        "open_to_work": bool_float(signals.get("open_to_work_flag")),
        "india_or_relocate": 1.0 if country == "india" or willing else 0.0,
        "target_city": 1.0 if city in rank.TARGET_LOCATIONS else 0.0,
        "verified_strength": verified_strength,
        "github_strength": 0.0 if github_raw < 0 else clamp(github_raw / 100.0),
        "top_rank_signal": clamp(
            0.30 * float(f["title"])
            + 0.25 * float(f["relevance_system"])
            + 0.20 * float(f["ranking"])
            + 0.15 * float(f["production"])
            + 0.10 * float(f["evaluation"])
        ),
        "career_depth_signal": clamp(
            0.30 * float(f["relevance_system"])
            + 0.25 * float(f["retrieval"])
            + 0.20 * float(f["ranking"])
            + 0.15 * float(f["production"])
            + 0.10 * float(f["shipper"])
        ),
        "availability_signal": clamp(
            0.40 * float(f["behavior"])
            + 0.25 * clamp(float(signals.get("recruiter_response_rate") or 0.0))
            + 0.20 * (1.0 if signals.get("open_to_work_flag") else 0.0)
            + 0.15 * clamp(1.0 - max(0.0, notice_days - 15.0) / 105.0)
        ),
        "risk_safe": 1.0 - float(scored["fake_risk"]),
    }
    return features


def feature_vector(scored: dict[str, Any]) -> list[float]:
    features = feature_dict(scored)
    return [float(features[name]) for name in FEATURE_NAMES]


def load_model(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature schema does not match current V2 feature schema.")
    return model


def model_score(scored: dict[str, Any], model: dict[str, Any] | None) -> float:
    if not model or model.get("model_type") == "v1_fallback":
        return float(scored["score"])

    values = feature_vector(scored)
    means = model["means"]
    scales = model["scales"]
    weights = model["weights"]
    z = float(model.get("intercept", 0.0))
    for value, mean, scale, weight in zip(values, means, scales, weights):
        z += ((value - mean) / (scale or 1.0)) * weight

    # Keep a little V1 signal as a guardrail. This is especially useful when the
    # manual label set is small.
    blend_v1 = float(model.get("blend_v1", 0.20))
    return (1.0 - blend_v1) * z + blend_v1 * float(scored["score"])


def compact_text(text: str, limit: int = 480) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def candidate_summary(scored: dict[str, Any]) -> dict[str, str]:
    c = scored["candidate"]
    p = c["profile"]
    s = c["redrob_signals"]
    career_parts = []
    for job in c.get("career_history") or []:
        career_parts.append(
            f"{job.get('title', '')} at {job.get('company', '')} ({job.get('duration_months', 0)}m): "
            f"{job.get('description', '')}"
        )
    skills = sorted(
        c.get("skills") or [],
        key=lambda x: (
            int(x.get("endorsements") or 0) + int(x.get("duration_months") or 0),
            x.get("name", ""),
        ),
        reverse=True,
    )
    skill_text = ", ".join(
        f"{x.get('name')}:{x.get('proficiency')}/{x.get('duration_months', 0)}m"
        for x in skills[:12]
    )
    hits = scored.get("hits") or {}
    evidence = " | ".join(rank.choose_evidence(hits, limit=8))
    risk = "; ".join(scored.get("risk_reasons") or [])
    return {
        "candidate_id": c["candidate_id"],
        "current_title": p.get("current_title", ""),
        "years": f"{float(p.get('years_of_experience') or 0):.1f}",
        "company": p.get("current_company", ""),
        "industry": p.get("current_industry", ""),
        "location": p.get("location", ""),
        "headline": p.get("headline", ""),
        "summary": compact_text(p.get("summary", ""), 520),
        "career_history": compact_text(" || ".join(career_parts), 1400),
        "top_skills": compact_text(skill_text, 520),
        "evidence_hits": compact_text(evidence, 520),
        "risk_reasons": compact_text(risk, 360),
        "signals": (
            f"active={s.get('last_active_date')}; open_to_work={s.get('open_to_work_flag')}; "
            f"response={float(s.get('recruiter_response_rate') or 0):.2f}; "
            f"notice={s.get('notice_period_days')}; github={s.get('github_activity_score')}; "
            f"saved_30d={s.get('saved_by_recruiters_30d')}"
        ),
    }


def read_labels(path: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "candidate_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Labels CSV must include candidate_id and label columns.")
        for row in reader:
            cid = (row.get("candidate_id") or "").strip()
            label_s = (row.get("label") or "").strip()
            if not cid or not label_s:
                continue
            try:
                label = float(label_s)
            except ValueError:
                continue
            if 0 <= label <= 5:
                labels[cid] = label
    return labels


def stream_scored_candidates(candidates_path: Path):
    for line in rank.open_jsonl(candidates_path):
        if not line.strip():
            continue
        candidate = json.loads(line)
        yield rank.score_candidate(candidate)
