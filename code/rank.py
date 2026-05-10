#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery ranker.

This ranker is intentionally CPU-only and standard-library-only for the
submission-time path. It streams the 100K JSONL file, extracts structured
evidence from each profile, applies a JD-specific scoring rubric, subtracts
fake-profile / honeypot risk, and writes the required top-100 CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 6, 9)


SERVICE_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "ltimindtree",
    "hcl",
    "tech mahindra",
    "mphasis",
    "persistent",
}

PRODUCT_COMPANIES = {
    "google",
    "meta",
    "facebook",
    "amazon",
    "microsoft",
    "apple",
    "netflix",
    "linkedin",
    "uber",
    "airbnb",
    "stripe",
    "salesforce",
    "adobe",
    "atlassian",
    "zomato",
    "swiggy",
    "flipkart",
    "razorpay",
    "cred",
    "paytm",
    "meesho",
    "ola",
    "inmobi",
    "sarvam ai",
    "krutrim",
    "yellow.ai",
    "gupshup",
    "wysa",
    "niramai",
    "mad street den",
    "rephrase.ai",
    "aganitha",
    "freshworks",
    "darwinbox",
    "fractal analytics",
}

TARGET_LOCATIONS = {
    "pune": 1.00,
    "noida": 1.00,
    "gurgaon": 0.90,
    "delhi": 0.88,
    "mumbai": 0.86,
    "hyderabad": 0.84,
    "bangalore": 0.84,
    "bengaluru": 0.84,
}

NON_TARGET_TITLE_RE = re.compile(
    r"\b("
    r"hr manager|marketing manager|sales executive|accountant|civil engineer|"
    r"mechanical engineer|graphic designer|content writer|operations manager|"
    r"customer support|project manager|business analyst|qa engineer|frontend engineer|"
    r"mobile developer"
    r")\b",
    re.I,
)

TARGET_TITLE_RE = re.compile(
    r"\b("
    r"senior ai engineer|lead ai engineer|staff machine learning engineer|"
    r"senior machine learning engineer|senior applied scientist|senior nlp engineer|"
    r"search engineer|recommendation systems engineer|machine learning engineer|"
    r"applied ml engineer|ai engineer|ml engineer|nlp engineer|data scientist|"
    r"senior data scientist|ai specialist"
    r")\b",
    re.I,
)

ADJACENT_TITLE_RE = re.compile(
    r"\b("
    r"software engineer|senior software engineer|backend engineer|data engineer|"
    r"senior data engineer|analytics engineer|cloud engineer|devops engineer|"
    r"full stack developer"
    r")\b",
    re.I,
)

SENIOR_RE = re.compile(r"\b(senior|staff|lead|principal|founding|architect)\b", re.I)
RESEARCH_RE = re.compile(r"\b(research|academic|lab|paper|publication)\b", re.I)

RETRIEVAL_TERMS = [
    "hybrid retrieval",
    "semantic search",
    "vector search",
    "dense vector",
    "dense retrieval",
    "sparse retrieval",
    "information retrieval",
    "retrieval",
    "embedding",
    "embeddings",
    "sentence-transformers",
    "sentence transformers",
    "bm25",
    "faiss",
    "milvus",
    "qdrant",
    "weaviate",
    "pinecone",
    "opensearch",
    "elasticsearch",
    "pgvector",
    "hnsw",
    "ann index",
]

RANKING_TERMS = [
    "learning-to-rank",
    "learning to rank",
    "ltr",
    "ranker",
    "ranking",
    "recommendation system",
    "recommendation systems",
    "recommender",
    "personalization",
    "discovery feed",
    "candidate-jd matching",
    "matching pipeline",
    "behavioral-signal",
    "behavioral signal",
]

EVAL_TERMS = [
    "ndcg",
    "mrr",
    "map@",
    "mean average precision",
    "offline evaluation",
    "online evaluation",
    "a/b",
    "ab test",
    "offline-online",
    "eval framework",
    "evaluation framework",
    "relevance",
    "relevance judgment",
    "relevance labels",
]

PRODUCTION_TERMS = [
    "production",
    "deployed",
    "serving",
    "served",
    "shipped",
    "scale",
    "qps",
    "latency",
    "p95",
    "millions",
    "m+",
    "users",
    "queries",
    "index refresh",
    "incremental refresh",
    "drift",
    "monitoring",
    "regression",
    "on-call",
]

RELEVANCE_SYSTEM_TERMS = [
    "connect users with relevant information",
    "relevant information at scale",
    "ranking and retrieval systems",
    "decide what to show",
    "surface the right thing",
    "right thing at the right time",
    "millions of items",
    "millions of users",
    "search and discovery experience",
    "search & discovery",
    "discovery experience",
    "most relevant results",
    "user's intent",
    "users intent",
    "relevance actually means",
    "matching layer",
    "matching pipeline",
    "hand-tuned heuristic",
    "explicit modeling and evaluation",
    "evaluation methodology",
    "offline metrics",
    "online engagement",
    "online numbers",
    "offline experimentation",
    "online a/b testing",
    "personalization infrastructure",
    "improves relevance over time",
    "feature monitoring",
    "drift detection",
    "retraining cadence",
    "intelligence layer end-to-end",
]

SHIPPER_TERMS = [
    "shipping real systems",
    "shipped",
    "working v1",
    "real users",
    "production load",
    "product judgment",
    "product experience",
    "direct collaboration with product",
    "collaboration with product",
    "product company",
    "senior ic",
    "tech-lead",
    "own the intelligence layer",
]

LLM_TERMS = [
    "llm",
    "llms",
    "rag",
    "large language",
    "fine-tuning",
    "fine tuning",
    "lora",
    "qlora",
    "peft",
    "transformer",
    "hugging face",
    "prompt",
]

WRONG_DOMAIN_TERMS = [
    "computer vision",
    "image classification",
    "object detection",
    "yolo",
    "gan",
    "gans",
    "speech recognition",
    "tts",
    "robotics",
]

CORE_AI_SKILL_RE = re.compile(
    r"\b("
    r"llms?|rag|embeddings?|vector search|semantic search|information retrieval|"
    r"recommendation systems?|learning to rank|bm25|faiss|pinecone|qdrant|"
    r"weaviate|milvus|opensearch|elasticsearch|sentence transformers|"
    r"hugging face transformers|fine-tuning llms|lora|qlora|peft|"
    r"machine learning|deep learning|nlp|mlops|bentoml|mlflow|kubeflow|"
    r"pytorch|tensorflow|scikit-learn|haystack|llamaindex|langchain|pgvector"
    r")\b",
    re.I,
)

SKILL_IMPORTANCE = {
    "information retrieval": 1.00,
    "semantic search": 1.00,
    "vector search": 1.00,
    "learning to rank": 1.00,
    "recommendation systems": 0.95,
    "embeddings": 0.92,
    "sentence transformers": 0.92,
    "faiss": 0.88,
    "bm25": 0.88,
    "opensearch": 0.82,
    "elasticsearch": 0.82,
    "pinecone": 0.80,
    "milvus": 0.80,
    "qdrant": 0.80,
    "weaviate": 0.80,
    "pgvector": 0.78,
    "rag": 0.70,
    "llms": 0.64,
    "fine-tuning llms": 0.58,
    "lora": 0.50,
    "qlora": 0.50,
    "peft": 0.50,
    "hugging face transformers": 0.50,
    "machine learning": 0.46,
    "deep learning": 0.42,
    "nlp": 0.42,
    "mlops": 0.38,
    "pytorch": 0.34,
    "tensorflow": 0.34,
    "scikit-learn": 0.32,
    "bentoml": 0.30,
    "mlflow": 0.30,
    "kubeflow": 0.28,
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.82,
    "expert": 1.00,
}


@dataclass(order=True)
class HeapItem:
    sort_key: tuple[float, str] = field(init=False, repr=False)
    score: float
    candidate_id: str
    row: dict[str, Any] = field(compare=False)

    def __post_init__(self) -> None:
        self.sort_key = (self.score, self.candidate_id)


def norm(text: Any) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").split())


def open_jsonl(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with path.open("r", encoding="utf-8") as f:
            yield from f


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def saturation(value: float, scale: float) -> float:
    if value <= 0:
        return 0.0
    return 1.0 - math.exp(-value / scale)


def term_score(text: str, terms: list[str], cap: float = 1.0) -> tuple[float, list[str]]:
    hits: list[str] = []
    weighted = 0.0
    for term in terms:
        if term in text:
            hits.append(term)
            weighted += 1.0
    return min(cap, weighted / max(1.0, min(5.0, len(terms) / 3))), hits


def weighted_term_score(
    career_text: str,
    profile_text: str,
    skills_text: str,
    terms: list[str],
    cap: float = 1.0,
) -> tuple[float, list[str]]:
    hits: list[str] = []
    total = 0.0
    for term in terms:
        weight = 0.0
        if term in career_text:
            weight += 1.00
        if term in profile_text:
            weight += 0.65
        if term in skills_text:
            weight += 0.25
        if weight:
            total += min(1.2, weight)
            hits.append(term)
    return min(cap, total / 4.0), hits


def years_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0:
        return 0.72 + 0.20 * (years - 4.0)
    if 9.0 < years <= 10.5:
        return 0.82 - 0.18 * ((years - 9.0) / 1.5)
    if 3.0 <= years < 4.0:
        return 0.46 + 0.20 * (years - 3.0)
    if 10.5 < years <= 12.0:
        return 0.52
    return 0.18


def title_score(title: str, headline: str, career_titles: str) -> tuple[float, list[str]]:
    combined_current = f"{norm(title)} {norm(headline)}"
    combined_all = f"{combined_current} {norm(career_titles)}"
    notes: list[str] = []

    exact_title = norm(title)
    exact_scores = {
        "senior ai engineer": 1.00,
        "lead ai engineer": 0.99,
        "staff machine learning engineer": 0.98,
        "senior machine learning engineer": 0.97,
        "senior applied scientist": 0.95,
        "senior nlp engineer": 0.94,
        "search engineer": 0.93,
        "recommendation systems engineer": 0.93,
        "machine learning engineer": 0.90,
        "applied ml engineer": 0.90,
        "ai engineer": 0.89,
        "senior data scientist": 0.87,
        "nlp engineer": 0.84,
        "ml engineer": 0.83,
        "data scientist": 0.80,
        "ai specialist": 0.68,
        "ai research engineer": 0.60,
        "computer vision engineer": 0.42,
        "junior ml engineer": 0.34,
    }
    if exact_title in exact_scores:
        notes.append(f"title:{title}")
        return exact_scores[exact_title], notes

    if TARGET_TITLE_RE.search(combined_current):
        notes.append(f"title:{title}")
        return 0.82 + (0.08 if SENIOR_RE.search(combined_current) else 0.0), notes

    if ADJACENT_TITLE_RE.search(combined_current):
        base = 0.50
        if "ml" in combined_all or "machine learning" in combined_all or "ai" in combined_all:
            base += 0.12
        if "search" in combined_all or "ranking" in combined_all or "recommendation" in combined_all:
            base += 0.15
        notes.append(f"adjacent title:{title}")
        return min(0.78, base), notes

    if NON_TARGET_TITLE_RE.search(combined_current):
        notes.append(f"non-target title:{title}")
        return 0.05, notes

    return 0.22, notes


def company_score(profile: dict[str, Any], career_history: list[dict[str, Any]]) -> tuple[float, float, list[str]]:
    current_company = norm(profile.get("current_company"))
    current_industry = norm(profile.get("current_industry"))
    all_companies = {current_company}
    all_industries = {current_industry}
    for job in career_history:
        all_companies.add(norm(job.get("company")))
        all_industries.add(norm(job.get("industry")))

    product_hit = any(c in PRODUCT_COMPANIES for c in all_companies)
    service_current = current_company in SERVICE_COMPANIES
    service_count = sum(1 for c in all_companies if c in SERVICE_COMPANIES)
    product_industry_hit = any(
        token in " ".join(all_industries)
        for token in ["software", "fintech", "e-commerce", "internet", "saas", "ai/ml", "healthtech ai"]
    )

    score = 0.48
    notes: list[str] = []
    if product_hit:
        score += 0.36
        notes.append("product company")
    if product_industry_hit:
        score += 0.14
        notes.append("product/AI industry")
    if service_current:
        score -= 0.18
        notes.append("current services company")
    if service_count >= max(1, len(all_companies) - 1) and not product_hit:
        score -= 0.24
        notes.append("mostly services background")

    service_penalty = 0.0
    if service_current:
        service_penalty += 0.08
    if service_count >= max(1, len(all_companies) - 1) and not product_hit:
        service_penalty += 0.14

    return clamp(score), service_penalty, notes


def location_score(profile: dict[str, Any], signals: dict[str, Any]) -> tuple[float, list[str]]:
    country = norm(profile.get("country"))
    location = norm(profile.get("location"))
    city = location.split(",")[0].strip()
    willing = bool(signals.get("willing_to_relocate"))
    mode = norm(signals.get("preferred_work_mode"))

    notes: list[str] = []
    if city in TARGET_LOCATIONS:
        notes.append(profile.get("location", "target location"))
        base = TARGET_LOCATIONS[city]
    elif country == "india":
        base = 0.68
        notes.append("India-based")
    elif willing:
        base = 0.46
        notes.append("non-India but willing to relocate")
    else:
        base = 0.24
        notes.append("location weak")

    if willing and base < 0.94:
        base += 0.10
    if mode in {"hybrid", "flexible", "onsite"}:
        base += 0.05
    return clamp(base), notes


def behavior_score(signals: dict[str, Any]) -> tuple[float, list[str]]:
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = max(0, (REFERENCE_DATE - last_active).days)
        recency = clamp(1.0 - days_inactive / 120.0)
    else:
        days_inactive = 999
        recency = 0.0

    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.25
    response_rate = clamp(float(signals.get("recruiter_response_rate") or 0.0))
    avg_response_hours = float(signals.get("avg_response_time_hours") or 999.0)
    response_speed = clamp(1.0 - avg_response_hours / 240.0)
    notice_days = float(signals.get("notice_period_days") or 180.0)
    notice = clamp(1.0 - max(0.0, notice_days - 15.0) / 105.0)
    profile_complete = clamp(float(signals.get("profile_completeness_score") or 0.0) / 100.0)
    github_raw = float(signals.get("github_activity_score") or -1.0)
    github = 0.0 if github_raw < 0 else clamp(github_raw / 100.0)
    saved = saturation(float(signals.get("saved_by_recruiters_30d") or 0.0), 8.0)
    views = saturation(float(signals.get("profile_views_received_30d") or 0.0), 45.0)
    interview = clamp(float(signals.get("interview_completion_rate") or 0.0))
    offer_raw = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)
    offer = 0.5 if offer_raw < 0 else clamp(offer_raw)
    verified = (
        0.34 * bool(signals.get("verified_email"))
        + 0.33 * bool(signals.get("verified_phone"))
        + 0.33 * bool(signals.get("linkedin_connected"))
    )

    score = (
        0.18 * recency
        + 0.15 * open_to_work
        + 0.17 * response_rate
        + 0.08 * response_speed
        + 0.12 * notice
        + 0.08 * profile_complete
        + 0.07 * github
        + 0.05 * saved
        + 0.03 * views
        + 0.04 * interview
        + 0.02 * offer
        + 0.01 * verified
    )

    notes: list[str] = []
    if days_inactive <= 30:
        notes.append(f"active {days_inactive}d ago")
    if response_rate >= 0.7:
        notes.append(f"{response_rate:.0%} response rate")
    if notice_days <= 30:
        notes.append(f"{int(notice_days)}d notice")
    return clamp(score), notes


def skill_score(skills: list[dict[str, Any]], signals: dict[str, Any]) -> tuple[float, int, list[str]]:
    assessments = {norm(k): float(v) for k, v in (signals.get("skill_assessment_scores") or {}).items()}
    total = 0.0
    trusted_hits: list[tuple[float, str]] = []
    ai_skill_count = 0

    for skill in skills:
        name = norm(skill.get("name"))
        if not name:
            continue
        if CORE_AI_SKILL_RE.search(name):
            ai_skill_count += 1
        importance = SKILL_IMPORTANCE.get(name)
        if importance is None:
            continue
        prof = PROFICIENCY_WEIGHT.get(norm(skill.get("proficiency")), 0.4)
        duration = saturation(float(skill.get("duration_months") or 0), 30.0)
        endorsements = saturation(float(skill.get("endorsements") or 0), 25.0)
        assessment = assessments.get(name)
        assessment_factor = 0.60 + 0.40 * clamp(assessment / 100.0) if assessment is not None else 0.82
        trust = 0.50 * prof + 0.25 * duration + 0.15 * endorsements + 0.10 * assessment_factor
        contribution = importance * trust
        total += contribution
        trusted_hits.append((contribution, skill.get("name", name)))

    trusted_hits.sort(reverse=True)
    notes = [name for _, name in trusted_hits[:5]]
    return min(1.0, total / 4.5), ai_skill_count, notes


def education_score(education: list[dict[str, Any]]) -> tuple[float, list[str]]:
    if not education:
        return 0.35, []
    best = 0.35
    notes: list[str] = []
    for edu in education:
        tier = norm(edu.get("tier"))
        field = norm(edu.get("field_of_study"))
        degree = norm(edu.get("degree"))
        score = {"tier 1": 1.0, "tier 2": 0.76, "tier 3": 0.52, "tier 4": 0.35}.get(tier, 0.40)
        if any(x in field for x in ["computer", "data", "ai", "machine learning", "mathematics", "statistics"]):
            score += 0.12
        if any(x in degree for x in ["m.tech", "m.s", "master", "ph.d", "phd"]):
            score += 0.06
        if score > best:
            best = score
            notes = [f"{edu.get('degree', '').strip()} {edu.get('field_of_study', '').strip()}".strip()]
    return clamp(best), notes


def fake_risk(
    candidate: dict[str, Any],
    career_text: str,
    profile_text: str,
    skills_text: str,
    feature_scores: dict[str, float],
    ai_skill_count: int,
) -> tuple[float, list[str]]:
    profile = candidate["profile"]
    skills = candidate.get("skills") or []
    career_history = candidate.get("career_history") or []
    title = norm(profile.get("current_title"))
    combined_title = f"{title} {norm(profile.get('headline'))}"

    risk = 0.0
    reasons: list[str] = []

    non_target_title = bool(NON_TARGET_TITLE_RE.search(combined_title))
    target_title = bool(TARGET_TITLE_RE.search(combined_title))
    adjacent_title = bool(ADJACENT_TITLE_RE.search(combined_title))
    career_evidence = (
        feature_scores["retrieval"]
        + feature_scores["ranking"]
        + feature_scores["evaluation"]
        + feature_scores["production"]
        + feature_scores["relevance_system"]
    )

    if non_target_title and ai_skill_count >= 6:
        risk += 0.38 + 0.025 * min(8, ai_skill_count - 6)
        reasons.append("AI keyword stuffing on non-AI title")
    if ai_skill_count >= 8 and career_evidence < 0.85:
        risk += 0.22
        reasons.append("AI skills not backed by career history")
    if ai_skill_count >= 10 and not (target_title or adjacent_title):
        risk += 0.24
        reasons.append("many AI skills with unrelated role")

    impossible_skill_count = 0
    low_duration_advanced = 0
    for skill in skills:
        prof = norm(skill.get("proficiency"))
        duration = int(skill.get("duration_months") or 0)
        if prof == "expert" and duration == 0:
            impossible_skill_count += 1
        if prof in {"advanced", "expert"} and duration <= 3:
            low_duration_advanced += 1
    if impossible_skill_count:
        risk += min(0.30, 0.10 * impossible_skill_count)
        reasons.append(f"{impossible_skill_count} expert skills with 0 months")
    if low_duration_advanced >= 4:
        risk += min(0.24, 0.05 * low_duration_advanced)
        reasons.append("many advanced/expert skills with tiny duration")

    years = float(profile.get("years_of_experience") or 0.0)
    total_months = sum(int(job.get("duration_months") or 0) for job in career_history)
    expected_months = years * 12.0
    if total_months > expected_months + 30:
        risk += min(0.28, (total_months - expected_months) / 220.0)
        reasons.append("career duration exceeds stated experience")
    if expected_months > 48 and total_months < expected_months - 42:
        risk += min(0.24, (expected_months - total_months) / 260.0)
        reasons.append("career duration far below stated experience")

    for job in career_history:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start and end and start > end:
            risk += 0.24
            reasons.append("job start date after end date")
        if job.get("is_current") and end is not None:
            risk += 0.12
            reasons.append("current job has end date")

    retrieval_or_ranking = (
        feature_scores["retrieval"]
        + feature_scores["ranking"]
        + feature_scores["evaluation"]
        + feature_scores["relevance_system"]
    )
    wrong_domain, wrong_hits = term_score(f"{career_text} {profile_text} {skills_text}", WRONG_DOMAIN_TERMS)
    if wrong_domain > 0.45 and retrieval_or_ranking < 0.65:
        risk += 0.18
        reasons.append("AI domain mismatch")

    research = bool(RESEARCH_RE.search(profile_text + " " + career_text))
    if research and feature_scores["production"] < 0.35 and retrieval_or_ranking < 0.45:
        risk += 0.16
        reasons.append("research signal without production proof")

    signals = candidate["redrob_signals"]
    completeness = float(signals.get("profile_completeness_score") or 0.0)
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    last_active = parse_date(signals.get("last_active_date"))
    inactive_days = (REFERENCE_DATE - last_active).days if last_active else 999
    verification_count = sum(
        bool(signals.get(key)) for key in ["verified_email", "verified_phone", "linkedin_connected"]
    )
    if completeness < 55:
        risk += 0.07
        reasons.append("low profile completeness")
    if verification_count == 0:
        risk += 0.05
        reasons.append("no verified contact/social links")
    if inactive_days > 150 and response_rate < 0.20:
        risk += 0.08
        reasons.append("stale and low response")

    return clamp(risk, 0.0, 1.0), reasons


def make_text_blocks(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    profile = candidate["profile"]
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    profile_text = norm(
        " ".join(
            [
                profile.get("headline", ""),
                profile.get("summary", ""),
                profile.get("current_title", ""),
                profile.get("current_company", ""),
                profile.get("current_industry", ""),
            ]
        )
    )
    career_text = norm(
        " ".join(
            [
                " ".join(
                    [
                        job.get("title", ""),
                        job.get("company", ""),
                        job.get("industry", ""),
                        job.get("description", ""),
                    ]
                )
                for job in career
            ]
        )
    )
    career_titles = norm(" ".join(job.get("title", "") for job in career))
    skills_text = norm(" ".join(skill.get("name", "") for skill in skills))
    return profile_text, career_text, career_titles, skills_text


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    career_history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    education = candidate.get("education") or []

    profile_text, career_text, career_titles, skills_text = make_text_blocks(candidate)

    t_score, title_notes = title_score(profile.get("current_title", ""), profile.get("headline", ""), career_titles)
    y_score = years_score(float(profile.get("years_of_experience") or 0.0))
    retrieval, retrieval_hits = weighted_term_score(career_text, profile_text, skills_text, RETRIEVAL_TERMS)
    ranking, ranking_hits = weighted_term_score(career_text, profile_text, skills_text, RANKING_TERMS)
    evaluation, eval_hits = weighted_term_score(career_text, profile_text, skills_text, EVAL_TERMS)
    production, production_hits = weighted_term_score(career_text, profile_text, skills_text, PRODUCTION_TERMS)
    relevance_system, relevance_hits = weighted_term_score(career_text, profile_text, "", RELEVANCE_SYSTEM_TERMS)
    shipper, shipper_hits = weighted_term_score(career_text, profile_text, "", SHIPPER_TERMS)
    llm, llm_hits = weighted_term_score(career_text, profile_text, skills_text, LLM_TERMS)
    s_score, ai_skill_count, trusted_skills = skill_score(skills, signals)
    company, service_penalty, company_notes = company_score(profile, career_history)
    loc, location_notes = location_score(profile, signals)
    behavior, behavior_notes = behavior_score(signals)
    education_value, education_notes = education_score(education)

    retrieval_ranking_evidence = (
        0.25 * retrieval
        + 0.22 * ranking
        + 0.17 * evaluation
        + 0.30 * relevance_system
        + 0.06 * llm
    )
    production_system_evidence = 0.58 * production + 0.24 * retrieval_ranking_evidence + 0.18 * shipper

    feature_scores = {
        "title": t_score,
        "years": y_score,
        "retrieval": retrieval,
        "ranking": ranking,
        "evaluation": evaluation,
        "production": production,
        "relevance_system": relevance_system,
        "shipper": shipper,
        "llm": llm,
        "skills": s_score,
        "company": company,
        "location": loc,
        "behavior": behavior,
        "education": education_value,
    }

    risk, risk_reasons = fake_risk(
        candidate,
        career_text,
        profile_text,
        skills_text,
        feature_scores,
        ai_skill_count,
    )

    # Gate-like multipliers keep keyword-only candidates out of the top ranks.
    evidence_gate = clamp(
        0.42
        + 0.28 * t_score
        + 0.24 * retrieval_ranking_evidence
        + 0.15 * production
        + 0.22 * relevance_system
    )
    if NON_TARGET_TITLE_RE.search(norm(profile.get("current_title"))):
        evidence_gate *= 0.65
    if retrieval_ranking_evidence < 0.28 and s_score > 0.72:
        evidence_gate *= 0.72

    base_fit = (
        0.205 * t_score
        + 0.110 * y_score
        + 0.245 * retrieval_ranking_evidence
        + 0.150 * production_system_evidence
        + 0.040 * s_score
        + 0.070 * relevance_system
        + 0.035 * shipper
        + 0.075 * company
        + 0.055 * loc
        + 0.065 * behavior
        + 0.025 * education_value
    )

    senior_bonus = 0.0
    if SENIOR_RE.search(norm(profile.get("current_title")) + " " + norm(profile.get("headline"))):
        senior_bonus += 0.018
    if retrieval >= 0.85 and ranking >= 0.65 and production >= 0.70:
        senior_bonus += 0.040
    if relevance_system >= 0.70 and production >= 0.65 and evaluation >= 0.40:
        senior_bonus += 0.060
    if evaluation >= 0.45:
        senior_bonus += 0.022
    if 6.0 <= float(profile.get("years_of_experience") or 0.0) <= 8.5:
        senior_bonus += 0.015

    logistics_penalty = 0.0
    notice_days = float(signals.get("notice_period_days") or 180.0)
    if notice_days > 90:
        logistics_penalty += 0.04
    if not signals.get("open_to_work_flag") and behavior < 0.45:
        logistics_penalty += 0.035
    years = float(profile.get("years_of_experience") or 0.0)
    if years < 4.0:
        logistics_penalty += 0.16
    elif years < 4.5:
        logistics_penalty += 0.07
    elif years < 5.0:
        logistics_penalty += 0.025
    if years > 12.0:
        logistics_penalty += 0.12
    elif years > 10.5:
        logistics_penalty += 0.045

    score = (base_fit + senior_bonus) * evidence_gate
    score -= 0.46 * risk + 0.10 * service_penalty + logistics_penalty
    score = max(-1.0, score)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": score,
        "fit_score": base_fit,
        "fake_risk": risk,
        "risk_reasons": risk_reasons,
        "features": feature_scores,
        "hits": {
            "title": title_notes,
            "retrieval": retrieval_hits,
            "ranking": ranking_hits,
            "evaluation": eval_hits,
            "production": production_hits,
            "relevance_system": relevance_hits,
            "shipper": shipper_hits,
            "llm": llm_hits,
            "skills": trusted_skills,
            "company": company_notes,
            "location": location_notes,
            "behavior": behavior_notes,
            "education": education_notes,
        },
        "candidate": candidate,
    }


def choose_evidence(hit_groups: dict[str, list[str]], limit: int = 4) -> list[str]:
    priority = ["relevance_system", "retrieval", "ranking", "evaluation", "production", "shipper", "skills"]
    chosen: list[str] = []
    seen: set[str] = set()
    for group in priority:
        for hit in hit_groups.get(group, []):
            clean = hit.replace("@", "").strip()
            key = norm(clean)
            if key and key not in seen:
                chosen.append(clean)
                seen.add(key)
            if len(chosen) >= limit:
                return chosen
    return chosen


def humanize_evidence(evidence: list[str], limit: int = 4) -> list[str]:
    mapping = {
        "connect users with relevant information": "large-scale relevance systems",
        "relevant information at scale": "large-scale relevance systems",
        "ranking and retrieval systems": "ranking/retrieval systems",
        "decide what to show": "ranking decisions",
        "surface the right thing": "ranking decisions",
        "right thing at the right time": "ranking decisions",
        "millions of items": "large-scale product ranking",
        "millions of users": "large-scale product ranking",
        "search and discovery experience": "search/discovery",
        "search & discovery": "search/discovery",
        "discovery experience": "search/discovery",
        "most relevant results": "search relevance",
        "user's intent": "intent-aware retrieval",
        "users intent": "intent-aware retrieval",
        "relevance actually means": "relevance evaluation",
        "matching layer": "matching-layer redesign",
        "matching pipeline": "matching pipeline",
        "hand-tuned heuristic": "heuristic-to-modeled ranking migration",
        "explicit modeling and evaluation": "modeling plus evaluation",
        "evaluation methodology": "evaluation methodology",
        "offline metrics": "offline ranking metrics",
        "online engagement": "offline-to-online evaluation",
        "online numbers": "offline-to-online evaluation",
        "offline experimentation": "offline experimentation",
        "online a/b testing": "online A/B testing",
        "personalization infrastructure": "personalization infrastructure",
        "improves relevance over time": "behavior-driven relevance",
        "feature monitoring": "feature monitoring",
        "drift detection": "drift detection",
        "retraining cadence": "model retraining cadence",
        "intelligence layer end-to-end": "end-to-end intelligence layer ownership",
        "hybrid retrieval": "hybrid retrieval",
        "semantic search": "semantic search",
        "vector search": "vector search",
        "information retrieval": "information retrieval",
        "learning-to-rank": "learning-to-rank",
        "learning to rank": "learning-to-rank",
        "recommendation system": "recommendation systems",
        "recommendation systems": "recommendation systems",
        "a/b": "A/B testing",
        "ab test": "A/B testing",
        "eval framework": "evaluation framework",
        "evaluation framework": "evaluation framework",
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        clean = mapping.get(norm(item), item.strip())
        key = norm(clean)
        if key and key not in seen:
            out.append(clean)
            seen.add(key)
        if len(out) >= limit:
            break
    return out


def make_reasoning(row: dict[str, Any], rank: int) -> str:
    candidate = row["candidate"]
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    features = row["features"]
    hits = row["hits"]

    title = profile.get("current_title", "Candidate")
    years = float(profile.get("years_of_experience") or 0.0)
    company = profile.get("current_company", "current company")
    location = profile.get("location", "")
    evidence = humanize_evidence(choose_evidence(hits, limit=8), limit=4)

    first_bits = [f"{years:.1f} yrs {title} at {company}"]
    if evidence:
        first_bits.append("career evidence in " + ", ".join(evidence[:3]))
    else:
        first_bits.append("some adjacent ML/product evidence")

    first = "; ".join(first_bits) + "."

    concern_parts: list[str] = []
    notice_days = int(signals.get("notice_period_days") or 0)
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    if location:
        concern_parts.append(location)
    if notice_days <= 30:
        concern_parts.append(f"{notice_days}d notice")
    elif notice_days >= 60 and rank > 20:
        concern_parts.append(f"{notice_days}d notice is a concern")
    if response_rate >= 0.70:
        concern_parts.append(f"{response_rate:.0%} response rate")
    elif response_rate < 0.25 and rank > 50:
        concern_parts.append(f"low {response_rate:.0%} response rate")
    if row["fake_risk"] >= 0.15:
        concern_parts.append("minor profile-risk penalty applied")
    if features["evaluation"] >= 0.40:
        concern_parts.append("ranking-eval signal")

    second = "Logistics/signals: " + ", ".join(concern_parts[:4]) + "." if concern_parts else ""
    return " ".join(x for x in [first, second] if x)


def iter_ranked(candidates_path: Path, keep: int) -> list[dict[str, Any]]:
    heap: list[HeapItem] = []
    seen = 0
    for line in open_jsonl(candidates_path):
        if not line.strip():
            continue
        seen += 1
        candidate = json.loads(line)
        row = score_candidate(candidate)
        item = HeapItem(score=row["score"], candidate_id=row["candidate_id"], row=row)
        if len(heap) < keep:
            heapq.heappush(heap, item)
        elif item.sort_key > heap[0].sort_key:
            heapq.heapreplace(heap, item)

    ranked = [item.row for item in heap]
    ranked.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return ranked


def write_submission(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top100 = rows[:100]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(top100, start=1):
            writer.writerow(
                [
                    row["candidate_id"],
                    rank,
                    f"{row['score']:.6f}",
                    make_reasoning(row, rank),
                ]
            )


def write_audit(rows: list[dict[str, Any]], audit_path: Path, limit: int) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "candidate_id",
                "score",
                "fake_risk",
                "current_title",
                "years",
                "company",
                "location",
                "retrieval",
                "ranking",
                "evaluation",
                "production",
                "relevance_system",
                "shipper",
                "skills",
                "behavior",
                "risk_reasons",
                "evidence_hits",
            ]
        )
        for rank, row in enumerate(rows[:limit], start=1):
            c = row["candidate"]
            p = c["profile"]
            h = row["hits"]
            writer.writerow(
                [
                    rank,
                    row["candidate_id"],
                    f"{row['score']:.6f}",
                    f"{row['fake_risk']:.3f}",
                    p.get("current_title", ""),
                    p.get("years_of_experience", ""),
                    p.get("current_company", ""),
                    p.get("location", ""),
                    f"{row['features']['retrieval']:.3f}",
                    f"{row['features']['ranking']:.3f}",
                    f"{row['features']['evaluation']:.3f}",
                    f"{row['features']['production']:.3f}",
                    f"{row['features']['relevance_system']:.3f}",
                    f"{row['features']['shipper']:.3f}",
                    f"{row['features']['skills']:.3f}",
                    f"{row['features']['behavior']:.3f}",
                    "; ".join(row["risk_reasons"]),
                    " | ".join(choose_evidence(h, limit=8)),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, type=Path, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--audit", type=Path, default=None, help="Optional audit CSV path for top candidates")
    parser.add_argument("--audit-limit", type=int, default=250, help="Rows to include in audit CSV")
    parser.add_argument("--keep", type=int, default=750, help="Internal heap size before final top-100")
    args = parser.parse_args()

    keep = max(100, args.keep)
    rows = iter_ranked(args.candidates, keep=keep)
    write_submission(rows, args.out)
    if args.audit:
        write_audit(rows, args.audit, limit=args.audit_limit)
    print(f"Wrote {args.out} with top 100 candidates.")
    if args.audit:
        print(f"Wrote audit file {args.audit}.")


if __name__ == "__main__":
    main()
