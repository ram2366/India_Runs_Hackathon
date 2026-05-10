#!/usr/bin/env python3
"""
Train a small offline V2 reranker from manual labels.

No sklearn/xgboost/lightgbm dependency is required. The model is a standardized
linear reranker trained with pairwise logistic loss. That is simple, fast, and
directly aligned with ranking quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import v2_common as v2


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid_safe_neg_grad(z: float) -> float:
    # derivative coefficient for log(1 + exp(-z)) w.r.t. z is -1/(1+exp(z)).
    if z > 50:
        return 0.0
    if z < -50:
        return -1.0
    return -1.0 / (1.0 + math.exp(z))


def standardize_fit(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    cols = list(zip(*xs))
    means = [statistics.mean(col) for col in cols]
    scales = []
    for col in cols:
        sd = statistics.pstdev(col)
        scales.append(sd if sd > 1e-9 else 1.0)
    return means, scales


def standardize_apply(xs: list[list[float]], means: list[float], scales: list[float]) -> list[list[float]]:
    return [[(x - m) / s for x, m, s in zip(row, means, scales)] for row in xs]


def make_pairs(y: list[float], max_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y[i] - y[j]) >= 1.0:
                if y[i] > y[j]:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def train_pairwise(
    x_train: list[list[float]],
    y_train: list[float],
    *,
    seed: int,
    epochs: int,
    max_pairs: int,
    lr: float,
    l2: float,
) -> list[float]:
    rng = random.Random(seed)
    dim = len(x_train[0])
    weights = [0.0] * dim

    for epoch in range(epochs):
        pairs = make_pairs(y_train, max_pairs=max_pairs, rng=rng)
        if not pairs:
            break
        epoch_lr = lr / math.sqrt(1.0 + epoch * 0.08)
        for hi, lo in pairs:
            dx = [a - b for a, b in zip(x_train[hi], x_train[lo])]
            z = dot(weights, dx)
            coeff = sigmoid_safe_neg_grad(z)
            for k in range(dim):
                grad = coeff * dx[k] + l2 * weights[k]
                weights[k] -= epoch_lr * grad
    return weights


def ndcg_at(labels: list[float], scores: list[float], k: int) -> float:
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))[:k]
    ideal = sorted(labels, reverse=True)[:k]

    def dcg(vals: list[float]) -> float:
        total = 0.0
        for idx, rel in enumerate(vals, start=1):
            total += ((2.0**rel) - 1.0) / math.log2(idx + 1.0)
        return total

    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg([labels[i] for i in order]) / ideal_dcg


def p_at(labels: list[float], scores: list[float], k: int, threshold: float = 3.0) -> float:
    if not labels:
        return 0.0
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))[:k]
    if not order:
        return 0.0
    return sum(1 for i in order if labels[i] >= threshold) / len(order)


def map_score(labels: list[float], scores: list[float], threshold: float = 3.0) -> float:
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))
    total_relevant = sum(1 for y in labels if y >= threshold)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank_idx, i in enumerate(order, start=1):
        if labels[i] >= threshold:
            hits += 1
            precision_sum += hits / rank_idx
    return precision_sum / total_relevant


def pair_accuracy(labels: list[float], scores: list[float]) -> float:
    correct = 0
    total = 0
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            total += 1
            if (labels[i] > labels[j]) == (scores[i] > scores[j]):
                correct += 1
    return correct / total if total else 0.0


def metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at(labels, scores, 10),
        "ndcg@50": ndcg_at(labels, scores, 50),
        "map_label>=3": map_score(labels, scores, 3.0),
        "p@10_label>=3": p_at(labels, scores, 10, 3.0),
        "pair_accuracy": pair_accuracy(labels, scores),
    }


def load_labeled_examples(candidates_path: Path, labels_path: Path) -> list[dict[str, Any]]:
    labels = v2.read_labels(labels_path)
    if not labels:
        return []
    examples: list[dict[str, Any]] = []
    missing = set(labels)
    for scored in v2.stream_scored_candidates(candidates_path):
        cid = scored["candidate_id"]
        if cid not in labels:
            continue
        examples.append(
            {
                "candidate_id": cid,
                "label": labels[cid],
                "v1_score": float(scored["score"]),
                "features": v2.feature_vector(scored),
                "scored": scored,
            }
        )
        missing.discard(cid)
        if not missing:
            break
    if missing:
        print(f"Warning: {len(missing)} labeled candidate_ids were not found in candidates file.", file=sys.stderr)
    return examples


def write_fallback_model(path: Path, reason: str) -> None:
    model = {
        "model_type": "v1_fallback",
        "feature_names": v2.FEATURE_NAMES,
        "reason": reason,
        "intercept": 0.0,
        "weights": [0.0 for _ in v2.FEATURE_NAMES],
        "means": [0.0 for _ in v2.FEATURE_NAMES],
        "scales": [1.0 for _ in v2.FEATURE_NAMES],
        "blend_v1": 1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 pairwise reranker from labels.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--model", default=Path("Project/v2/model.json"), type=Path)
    parser.add_argument("--metrics", default=Path("Project/v2/training_metrics.json"), type=Path)
    parser.add_argument("--predictions", default=Path("Project/v2/labeled_predictions.csv"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--min-labels", type=int, default=80)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    examples = load_labeled_examples(args.candidates, args.labels)
    label_values = sorted({ex["label"] for ex in examples})
    if len(examples) < args.min_labels or len(label_values) < 2:
        reason = (
            f"Need at least {args.min_labels} labels and 2 label values; "
            f"found {len(examples)} labels and values={label_values}."
        )
        if args.allow_fallback:
            write_fallback_model(args.model, reason)
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps({"fallback": True, "reason": reason}, indent=2), encoding="utf-8")
            print(f"Wrote fallback V1 model: {args.model}")
            print(reason)
            return
        raise SystemExit(reason)

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_count = max(20, int(len(examples) * 0.20))
    val = examples[:val_count]
    train = examples[val_count:]

    x_train_raw = [ex["features"] for ex in train]
    y_train = [ex["label"] for ex in train]
    means, scales = standardize_fit(x_train_raw)
    x_train = standardize_apply(x_train_raw, means, scales)
    weights = train_pairwise(
        x_train,
        y_train,
        seed=args.seed,
        epochs=args.epochs,
        max_pairs=args.max_pairs,
        lr=args.lr,
        l2=args.l2,
    )

    def predict(exs: list[dict[str, Any]]) -> list[float]:
        xs = standardize_apply([ex["features"] for ex in exs], means, scales)
        out = []
        for ex, row in zip(exs, xs):
            model_part = dot(weights, row)
            out.append(0.80 * model_part + 0.20 * ex["v1_score"])
        return out

    train_scores = predict(train)
    val_scores = predict(val)
    all_scores = predict(examples)

    metrics_obj = {
        "label_count": len(examples),
        "train_count": len(train),
        "validation_count": len(val),
        "label_values": label_values,
        "train": metrics(y_train, train_scores),
        "validation": metrics([ex["label"] for ex in val], val_scores),
        "all_labeled": metrics([ex["label"] for ex in examples], all_scores),
        "feature_names": v2.FEATURE_NAMES,
    }

    model = {
        "model_type": "pairwise_linear_v2",
        "feature_names": v2.FEATURE_NAMES,
        "means": means,
        "scales": scales,
        "weights": weights,
        "intercept": 0.0,
        "blend_v1": 0.20,
        "label_count": len(examples),
        "label_values": label_values,
        "metrics": metrics_obj,
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2), encoding="utf-8")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")

    with args.predictions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "label", "v1_score", "v2_labeled_score", "split"])
        val_ids = {ex["candidate_id"] for ex in val}
        for ex, score in sorted(zip(examples, all_scores), key=lambda x: -x[1]):
            writer.writerow(
                [
                    ex["candidate_id"],
                    ex["label"],
                    f"{ex['v1_score']:.6f}",
                    f"{score:.6f}",
                    "validation" if ex["candidate_id"] in val_ids else "train",
                ]
            )

    print(f"Wrote model: {args.model}")
    print(f"Wrote metrics: {args.metrics}")
    print(f"Wrote labeled predictions: {args.predictions}")
    print(json.dumps(metrics_obj["validation"], indent=2))


if __name__ == "__main__":
    main()
