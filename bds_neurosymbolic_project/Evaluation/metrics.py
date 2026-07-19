"""Metric đánh giá recommendation Top-K."""
from __future__ import annotations

import math
import pandas as pd


def precision_at_k(relevances: list[int | float], k: int = 5) -> float:
    vals = relevances[:k]
    if not vals:
        return 0.0
    return sum(1 for x in vals if x > 0) / len(vals)


def hit_rate_at_k(relevances: list[int | float], k: int = 5) -> float:
    return 1.0 if any(x > 0 for x in relevances[:k]) else 0.0


def mrr_at_k(relevances: list[int | float], k: int = 5) -> float:
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevances: list[int | float], k: int = 5) -> float:
    vals = relevances[:k]
    dcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(vals))
    ideal = sorted(vals, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_metrics(rows: list[dict], k: int = 5) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.groupby("method")[[f"precision@{k}", f"hit_rate@{k}", f"mrr@{k}", f"ndcg@{k}", "avg_relevance"]].mean().reset_index()
