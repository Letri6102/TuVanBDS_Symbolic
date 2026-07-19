"""Baseline keyword, hard-filter, TF-IDF và đánh giá so với proposed."""
from __future__ import annotations

import re
import math
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from Data_Preprocessing.text_utils import norm_text
from Recommendation.query_parser import parse_user_query
from Recommendation.recommender import recommend
from Recommendation.scoring import hard_constraint_pass
from Recommendation.semantic import fit_semantic_index
from Evaluation.metrics import precision_at_k, hit_rate_at_k, mrr_at_k, ndcg_at_k


def property_corpus_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""), row.get("description", ""), row.get("raw_location", ""),
        row.get("district", ""), row.get("property_type", ""), row.get("legal_class", ""),
        " ".join(row.get("amenity_tags", []) if isinstance(row.get("amenity_tags", []), list) else []),
        " ".join(row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []),
    ]
    return norm_text(" ".join(map(str, parts)))


def keyword_baseline(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    q_tokens = set(re.findall(r"[a-z0-9]+", norm_text(query)))
    rows = []
    for _, row in df.iterrows():
        text = property_corpus_text(row)
        score = sum(1 for t in q_tokens if t in text) / max(len(q_tokens), 1)
        rec = row.to_dict()
        rec["final_score"] = score
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def hard_filter_baseline(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    profile = parse_user_query(query)
    rows = []
    for _, row in df.iterrows():
        if hard_constraint_pass(row, profile, relax_level=0):
            rec = row.to_dict()
            rec["final_score"] = 1.0
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).head(top_k).reset_index(drop=True)


def tfidf_baseline(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    corpus = [property_corpus_text(r) for _, r in df.iterrows()]
    docs = [re.findall(r"[a-z0-9]+", text) for text in corpus]
    query_tokens = re.findall(r"[a-z0-9]+", norm_text(query))
    doc_freq: Counter[str] = Counter()
    for tokens in docs + [query_tokens]:
        doc_freq.update(set(tokens))
    n_docs = len(docs) + 1
    idf = {token: math.log((1 + n_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}

    def vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(sum(counts.values()), 1)
        return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}

    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    query_vec = vector(query_tokens)
    sims = np.array([cosine(query_vec, vector(tokens)) for tokens in docs])
    out = df.copy()
    out["final_score"] = sims
    return out.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def proposed_method(query: str, df: pd.DataFrame, top_k: int = 5, semantic_index: dict | None = None) -> pd.DataFrame:
    _, recs = recommend(query, df, top_k=top_k, relax=True, semantic_index=semantic_index)
    return recs


def heuristic_relevance_grade(row: pd.Series, expected_facts: list[str], avoid_risks: list[str]) -> int:
    facts = row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []
    risks = row.get("risk_flags", []) if isinstance(row.get("risk_flags", []), list) else []
    matched = len(set(facts) & set(expected_facts))
    bad = len(set(risks) & set(avoid_risks))
    grade = matched - bad
    return max(0, min(3, grade))


def evaluate_methods(test_cases: list[dict], df: pd.DataFrame, top_k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    semantic_index = fit_semantic_index(df)

    def proposed_with_index(query: str, data: pd.DataFrame, k: int) -> pd.DataFrame:
        return proposed_method(query, data, k, semantic_index=semantic_index)

    methods: dict[str, Callable[[str, pd.DataFrame, int], pd.DataFrame]] = {
        "keyword": keyword_baseline,
        "hard_filter": hard_filter_baseline,
        "tfidf": tfidf_baseline,
        "proposed_neuro_symbolic": proposed_with_index,
    }
    detail_rows = []
    metric_rows = []
    for case in test_cases:
        for name, func in methods.items():
            recs = func(case["query"], df, top_k)
            relevances = []
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                rel = heuristic_relevance_grade(row, case.get("expected_facts", []), case.get("avoid_risks", []))
                relevances.append(rel)
                detail_rows.append({
                    "case_id": case["case_id"], "method": name, "rank": rank,
                    "property_id": row.get("property_id"), "final_score": row.get("final_score"),
                    "relevance": rel, "query": case["query"],
                    "title": row.get("title"),
                    "district": row.get("district"),
                    "price_billion": row.get("price_billion"),
                    "area_m2": row.get("area_m2"),
                    "legal_class": row.get("legal_class"),
                    "legal_score": row.get("legal_score"),
                    "amenity_tags": row.get("amenity_tags"),
                    "symbolic_score": row.get("symbolic_score"),
                    "semantic_score": row.get("semantic_score"),
                    "semantic_weight": row.get("semantic_weight"),
                    "symbolic_facts": row.get("symbolic_facts"),
                    "risk_flags": row.get("risk_flags"),
                    "match_type": row.get("match_type"),
                    "activation_score": row.get("activation_score"),
                })
            metric_rows.append({
                "case_id": case["case_id"], "method": name,
                f"precision@{top_k}": precision_at_k(relevances, top_k),
                f"hit_rate@{top_k}": hit_rate_at_k(relevances, top_k),
                f"mrr@{top_k}": mrr_at_k(relevances, top_k),
                f"ndcg@{top_k}": ndcg_at_k(relevances, top_k),
                "avg_relevance": float(np.mean(relevances)) if relevances else 0.0,
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(detail_rows)
