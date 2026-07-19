"""Module đề xuất Top-K BĐS."""
from __future__ import annotations

import pandas as pd

from Recommendation.query_parser import parse_user_query
from Recommendation.scoring import hard_constraint_pass, score_property
from Recommendation.explanation import build_explanation
from Recommendation.semantic import semantic_similarity_scores
from Symbolic_Knowledge.location_similarity import district_similarity


def _match_type(row: pd.Series, profile: dict, relax_level: int) -> str:
    desired = profile.get("district")
    if desired is None:
        return "profile_match" if relax_level == 0 else "relaxed_match"
    sim = district_similarity(row.get("district"), desired)
    if sim >= 0.99:
        return "direct_location_match"
    if sim >= float(profile.get("min_location_similarity", 0.0) or 0.0):
        return "similar_location_match"
    return "expanded_location_match"


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _semantic_weight(profile: dict) -> float:
    if profile.get("query_intent") == "advice":
        return 0.22
    if profile.get("desired_amenities") or profile.get("semantic_notes"):
        return 0.20
    return 0.15


def recommend(
    query: str,
    df: pd.DataFrame,
    top_k: int = 5,
    relax: bool = True,
    use_semantic: bool = True,
    semantic_index: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    profile = parse_user_query(query)
    selected = []
    used_relax = 0
    max_relax = 2 if relax else 0
    semantic_scores = _normalize_scores(semantic_similarity_scores(query, df, index=semantic_index)) if use_semantic else [0.0] * len(df)
    semantic_weight = _semantic_weight(profile) if use_semantic else 0.0
    semantic_backend = semantic_index.get("backend", "auto") if isinstance(semantic_index, dict) else "auto"
    semantic_method = semantic_index.get("method", "semantic_retrieval") if isinstance(semantic_index, dict) else "semantic_retrieval"

    for relax_level in range(max_relax + 1):
        rows = []
        for pos, (_, row) in enumerate(df.iterrows()):
            if not hard_constraint_pass(row, profile, relax_level=relax_level):
                continue
            symbolic_score, detail = score_property(row, profile)
            semantic_score = semantic_scores[pos] if pos < len(semantic_scores) else 0.0
            final = (1 - semantic_weight) * symbolic_score + semantic_weight * semantic_score
            record = row.to_dict()
            record["final_score"] = final
            record["symbolic_score"] = symbolic_score
            record["semantic_score"] = semantic_score
            record["semantic_weight"] = semantic_weight
            record["semantic_backend"] = semantic_backend
            record["semantic_method"] = semantic_method
            for k, v in detail.items():
                record[f"{k}_match_score"] = v
            record["activation_score"] = detail.get("location", 0.0)
            record["match_type"] = _match_type(row, profile, relax_level)
            rows.append(record)
        if len(rows) >= top_k or relax_level == max_relax:
            selected = rows
            used_relax = relax_level
            break

    result = pd.DataFrame(selected)
    if result.empty:
        return profile, result
    result = result.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
    explanations = []
    for _, rec in result.iterrows():
        detail = {k.replace("_match_score", ""): rec[k] for k in result.columns if k.endswith("_match_score")}
        explanations.append(build_explanation(rec, profile, detail))
    result["explanation"] = explanations
    result["relax_level_used"] = used_relax
    return profile, result
