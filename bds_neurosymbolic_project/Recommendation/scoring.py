"""Fuzzy scoring cho recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from Symbolic_Knowledge.location_similarity import district_similarity


def fuzzy_budget_score(price: float, budget: float | None) -> float:
    if budget is None or pd.isna(price):
        return 0.6
    if price <= budget:
        diff = (budget - price) / max(budget, 1e-9)
        return max(0.75, 1 - 0.25 * diff)
    over = (price - budget) / max(budget, 1e-9)
    if over <= 0.10:
        return 0.85
    if over <= 0.20:
        return 0.65
    if over <= 0.35:
        return 0.40
    return 0.10


def budget_level_score(price: float, budget_level: str | None) -> float:
    if not budget_level or pd.isna(price):
        return 0.6
    if budget_level == "low":
        if price <= 3:
            return 1.0
        if price <= 5:
            return 0.85
        if price <= 8:
            return 0.55
        return 0.25
    if budget_level == "upper_medium":
        if 5 <= price <= 15:
            return 1.0
        if 3 <= price < 5 or 15 < price <= 25:
            return 0.80
        if price < 3:
            return 0.55
        return 0.45
    if budget_level == "high":
        if price >= 15:
            return 1.0
        if price >= 10:
            return 0.85
        if price >= 5:
            return 0.55
        return 0.30
    return 0.6


def fuzzy_area_score(area: float, target_area: float | None) -> float:
    if target_area is None or pd.isna(area):
        return 0.6
    diff = abs(area - target_area) / max(target_area, 1e-9)
    if diff <= 0.10:
        return 1.0
    if diff <= 0.25:
        return 0.8
    if diff <= 0.50:
        return 0.5
    return 0.2


def location_match_score(row_district: str, desired_district: str | None, location_mode: str = "unspecified") -> float:
    if location_mode == "central" and desired_district is None:
        zone_scores = {
            "central": 1.0,
            "urban": 0.70,
            "developing": 0.55,
            "suburban": 0.35,
            "unknown": 0.20,
        }
        from Data_Preprocessing.parsers import location_zone
        return zone_scores.get(location_zone(row_district), 0.4)
    if desired_district is None:
        return 0.6
    return district_similarity(row_district, desired_district)


def amenity_match_score(row_tags: list[str], desired_tags: list[str]) -> float:
    if not desired_tags:
        return 0.6
    if not isinstance(row_tags, list):
        return 0.0
    return len(set(row_tags) & set(desired_tags)) / max(len(set(desired_tags)), 1)


def property_type_score(row_type: str, preferred_type: str | None) -> float:
    if not preferred_type:
        return 0.6
    if row_type == preferred_type:
        return 1.0
    compatible = {
        "house": {"townhouse", "villa"},
        "townhouse": {"house", "villa"},
        "apartment": set(),
        "land": set(),
    }
    return 0.65 if row_type in compatible.get(preferred_type, set()) else 0.25


def purpose_match_score(row: pd.Series, purpose: str) -> float:
    mapping = {
        "family": "family_suitability_score",
        "business": "business_potential_score",
        "rental": "rental_potential_score",
        "investment": "investment_potential_score",
    }
    if purpose in mapping:
        return float(row.get(mapping[purpose], 0.0))
    return max(
        float(row.get("family_suitability_score", 0.0)),
        float(row.get("business_potential_score", 0.0)),
        float(row.get("rental_potential_score", 0.0)),
        float(row.get("investment_potential_score", 0.0)),
    )

WEIGHTS_BY_PURPOSE = {
    "family": {"price": 0.24, "location": 0.22, "legal": 0.18, "area": 0.10, "amenity": 0.10, "type": 0.04, "purpose": 0.12},
    "business": {"price": 0.18, "location": 0.22, "legal": 0.12, "area": 0.08, "amenity": 0.10, "type": 0.08, "purpose": 0.22},
    "rental": {"price": 0.18, "location": 0.24, "legal": 0.12, "area": 0.08, "amenity": 0.10, "type": 0.06, "purpose": 0.22},
    "investment": {"price": 0.25, "location": 0.20, "legal": 0.18, "area": 0.05, "amenity": 0.05, "type": 0.05, "purpose": 0.22},
    "general": {"price": 0.23, "location": 0.22, "legal": 0.15, "area": 0.09, "amenity": 0.08, "type": 0.06, "purpose": 0.17},
}


def hard_constraint_pass(row: pd.Series, profile: dict, relax_level: int = 0) -> bool:
    # relax_level=0 chặt; 1 nới phòng ngủ/vị trí; 2 nới pháp lý nhưng phạt bằng điểm
    budget = profile.get("budget_billion")
    if budget is not None and not pd.isna(row.get("price_billion")):
        max_over = 0.10 if relax_level == 0 else (0.20 if relax_level == 1 else 0.35)
        if row.get("price_billion") > budget * (1 + max_over):
            return False
    desired_district = profile.get("district")
    if desired_district is not None:
        base_min = float(profile.get("min_location_similarity", 0.0) or 0.0)
        if relax_level == 0:
            min_sim = base_min
        elif relax_level == 1:
            min_sim = min(base_min, 0.50)
        else:
            min_sim = min(base_min, 0.30)
        if min_sim > 0 and district_similarity(row.get("district"), desired_district) < min_sim:
            return False
    if profile.get("bedrooms_min") is not None and relax_level < 1:
        if pd.isna(row.get("bedrooms")) or row.get("bedrooms") < profile["bedrooms_min"]:
            return False
    if profile.get("legal_required") and relax_level < 2:
        if row.get("legal_class") != "clear_ownership":
            return False
    return True


def score_property(row: pd.Series, profile: dict) -> tuple[float, dict[str, float]]:
    budget = profile.get("budget_billion")
    detail = {
        "price": fuzzy_budget_score(row.get("price_billion", np.nan), budget)
        if budget is not None else budget_level_score(row.get("price_billion", np.nan), profile.get("budget_level")),
        "location": location_match_score(row.get("district"), profile.get("district"), profile.get("location_mode", "unspecified")),
        "legal": float(row.get("legal_score", 0.2)) if profile.get("legal_required") else 0.7 + 0.3 * float(row.get("legal_score", 0.2)),
        "area": fuzzy_area_score(row.get("area_m2", np.nan), profile.get("target_area_m2")),
        "amenity": amenity_match_score(row.get("amenity_tags", []), profile.get("desired_amenities", [])),
        "type": property_type_score(row.get("property_type"), profile.get("preferred_property_type")),
        "purpose": purpose_match_score(row, profile.get("purpose", "general")),
    }
    weights = WEIGHTS_BY_PURPOSE.get(profile.get("purpose", "general"), WEIGHTS_BY_PURPOSE["general"]).copy()
    for key, multiplier in profile.get("preference_weights", {}).items():
        if key in weights:
            weights[key] *= float(multiplier)
    total_weight = sum(weights.values())
    base = sum(detail[k] * weights[k] for k in weights) / max(total_weight, 1e-9)
    risk_penalty = 1 - 0.30 * float(row.get("risk_score", 0.0))
    quality_factor = 0.85 + 0.15 * float(row.get("data_quality_score", 0.5))
    final = base * risk_penalty * quality_factor
    return float(max(0.0, min(1.0, final))), detail
