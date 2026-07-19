"""Tạo symbolic bands và các điểm fuzzy-symbolic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def has_tag(tags, tag: str) -> bool:
    return isinstance(tags, list) and tag in tags


def price_band(price: float) -> str:
    if pd.isna(price):
        return "unknown"
    if price < 3:
        return "very_low"
    if price < 5:
        return "low"
    if price < 10:
        return "medium"
    if price < 20:
        return "medium_high"
    return "high"


def area_band(area: float) -> str:
    if pd.isna(area):
        return "unknown"
    if area < 50:
        return "small"
    if area < 90:
        return "medium"
    if area < 150:
        return "large"
    return "very_large"


def bedrooms_band(bedrooms: float) -> str:
    if pd.isna(bedrooms):
        return "unknown"
    if bedrooms <= 1:
        return "studio_or_1br"
    if bedrooms == 2:
        return "2br"
    if bedrooms == 3:
        return "3br"
    return "4br_plus"


def road_band(road_width: float) -> str:
    if pd.isna(road_width):
        return "unknown"
    if road_width < 3:
        return "small_alley"
    if road_width < 5:
        return "motorbike_alley"
    if road_width < 8:
        return "car_access"
    return "wide_road"


def frontage_band(frontage: float) -> str:
    if pd.isna(frontage):
        return "unknown"
    if frontage < 3.5:
        return "narrow"
    if frontage < 5:
        return "standard"
    if frontage < 8:
        return "wide"
    return "very_wide"


def symbolic_level(score: float, low: float = 0.4, mid: float = 0.7) -> str:
    if pd.isna(score):
        return "unknown"
    if score < low:
        return "low"
    if score < mid:
        return "medium"
    return "high"


def price_reasonable_score(row: pd.Series) -> float:
    price = row.get("price_per_m2_million", np.nan)
    median = row.get("district_median_price_per_m2", np.nan)
    if pd.isna(price) or pd.isna(median) or median <= 0:
        return 0.5
    ratio = price / median
    if ratio <= 0.85:
        return 1.0
    if ratio <= 1.0:
        return 0.9
    if ratio <= 1.2:
        return 0.7
    if ratio <= 1.5:
        return 0.4
    return 0.15


def family_suitability(row: pd.Series) -> float:
    score = 0.0
    if row.get("bedrooms", np.nan) >= 2:
        score += 0.25
    if row.get("bathrooms", np.nan) >= 2:
        score += 0.15
    if row.get("area_m2", np.nan) >= 50:
        score += 0.20
    if row.get("legal_score", 0) >= 0.8:
        score += 0.18
    if row.get("location_zone") in ["central", "urban", "developing"]:
        score += 0.07
    if has_tag(row.get("amenity_tags"), "school"):
        score += 0.08
    if has_tag(row.get("amenity_tags"), "market"):
        score += 0.04
    if has_tag(row.get("amenity_tags"), "hospital"):
        score += 0.03
    if has_tag(row.get("amenity_tags"), "security"):
        score += 0.05
    return min(score, 1.0)


def business_potential(row: pd.Series) -> float:
    score = 0.0
    if row.get("property_type") in ["townhouse", "villa", "house"]:
        score += 0.18
    if has_tag(row.get("amenity_tags"), "business"):
        score += 0.22
    frontage = row.get("frontage_m", np.nan)
    road = row.get("road_width_m", np.nan)
    if not pd.isna(frontage) and frontage >= 6:
        score += 0.22
    elif not pd.isna(frontage) and frontage >= 4:
        score += 0.16
    if not pd.isna(road) and road >= 8:
        score += 0.20
    elif not pd.isna(road) and road >= 5:
        score += 0.14
    if has_tag(row.get("amenity_tags"), "car_access"):
        score += 0.10
    if row.get("location_zone") in ["central", "urban"]:
        score += 0.08
    return min(score, 1.0)


def rental_potential(row: pd.Series) -> float:
    score = 0.0
    if row.get("property_type") in ["apartment", "house", "townhouse"]:
        score += 0.15
    if has_tag(row.get("amenity_tags"), "rental"):
        score += 0.25
    if row.get("location_zone") == "central":
        score += 0.22
    elif row.get("location_zone") in ["urban", "developing"]:
        score += 0.16
    if 1 <= row.get("bedrooms", 0) <= 3:
        score += 0.12
    if row.get("legal_score", 0) >= 0.8:
        score += 0.12
    if has_tag(row.get("amenity_tags"), "furnished"):
        score += 0.06
    if has_tag(row.get("amenity_tags"), "mall") or has_tag(row.get("amenity_tags"), "market"):
        score += 0.05
    score += 0.05 * row.get("price_reasonable_score", 0.5)
    return min(score, 1.0)


def investment_potential(row: pd.Series) -> float:
    score = 0.0
    if row.get("location_zone") == "central":
        score += 0.25
    elif row.get("location_zone") == "developing":
        score += 0.22
    elif row.get("location_zone") == "urban":
        score += 0.16
    if row.get("legal_score", 0) >= 0.8:
        score += 0.18
    elif row.get("legal_score", 0) >= 0.6:
        score += 0.10
    score += 0.22 * row.get("price_reasonable_score", 0.5)
    if has_tag(row.get("amenity_tags"), "planning_infra"):
        score += 0.14
    if has_tag(row.get("amenity_tags"), "urgent_sale"):
        score += 0.06
    if row.get("property_type") in ["land", "townhouse", "house", "apartment"]:
        score += 0.08
    score += 0.07 * row.get("data_quality_score", 0.5)
    return min(score, 1.0)


def risk_score(row: pd.Series) -> float:
    score = 0.0
    if row.get("legal_class") == "unknown":
        score += 0.25
    elif row.get("legal_class") == "disputed_planning":
        score += 0.35
    elif row.get("legal_class") in ["pending", "contract_based"]:
        score += 0.15
    dq = row.get("data_quality_score", 1.0)
    if dq < 0.6:
        score += 0.25
    elif dq < 0.8:
        score += 0.10
    if row.get("price_reasonable_score", 0.5) < 0.4:
        score += 0.15
    flags = row.get("data_flags", []) if isinstance(row.get("data_flags", []), list) else []
    if "abnormal_price_per_m2" in flags:
        score += 0.20
    if "missing_district" in flags:
        score += 0.10
    return min(score, 1.0)


def add_symbolic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    med = df.groupby("district")["price_per_m2_million"].transform("median")
    global_med = df["price_per_m2_million"].median()
    df["district_median_price_per_m2"] = med.fillna(global_med)

    df["price_band"] = df["price_billion"].apply(price_band)
    df["area_band"] = df["area_m2"].apply(area_band)
    df["bedrooms_band"] = df["bedrooms"].apply(bedrooms_band)
    df["road_band"] = df["road_width_m"].apply(road_band)
    df["frontage_band"] = df["frontage_m"].apply(frontage_band)

    df["price_reasonable_score"] = df.apply(price_reasonable_score, axis=1)
    df["family_suitability_score"] = df.apply(family_suitability, axis=1)
    df["business_potential_score"] = df.apply(business_potential, axis=1)
    df["rental_potential_score"] = df.apply(rental_potential, axis=1)
    df["investment_potential_score"] = df.apply(investment_potential, axis=1)
    df["risk_score"] = df.apply(risk_score, axis=1)

    for col in [
        "family_suitability_score",
        "business_potential_score",
        "rental_potential_score",
        "investment_potential_score",
        "risk_score",
        "data_quality_score",
    ]:
        df[col.replace("_score", "_level")] = df[col].apply(symbolic_level)

    return df
