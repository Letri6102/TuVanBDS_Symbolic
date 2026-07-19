"""Parse câu hỏi tiếng Việt thành symbolic profile."""
from __future__ import annotations

import re
from typing import Any

from Data_Preprocessing.parsers import AMENITY_KEYWORDS, NUMBER_RE, extract_district, infer_property_type, vn_float
from Data_Preprocessing.text_utils import clean_text, norm_text


def _add_unique(values: list[str], *items: str) -> list[str]:
    for item in items:
        if item and item not in values:
            values.append(item)
    return values


def parse_user_query(query: str) -> dict[str, Any]:
    q_raw = clean_text(query)
    q = norm_text(query)
    profile: dict[str, Any] = {
        "raw_query": q_raw,
        "query_intent": "recommendation",
        "budget_billion": None,
        "budget_level": None,
        "target_area_m2": None,
        "district": None,
        "location_mode": "unspecified",
        "min_location_similarity": 0.0,
        "bedrooms_min": None,
        "legal_required": False,
        "desired_amenities": [],
        "purpose": "general",
        "customer_group": None,
        "preferred_property_type": None,
        "preference_weights": {},
        "semantic_notes": [],
    }

    if any(k in q for k in ["nen mua", "nhu the nao", "tu van", "goi y", "phu hop voi tai chinh", "tai chinh"]):
        profile["query_intent"] = "advice"

    m = re.search(rf"({NUMBER_RE})\s*(ty|tỷ)", q_raw.lower())
    if m:
        profile["budget_billion"] = vn_float(m.group(1))

    m = re.search(rf"({NUMBER_RE})\s*(m2|m²)", q_raw.lower())
    if m:
        profile["target_area_m2"] = vn_float(m.group(1))

    m = re.search(r"(\d+)\s*(pn|phong ngu|phòng ngủ)", q)
    if m:
        profile["bedrooms_min"] = int(m.group(1))

    district = extract_district(q_raw)
    if district != "Không rõ":
        profile["district"] = district
        profile["location_mode"] = "nearby" if any(k in q for k in ["loanh quanh", "quanh", "gan", "lan can", "khu vuc", "xung quanh"]) else "exact_or_similar"
        profile["min_location_similarity"] = 0.65 if profile["location_mode"] == "nearby" else 0.75

    if any(k in q for k in ["gan trung tam", "trung tam", "khu trung tam"]):
        profile["location_mode"] = "central"
        profile["semantic_notes"].append("near_center")

    if any(k in q for k in ["tai chinh hoi rung rinh", "rung rinh", "du dai", "ngan sach thoai mai"]):
        profile["budget_level"] = "upper_medium"
        profile["semantic_notes"].append("comfortable_budget")
    elif any(k in q for k in ["gia re", "vua tien", "tai chinh han che", "ngan sach thap"]):
        profile["budget_level"] = "low"
    elif any(k in q for k in ["cao cap", "hang sang", "vip"]):
        profile["budget_level"] = "high"

    if any(k in q for k in ["phap ly ro", "so hong", "so do", "so rieng", "an toan phap ly"]):
        profile["legal_required"] = True
    elif profile.get("query_intent") == "advice":
        profile["legal_required"] = True

    if any(k in q for k in ["kinh doanh", "mat tien", "mo cua hang", "cua hang", "van phong"]):
        profile["purpose"] = "business"
    elif any(k in q for k in ["dau tu", "tang gia", "sinh loi", "tiem nang"]):
        profile["purpose"] = "investment"
    elif any(k in q for k in ["cho thue", "dong tien", "khai thac thue"]):
        profile["purpose"] = "rental"
    elif any(k in q for k in ["gia dinh", "cho con", "an cu", "de o", "để ở"]):
        profile["purpose"] = "family"

    desired = []
    for tag, keywords in AMENITY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            desired.append(tag)
    if any(k in q for k in ["gan truong", "truong hoc", "cho con", "con nho", "gia dinh co con"]):
        desired = _add_unique(desired, "school")
        profile["customer_group"] = "family_with_children"
        profile["purpose"] = "family"
    if any(k in q for k in ["yen tinh", "an ninh", "khu dan tri", "an toan"]):
        desired = _add_unique(desired, "security")
    if any(k in q for k in ["gan metro", "metro", "ha tang", "quy hoach", "duong vanh dai"]):
        desired = _add_unique(desired, "planning_infra")
        if profile["purpose"] == "general":
            profile["purpose"] = "investment"
    profile["desired_amenities"] = sorted(set(desired))

    ptype = infer_property_type(q)
    if ptype != "unknown":
        profile["preferred_property_type"] = ptype

    weights: dict[str, float] = {}
    if profile.get("district"):
        weights["location"] = 1.25 if profile["location_mode"] == "nearby" else 1.15
    if profile.get("budget_level") == "upper_medium":
        weights["price"] = 0.90
        weights["purpose"] = 1.10
    if profile.get("customer_group") == "family_with_children":
        weights["amenity"] = 1.25
        weights["purpose"] = 1.20
        weights["legal"] = 1.10
    if profile.get("legal_required"):
        weights["legal"] = max(weights.get("legal", 1.0), 1.20)
    profile["preference_weights"] = weights

    return profile
