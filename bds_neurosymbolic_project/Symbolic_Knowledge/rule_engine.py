"""Symbolic rule engine cho BĐS."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SYMBOLIC_FACT_CATALOG = {
    "low_legal_risk": "Pháp lý rõ, rủi ro pháp lý thấp.",
    "suitable_for_family": "Phù hợp nhu cầu ở gia đình.",
    "suitable_for_business": "Có tiềm năng kinh doanh.",
    "suitable_for_rental": "Có tiềm năng khai thác cho thuê.",
    "suitable_for_investment": "Có tiềm năng đầu tư.",
    "good_price_by_area": "Giá/m² hợp lý so với khu vực.",
    "good_car_access": "Đường vào phù hợp xe hơi hoặc đường rộng.",
    "high_data_confidence": "Dữ liệu tương đối đầy đủ và đáng tin cậy.",
}

RISK_FLAG_CATALOG = {
    "need_legal_verification": "Cần kiểm tra thêm pháp lý.",
    "possibly_overpriced": "Giá/m² có thể cao hơn mặt bằng khu vực.",
    "low_data_confidence": "Dữ liệu thiếu nhiều trường quan trọng.",
    "missing_location_warning": "Vị trí chưa được nhận diện rõ.",
    "missing_price_warning": "Thiếu thông tin giá.",
    "missing_area_warning": "Thiếu thông tin diện tích.",
    "planning_or_dispute_warning": "Có dấu hiệu tranh chấp hoặc quy hoạch treo.",
}

STRICT_RULE_BASE = [
    {
        "rule_id": "R01_LEGAL_CLEAR",
        "type": "fact",
        "conclusion": "low_legal_risk",
        "condition": "legal_class == clear_ownership",
        "strength": 1.0,
        "description": "Pháp lý rõ ràng nếu có sổ đỏ/sổ hồng.",
    },
    {
        "rule_id": "R02_LEGAL_WARNING",
        "type": "risk",
        "conclusion": "need_legal_verification",
        "condition": "legal_class in {unknown, pending, contract_based, other, disputed_planning}",
        "strength": 0.9,
        "description": "Pháp lý chưa đủ chắc chắn nên cần kiểm tra.",
    },
    {
        "rule_id": "R03_FAMILY_STRONG",
        "type": "fact",
        "conclusion": "suitable_for_family",
        "condition": "family_suitability_score >= 0.70",
        "strength": 0.85,
        "description": "Điểm phù hợp gia đình cao.",
    },
    {
        "rule_id": "R04_BUSINESS_STRONG",
        "type": "fact",
        "conclusion": "suitable_for_business",
        "condition": "business_potential_score >= 0.65",
        "strength": 0.85,
        "description": "Điểm tiềm năng kinh doanh cao.",
    },
    {
        "rule_id": "R05_RENTAL_STRONG",
        "type": "fact",
        "conclusion": "suitable_for_rental",
        "condition": "rental_potential_score >= 0.65",
        "strength": 0.85,
        "description": "Điểm tiềm năng cho thuê cao.",
    },
    {
        "rule_id": "R06_INVESTMENT_STRONG",
        "type": "fact",
        "conclusion": "suitable_for_investment",
        "condition": "investment_potential_score >= 0.70",
        "strength": 0.85,
        "description": "Điểm tiềm năng đầu tư cao.",
    },
    {
        "rule_id": "R07_GOOD_PRICE",
        "type": "fact",
        "conclusion": "good_price_by_area",
        "condition": "price_reasonable_score >= 0.80",
        "strength": 0.75,
        "description": "Giá/m² tốt so với trung vị khu vực.",
    },
    {
        "rule_id": "R08_CAR_ACCESS",
        "type": "fact",
        "conclusion": "good_car_access",
        "condition": "road_band in {car_access, wide_road} or car_access tag",
        "strength": 0.7,
        "description": "Đường vào tốt hoặc có tag xe hơi/ô tô.",
    },
    {
        "rule_id": "R09_LOW_CONFIDENCE",
        "type": "risk",
        "conclusion": "low_data_confidence",
        "condition": "data_quality_score < 0.70",
        "strength": 0.9,
        "description": "Dữ liệu thiếu nhiều trường quan trọng.",
    },
    {
        "rule_id": "R10_OVERPRICED",
        "type": "risk",
        "conclusion": "possibly_overpriced",
        "condition": "price_reasonable_score < 0.40",
        "strength": 0.75,
        "description": "Giá/m² cao hơn đáng kể so với khu vực.",
    },
    {
        "rule_id": "R11_PLANNING_DISPUTE",
        "type": "risk",
        "conclusion": "planning_or_dispute_warning",
        "condition": "legal_class == disputed_planning",
        "strength": 0.95,
        "description": "Có dấu hiệu tranh chấp hoặc quy hoạch treo.",
    },
]


def _has_tag(row: pd.Series, tag: str) -> bool:
    tags = row.get("amenity_tags", [])
    return isinstance(tags, list) and tag in tags


def apply_rules_to_row(row: pd.Series) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    facts: list[str] = []
    risks: list[str] = []
    triggered: list[str] = []
    evidence: list[dict[str, Any]] = []

    def fire(rule_id: str, conclusion: str, kind: str, strength: float, ev: str) -> None:
        triggered.append(rule_id)
        if kind == "fact" and conclusion not in facts:
            facts.append(conclusion)
        if kind == "risk" and conclusion not in risks:
            risks.append(conclusion)
        evidence.append({"rule_id": rule_id, "conclusion": conclusion, "strength": strength, "evidence": ev})

    if row.get("legal_class") == "clear_ownership":
        fire("R01_LEGAL_CLEAR", "low_legal_risk", "fact", 1.0, f"legal_class={row.get('legal_class')}")
    elif row.get("legal_class") in ["unknown", "pending", "contract_based", "other", "disputed_planning"]:
        fire("R02_LEGAL_WARNING", "need_legal_verification", "risk", 0.9, f"legal_class={row.get('legal_class')}")
    if row.get("legal_class") == "disputed_planning":
        fire("R11_PLANNING_DISPUTE", "planning_or_dispute_warning", "risk", 0.95, f"legal_class={row.get('legal_class')}")

    if row.get("family_suitability_score", 0) >= 0.70:
        fire("R03_FAMILY_STRONG", "suitable_for_family", "fact", 0.85, f"family_score={row.get('family_suitability_score'):.2f}")
    if row.get("business_potential_score", 0) >= 0.65:
        fire("R04_BUSINESS_STRONG", "suitable_for_business", "fact", 0.85, f"business_score={row.get('business_potential_score'):.2f}")
    if row.get("rental_potential_score", 0) >= 0.65:
        fire("R05_RENTAL_STRONG", "suitable_for_rental", "fact", 0.85, f"rental_score={row.get('rental_potential_score'):.2f}")
    if row.get("investment_potential_score", 0) >= 0.70:
        fire("R06_INVESTMENT_STRONG", "suitable_for_investment", "fact", 0.85, f"investment_score={row.get('investment_potential_score'):.2f}")
    if row.get("price_reasonable_score", 0.5) >= 0.80:
        fire("R07_GOOD_PRICE", "good_price_by_area", "fact", 0.75, f"price_reasonable_score={row.get('price_reasonable_score'):.2f}")
    if row.get("road_band") in ["car_access", "wide_road"] or _has_tag(row, "car_access"):
        fire("R08_CAR_ACCESS", "good_car_access", "fact", 0.70, f"road_band={row.get('road_band')}, tags={row.get('amenity_tags')}")

    if row.get("data_quality_score", 1) < 0.70:
        fire("R09_LOW_CONFIDENCE", "low_data_confidence", "risk", 0.90, f"data_quality_score={row.get('data_quality_score'):.2f}")
    if row.get("price_reasonable_score", 0.5) < 0.40:
        fire("R10_OVERPRICED", "possibly_overpriced", "risk", 0.75, f"price_reasonable_score={row.get('price_reasonable_score'):.2f}")

    flags = row.get("data_flags", []) if isinstance(row.get("data_flags", []), list) else []
    if "missing_district" in flags:
        risks.append("missing_location_warning")
    if "missing_price" in flags:
        risks.append("missing_price_warning")
    if "missing_area" in flags:
        risks.append("missing_area_warning")

    if row.get("data_quality_score", 0) >= 0.85 and "high_data_confidence" not in facts:
        facts.append("high_data_confidence")

    return sorted(set(facts)), sorted(set(risks)), triggered, evidence


def apply_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    outputs = df.apply(apply_rules_to_row, axis=1)
    df["symbolic_facts"] = [o[0] for o in outputs]
    df["risk_flags"] = [o[1] for o in outputs]
    df["triggered_rules"] = [o[2] for o in outputs]
    df["rule_evidence"] = [o[3] for o in outputs]
    return df


def save_rule_artifacts(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in {
        "symbolic_rule_base.json": STRICT_RULE_BASE,
        "symbolic_fact_catalog.json": SYMBOLIC_FACT_CATALOG,
        "risk_flag_catalog.json": RISK_FLAG_CATALOG,
    }.items():
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
