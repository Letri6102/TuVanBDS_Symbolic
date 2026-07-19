"""Sinh giải thích dựa trên score, facts, risks và evidence."""
from __future__ import annotations

import pandas as pd


def fmt_num(x, suffix="", ndigits=2):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):.{ndigits}f}{suffix}"


def build_explanation(row: pd.Series, profile: dict, detail: dict[str, float]) -> str:
    parts: list[str] = []
    parts.append(f"Điểm phù hợp tổng: {fmt_num(row.get('final_score', None))}.")
    if row.get("semantic_score") is not None:
        parts.append(f"Điểm symbolic={fmt_num(row.get('symbolic_score'))}, semantic={fmt_num(row.get('semantic_score'))}.")
    if profile.get("budget_billion") is not None:
        parts.append(f"Giá {fmt_num(row.get('price_billion'), ' tỷ')} so với ngân sách {fmt_num(profile.get('budget_billion'), ' tỷ')}, điểm giá {detail.get('price', 0):.2f}.")
    if profile.get("district") is not None:
        parts.append(f"Vị trí {row.get('district')} có điểm tương đồng {detail.get('location', 0):.2f} với {profile.get('district')}.")
    if row.get("match_type") == "similar_location_match":
        parts.append("Đây là lựa chọn ở khu vực tương đồng với vị trí mong muốn.")
    elif row.get("match_type") == "expanded_location_match":
        parts.append("Đây là lựa chọn thay thế sau khi nới rộng khu vực tìm kiếm.")
    desired = profile.get("desired_amenities", [])
    if desired:
        matched = sorted(set(row.get("amenity_tags", []) if isinstance(row.get("amenity_tags", []), list) else []) & set(desired))
        if matched:
            parts.append("Tiện ích khớp nhu cầu: " + ", ".join(matched) + ".")
    parts.append(f"Pháp lý thuộc nhóm {row.get('legal_class')}, legal_score={fmt_num(row.get('legal_score'))}.")
    facts = row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []
    risks = row.get("risk_flags", []) if isinstance(row.get("risk_flags", []), list) else []
    if facts:
        parts.append("Facts suy luận: " + ", ".join(facts) + ".")
    if risks:
        parts.append("Cảnh báo: " + ", ".join(risks) + ".")
    evidence = row.get("rule_evidence", []) if isinstance(row.get("rule_evidence", []), list) else []
    if evidence:
        ev_text = []
        for ev in evidence[:3]:
            if isinstance(ev, dict):
                ev_text.append(f"{ev.get('rule_id')}: {ev.get('evidence')}")
        if ev_text:
            parts.append("Bằng chứng luật: " + " | ".join(ev_text) + ".")
    return " ".join(parts)
