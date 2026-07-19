"""LLM advisory agent cho phần thuyết minh Top 3 BĐS.

Agent nhận kết quả neuro-symbolic đã tính sẵn, đóng gói thành context có cấu
trúc và yêu cầu LLM viết bài phân tích tự nhiên. Nếu chưa cấu hình LLM API,
module trả về fallback narrative dựa trên cùng dữ liệu để app/Colab vẫn chạy.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_money(value: Any) -> str:
    x = _safe_float(value)
    if x is None:
        return "không rõ giá"
    return f"{x:.2f} tỷ"


def _fmt_area(value: Any) -> str:
    x = _safe_float(value)
    if x is None:
        return "không rõ diện tích"
    return f"{x:.0f} m2"


def _fmt_score(value: Any) -> str:
    x = _safe_float(value)
    if x is None:
        return "N/A"
    return f"{x:.2f}"


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    return []


def _take(values: list, n: int = 5) -> list:
    return values[:n] if isinstance(values, list) else []


def _score_label(score: Any) -> str:
    x = _safe_float(score)
    if x is None:
        return "chưa rõ"
    if x >= 0.80:
        return "rất phù hợp"
    if x >= 0.65:
        return "phù hợp"
    if x >= 0.50:
        return "tạm phù hợp"
    return "yếu"


def _profile_summary(profile: dict) -> dict:
    return {
        "intent": profile.get("query_intent"),
        "budget_billion": profile.get("budget_billion"),
        "budget_level": profile.get("budget_level"),
        "district": profile.get("district"),
        "location_mode": profile.get("location_mode"),
        "preferred_property_type": profile.get("preferred_property_type"),
        "purpose": profile.get("purpose"),
        "customer_group": profile.get("customer_group"),
        "legal_required": profile.get("legal_required"),
        "desired_amenities": profile.get("desired_amenities", []),
        "preference_weights": profile.get("preference_weights", {}),
    }


def build_top3_context(query: str, profile: dict, recs: pd.DataFrame) -> dict:
    """Tạo context đủ bằng chứng cho LLM, tránh phải đọc toàn bộ dataframe."""
    items = []
    for rank, (_, row) in enumerate(recs.head(3).iterrows(), start=1):
        match_scores = {
            key.replace("_match_score", ""): _safe_float(row.get(key))
            for key in recs.columns
            if key.endswith("_match_score")
        }
        items.append(
            {
                "rank": rank,
                "property_id": row.get("property_id"),
                "title": row.get("title"),
                "district": row.get("district"),
                "property_type": row.get("property_type"),
                "price_billion": _safe_float(row.get("price_billion")),
                "area_m2": _safe_float(row.get("area_m2")),
                "price_per_m2_million": _safe_float(row.get("price_per_m2_million")),
                "legal_class": row.get("legal_class"),
                "legal_score": _safe_float(row.get("legal_score")),
                "data_quality_score": _safe_float(row.get("data_quality_score")),
                "risk_score": _safe_float(row.get("risk_score")),
                "amenity_tags": _take(_as_list(row.get("amenity_tags")), 8),
                "symbolic_facts": _take(_as_list(row.get("symbolic_facts")), 10),
                "risk_flags": _take(_as_list(row.get("risk_flags")), 8),
                "final_score": _safe_float(row.get("final_score")),
                "symbolic_score": _safe_float(row.get("symbolic_score")),
                "semantic_score": _safe_float(row.get("semantic_score")),
                "semantic_weight": _safe_float(row.get("semantic_weight")),
                "semantic_backend": row.get("semantic_backend"),
                "semantic_method": row.get("semantic_method"),
                "match_type": row.get("match_type"),
                "match_scores": match_scores,
                "explanation": row.get("explanation"),
            }
        )
    semantic_backend = items[0].get("semantic_backend") if items else "semantic"
    semantic_method = items[0].get("semantic_method") if items else "semantic_retrieval"
    return {
        "query": query,
        "interpreted_profile": _profile_summary(profile),
        "ranking_note": (
            "final_score là tổ hợp symbolic_score và semantic_score. "
            "symbolic_score đến từ fuzzy scoring, luật pháp lý, ngân sách, vị trí, tiện ích, rủi ro và chất lượng dữ liệu. "
            f"semantic_score đến từ backend {semantic_backend} ({semantic_method})."
        ),
        "top3": items,
    }


def build_llm_prompt(context: dict) -> list[dict[str, str]]:
    system = (
        "Bạn là chuyên viên tư vấn bất động sản và là agent giải thích hệ thống neuro-symbolic. "
        "Bạn chỉ được dùng dữ liệu trong context, không tự thêm thông tin thị trường, pháp lý hoặc địa chỉ không có trong context. "
        "Hãy chuyển các điểm số mờ, luật symbolic, semantic score, facts và risks thành bài phân tích tiếng Việt tự nhiên, thuyết phục nhưng trung thực. "
        "Luôn nêu ưu điểm, nhược điểm/rủi ro của từng BĐS và phần cần kiểm chứng trước giao dịch."
    )
    user = (
        "Hãy viết bài tư vấn cho người dùng theo cấu trúc:\n"
        "1. Nhận định nhanh theo nhu cầu.\n"
        "2. Phân tích Top 3 BĐS, mỗi BĐS gồm: vì sao phù hợp, ưu điểm, nhược điểm/rủi ro, phù hợp với ai.\n"
        "3. So sánh và khuyến nghị nên ưu tiên BĐS nào.\n"
        "4. Các việc cần kiểm chứng trước khi quyết định.\n\n"
        "Context JSON:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_openai_chat(messages: list[dict[str, str]]) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Thiếu package openai. Hãy cài: pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa có OPENAI_API_KEY nên không thể gọi LLM.")

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.25"))
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _item_narrative(item: dict, budget: float | None) -> list[str]:
    lines = []
    title = item.get("title") or item.get("property_id")
    facts = item.get("symbolic_facts") or []
    risks = item.get("risk_flags") or []
    amenities = item.get("amenity_tags") or []
    match_scores = item.get("match_scores") or {}

    lines.append(
        f"**{item['rank']}. {item.get('property_id')} - {title}**: "
        f"ở {item.get('district')}, loại {item.get('property_type')}, "
        f"giá {_fmt_money(item.get('price_billion'))}, diện tích {_fmt_area(item.get('area_m2'))}. "
        f"Điểm tổng {_fmt_score(item.get('final_score'))} ({_score_label(item.get('final_score'))}), "
        f"trong đó symbolic={_fmt_score(item.get('symbolic_score'))}, semantic={_fmt_score(item.get('semantic_score'))}."
    )

    strengths = []
    if budget and item.get("price_billion") is not None:
        price = item["price_billion"]
        if price <= budget:
            strengths.append(f"giá nằm trong ngân sách khoảng {_fmt_money(budget)}")
        elif price <= budget * 1.15:
            strengths.append("giá hơi vượt ngân sách nhưng vẫn trong vùng có thể cân nhắc")
    if item.get("legal_class") == "clear_ownership":
        strengths.append("pháp lý rõ hơn nhóm còn lại")
    if item.get("data_quality_score") and item["data_quality_score"] >= 0.75:
        strengths.append("dữ liệu tương đối đầy đủ")
    if facts:
        strengths.append("facts nổi bật: " + ", ".join(facts[:4]))
    if amenities:
        strengths.append("tiện ích/đặc điểm ghi nhận: " + ", ".join(amenities[:4]))
    if strengths:
        lines.append("Ưu điểm: " + "; ".join(strengths) + ".")

    weaknesses = []
    if item.get("legal_class") != "clear_ownership":
        weaknesses.append(f"pháp lý đang ở nhóm {item.get('legal_class')}")
    if item.get("area_m2") is None:
        weaknesses.append("thiếu diện tích chuẩn hoá")
    if item.get("price_per_m2_million") is None:
        weaknesses.append("chưa tính được giá/m2")
    if risks:
        weaknesses.append("cảnh báo: " + ", ".join(risks[:4]))
    if match_scores.get("type") is not None and match_scores["type"] < 0.6:
        weaknesses.append("loại hình chưa khớp hoàn toàn với nhu cầu")
    if weaknesses:
        lines.append("Điểm cần cân nhắc: " + "; ".join(weaknesses) + ".")

    if item.get("explanation"):
        lines.append(f"Lập luận hệ thống: {item['explanation']}")
    return lines


def build_fallback_agent_report(query: str, profile: dict, recs: pd.DataFrame) -> str:
    """Narrative fallback khi chưa gọi LLM API."""
    if recs.empty:
        return "Chưa có BĐS phù hợp để phân tích. Nên nới ngân sách, khu vực hoặc loại hình tìm kiếm."

    budget = profile.get("budget_billion")
    lines = [
        "## Bài tư vấn Top 3 BĐS",
        f"Với truy vấn: **{query}**, hệ thống đã chuyển nhu cầu thành profile symbolic và kết hợp điểm semantic từ "
        f"{build_top3_context(query, profile, recs).get('top3', [{}])[0].get('semantic_backend', 'semantic retrieval')}. "
        "Các lựa chọn dưới đây không chỉ được xếp hạng theo bảng điểm, mà còn được diễn giải theo ngân sách, pháp lý, dữ liệu, tiện ích và rủi ro.",
        "",
    ]

    for item in build_top3_context(query, profile, recs)["top3"]:
        lines.extend(_item_narrative(item, budget))
        lines.append("")

    best = recs.iloc[0]
    lines.append("## Khuyến nghị")
    lines.append(
        f"Nếu cần chọn trước để đi kiểm tra thực tế, nên ưu tiên **{best.get('property_id')}** vì có điểm tổng cao nhất "
        f"({_fmt_score(best.get('final_score'))}) và mức phù hợp symbolic/semantic tốt nhất trong Top 3. "
        "Tuy vậy, quyết định cuối cùng nên dựa trên kiểm tra sổ/pháp lý, hiện trạng nhà, quy hoạch và thương lượng giá thực tế."
    )
    return "\n".join(lines)


def build_agent_report(query: str, profile: dict, recs: pd.DataFrame, use_llm: bool = True) -> str:
    """Sinh bài phân tích Top 3 bằng LLM nếu có cấu hình, nếu không thì fallback."""
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    if use_llm and provider in {"openai", "auto"}:
        try:
            context = build_top3_context(query, profile, recs)
            return _call_openai_chat(build_llm_prompt(context))
        except Exception as exc:
            fallback = build_fallback_agent_report(query, profile, recs)
            return (
                f"{fallback}\n\n"
                f"_Ghi chú kỹ thuật: chưa gọi được LLM API nên đang dùng fallback narrative. Lý do: {exc}_"
            )
    return build_fallback_agent_report(query, profile, recs)
