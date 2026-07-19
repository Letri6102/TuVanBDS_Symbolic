"""Sinh tư vấn tự nhiên từ profile, Top-K và tri thức symbolic."""
from __future__ import annotations

import pandas as pd


def _fmt_money(x) -> str:
    if pd.isna(x):
        return "không rõ giá"
    return f"{float(x):.2f} tỷ"


def _fmt_area(x) -> str:
    if pd.isna(x):
        return "không rõ diện tích"
    return f"{float(x):.0f} m2"


PROPERTY_TYPE_VI = {
    "house": "nhà riêng",
    "townhouse": "nhà phố",
    "apartment": "căn hộ",
    "villa": "biệt thự",
    "land": "đất",
    "unknown": "BĐS",
}


LEGAL_VI = {
    "clear_ownership": "sổ hồng/sổ đỏ, pháp lý rõ",
    "pending": "đang chờ sổ, cần kiểm tra thêm",
    "contract_based": "hợp đồng/chứng từ mua bán, cần kiểm tra kỹ",
    "disputed_planning": "có dấu hiệu quy hoạch hoặc tranh chấp, cần thẩm định kỹ",
    "other": "pháp lý khác, cần kiểm tra thêm",
    "unknown": "chưa rõ pháp lý",
}


AMENITY_VI = {
    "school": "gần trường học",
    "market": "gần chợ/tiện ích mua sắm",
    "hospital": "gần bệnh viện",
    "security": "khu vực có yếu tố an ninh",
    "business": "có thể khai thác kinh doanh",
    "rental": "có tiềm năng cho thuê",
    "furnished": "có nội thất",
    "car_access": "đường xe hơi",
    "planning_infra": "có yếu tố hạ tầng/quy hoạch",
    "urgent_sale": "chủ cần bán gấp",
    "mall": "gần trung tâm thương mại",
}


def _fmt_int_feature(value, label: str) -> str | None:
    if pd.isna(value):
        return None
    try:
        return f"{int(float(value))} {label}"
    except Exception:
        return None


def _fmt_structure(row: pd.Series) -> str:
    parts = [
        _fmt_int_feature(row.get("bedrooms"), "phòng ngủ"),
        _fmt_int_feature(row.get("bathrooms"), "WC"),
        _fmt_int_feature(row.get("floors"), "tầng"),
    ]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else "chưa rõ số phòng/phòng tắm/số tầng"


def _property_type_vi(value) -> str:
    return PROPERTY_TYPE_VI.get(str(value), "BĐS")


def _legal_vi(value) -> str:
    return LEGAL_VI.get(str(value), "cần kiểm tra pháp lý")


def _amenities_vi(tags: list[str], limit: int = 4) -> str:
    if not isinstance(tags, list) or not tags:
        return "chưa ghi nhận tiện ích nổi bật"
    mapped = [AMENITY_VI.get(tag, tag) for tag in tags[:limit]]
    return ", ".join(mapped)


def _customer_strengths(row: pd.Series, profile: dict) -> list[str]:
    strengths: list[str] = []
    budget = profile.get("budget_billion")
    price = row.get("price_billion")
    if budget is not None and not pd.isna(price):
        if price <= budget:
            strengths.append("giá nằm trong ngân sách")
        elif price <= budget * 1.15:
            strengths.append("giá hơi vượt ngân sách nhưng vẫn có thể cân nhắc nếu thương lượng được")
    bedrooms_min = profile.get("bedrooms_min")
    bedrooms = row.get("bedrooms")
    if bedrooms_min is not None and not pd.isna(bedrooms) and bedrooms >= bedrooms_min:
        strengths.append(f"đáp ứng nhu cầu tối thiểu {bedrooms_min} phòng ngủ")
    if row.get("legal_class") == "clear_ownership":
        strengths.append("pháp lý rõ")
    if row.get("data_quality_score", 0) >= 0.85:
        strengths.append("thông tin tin đăng tương đối đầy đủ")
    facts = row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []
    if "good_price_by_area" in facts:
        strengths.append("giá/m2 đang tương đối tốt so với khu vực trong dữ liệu")
    if "good_car_access" in facts:
        strengths.append("đường vào thuận tiện cho xe hơi")
    if "suitable_for_family" in facts:
        strengths.append("phù hợp nhu cầu ở gia đình")
    if "suitable_for_rental" in facts:
        strengths.append("có thể cân nhắc khai thác cho thuê")
    return strengths[:4]


def _customer_cautions(row: pd.Series) -> list[str]:
    cautions: list[str] = []
    risks = row.get("risk_flags", []) if isinstance(row.get("risk_flags", []), list) else []
    if "need_legal_verification" in risks:
        cautions.append("cần kiểm tra kỹ pháp lý trước khi đặt cọc")
    if "possibly_overpriced" in risks:
        cautions.append("nên so sánh thêm giá/m2 khu vực để tránh mua cao")
    if "low_data_confidence" in risks:
        cautions.append("tin đăng còn thiếu dữ liệu, nên xác minh lại trực tiếp")
    if "missing_location_warning" in risks:
        cautions.append("vị trí cần xác minh lại")
    if pd.isna(row.get("bedrooms")):
        cautions.append("chưa rõ số phòng ngủ trong dữ liệu chuẩn hoá")
    if pd.isna(row.get("area_m2")):
        cautions.append("chưa rõ diện tích chuẩn hoá")
    return cautions[:3]


def _top_counts(df: pd.DataFrame, col: str, n: int = 4) -> list[str]:
    if df.empty or col not in df.columns:
        return []
    ignored = {"", "unknown", "không rõ", "khong ro", "nan", "none"}
    values = df[col].dropna().astype(str)
    values = values[~values.str.lower().isin(ignored)]
    return [str(v) for v in values.value_counts().head(n).index.tolist()]


def _filter_budget_pool(df: pd.DataFrame, budget: float | None) -> pd.DataFrame:
    if budget is None:
        return df.copy()
    return df[
        df["price_billion"].notna()
        & (df["price_billion"] <= budget * 1.10)
        & (df["price_billion"] >= max(0.5, budget * 0.35))
    ].copy()


def _describe_budget_shape(pool: pd.DataFrame, profile: dict) -> str | None:
    if pool.empty:
        return None
    type_counts = pool["property_type"].dropna().astype(str).value_counts() if "property_type" in pool else pd.Series(dtype=int)
    apartment_count = int(type_counts.get("apartment", 0))
    landed_count = int(type_counts.get("house", 0) + type_counts.get("townhouse", 0) + type_counts.get("villa", 0))
    median_price = pool["price_billion"].median() if "price_billion" in pool else None
    median_area = pool["area_m2"].median() if "area_m2" in pool else None
    area_text = _fmt_area(median_area)
    price_text = _fmt_money(median_price)

    if profile.get("preferred_property_type") == "house":
        if landed_count >= apartment_count:
            return (
                f"Với nhu cầu mua nhà, mặt bằng dữ liệu cho thấy phương án khả thi là nhà riêng/nhà phố "
                f"ở mức khoảng {price_text}, diện tích điển hình {area_text}; nên ưu tiên căn pháp lý rõ hơn là cố lấy vị trí quá trung tâm."
            )
        return (
            f"Nếu giữ ngân sách này và vẫn muốn nhà đất, khả năng cao phải đổi lấy vị trí xa hơn hoặc diện tích nhỏ hơn. "
            f"Nhóm căn hộ trong tầm giá có thể thực tế hơn nếu ưu tiên tiện ích, pháp lý và khả năng ở ngay."
        )
    return f"Mặt bằng ứng viên trong tầm giá có giá trung vị khoảng {price_text}, diện tích trung vị {area_text}."


def build_advisory_response(query: str, profile: dict, recs: pd.DataFrame, df: pd.DataFrame) -> str:
    budget = profile.get("budget_billion")
    pool = _filter_budget_pool(df, budget)
    safer_pool = pool[pool.get("legal_score", 0) >= 0.6] if not pool.empty and "legal_score" in pool else pool
    lines: list[str] = []

    lines.append(f"Truy vấn: {query}")
    if budget is not None:
        lines.append(f"Với tài chính khoảng {_fmt_money(budget)}, hệ thống ưu tiên các BĐS không vượt ngân sách quá nhiều, pháp lý rõ và dữ liệu đủ tin cậy.")
    elif profile.get("budget_level"):
        lines.append(f"Ngân sách được hiểu là nhóm {profile['budget_level']}; hệ thống dùng khoảng giá mờ thay vì một con số cứng.")
    else:
        lines.append("Do query chưa nêu ngân sách cụ thể, hệ thống cân bằng giữa vị trí, pháp lý, tiện ích và mức độ phù hợp mục đích.")

    if not pool.empty:
        districts = _top_counts(pool, "district")
        ptypes = _top_counts(pool, "property_type")
        median_area = pool["area_m2"].median() if "area_m2" in pool else None
        if budget is not None:
            lines.append(
                "Trong dữ liệu hiện có, nhóm phù hợp ngân sách tập trung nhiều ở "
                f"{', '.join(districts) if districts else 'một số khu vực chưa rõ'}, "
                f"loại hình phổ biến là {', '.join(ptypes) if ptypes else 'chưa rõ'}, "
                f"diện tích trung vị khoảng {_fmt_area(median_area)}."
            )
        else:
            lines.append(
                "Do chưa có ngân sách cụ thể, hệ thống đang ưu tiên các tin khớp nhu cầu chính. "
                f"Nhóm kết quả nổi bật tập trung ở {', '.join(districts) if districts else 'một số khu vực chưa rõ'}, "
                f"loại hình phổ biến là {', '.join(ptypes) if ptypes else 'chưa rõ'}, "
                f"diện tích trung vị khoảng {_fmt_area(median_area)}."
            )
        shape = _describe_budget_shape(pool, profile)
        if shape:
            lines.append(shape)
        if budget is not None and profile.get("preferred_property_type") == "house":
            house_pool = pool[pool.get("property_type", "") == "house"]
            townhouse_pool = pool[pool.get("property_type", "") == "townhouse"]
            apartment_pool = pool[pool.get("property_type", "") == "apartment"]
            if len(house_pool) + len(townhouse_pool) < max(3, len(apartment_pool)):
                lines.append(
                    "Nếu muốn đúng nghĩa nhà riêng/nhà phố, nên chấp nhận khu vực xa trung tâm hơn hoặc diện tích nhỏ hơn. "
                    "Nếu ưu tiên an toàn pháp lý và tiện ích, căn hộ trong ngân sách là phương án thực tế hơn."
                )
    else:
        lines.append("Dữ liệu hiện có không có nhiều BĐS nằm trong khoảng ngân sách này; nên nới khu vực, loại hình hoặc diện tích.")

    if not safer_pool.empty and budget is not None:
        clear_ratio = len(safer_pool) / max(len(pool), 1)
        lines.append(f"Khoảng {clear_ratio:.0%} ứng viên trong tầm giá có pháp lý từ mức tương đối rõ trở lên.")

    lines.append("Tiêu chí nên ưu tiên:")
    priorities = []
    if budget is not None:
        priorities.append("giữ giá dưới hoặc xấp xỉ ngân sách, tránh vượt quá 10-20% nếu không có lợi thế vị trí rõ")
    priorities.append("pháp lý clear_ownership hoặc ít nhất không có cảnh báo need_legal_verification")
    priorities.append("data_quality_score cao để giảm rủi ro dữ liệu thiếu")
    if profile.get("customer_group") == "family_with_children" or "school" in profile.get("desired_amenities", []):
        priorities.append("gần trường học, an ninh và diện tích đủ cho sinh hoạt gia đình")
    if profile.get("purpose") == "investment":
        priorities.append("có planning_infra, giá/m2 hợp lý và tiềm năng thanh khoản")
    if profile.get("purpose") == "rental":
        priorities.append("có rental/furnished, gần khu trung tâm hoặc khu có nhu cầu thuê")
    for item in priorities:
        lines.append(f"- {item}.")

    if not recs.empty:
        lines.append("Gợi ý nổi bật:")
        for i, (_, row) in enumerate(recs.head(3).iterrows(), start=1):
            tags = row.get("amenity_tags", [])
            tags_text = ", ".join(tags[:4]) if isinstance(tags, list) and tags else "chưa rõ tiện ích"
            facts = row.get("symbolic_facts", [])
            facts_text = ", ".join(facts[:3]) if isinstance(facts, list) and facts else "chưa có facts mạnh"
            lines.append(
                f"{i}. {row.get('property_id')} - {row.get('district')}, "
                f"{_fmt_money(row.get('price_billion'))}, {_fmt_area(row.get('area_m2'))}, "
                f"pháp lý {row.get('legal_class')}, tiện ích: {tags_text}. "
                f"Lý do: {facts_text}."
            )

    lines.append("Lưu ý: đây là tư vấn theo dữ liệu crawl và luật symbolic hiện có; trước khi quyết định thật cần kiểm tra lại pháp lý, hiện trạng và giá thị trường tại thời điểm giao dịch.")
    return "\n".join(lines)


def build_customer_friendly_advice(query: str, profile: dict, recs: pd.DataFrame) -> str:
    """Sinh lời tư vấn tự nhiên cho khách, hạn chế thuật ngữ score/rule."""
    lines: list[str] = []
    if recs.empty:
        return (
            f"Với nhu cầu '{query}', hiện dữ liệu chưa có BĐS thật sự phù hợp. "
            "Anh/chị nên nới thêm ngân sách, khu vực hoặc loại hình để hệ thống tìm được nhiều lựa chọn hơn."
        )

    budget = profile.get("budget_billion")
    bedrooms_min = profile.get("bedrooms_min")
    ptype = profile.get("preferred_property_type")

    intro_parts = [f"Với nhu cầu '{query}'"]
    if budget is not None:
        intro_parts.append(f"và ngân sách khoảng {_fmt_money(budget)}")
    if bedrooms_min is not None:
        intro_parts.append(f"ưu tiên từ {bedrooms_min} phòng ngủ")
    if ptype:
        intro_parts.append(f"loại hình {_property_type_vi(ptype)}")
    lines.append(", ".join(intro_parts) + ", em gợi ý anh/chị xem trước các lựa chọn sau:")

    if budget is None:
        lines.append(
            "Anh/chị chưa nêu ngân sách nên giá các căn đề xuất có thể dao động khá rộng; "
            "mình nên xem đây là danh sách tham khảo ban đầu, sau đó lọc lại theo tầm tài chính thực tế."
        )

    for i, (_, row) in enumerate(recs.head(3).iterrows(), start=1):
        district = row.get("district") if not pd.isna(row.get("district")) else "khu vực chưa rõ"
        ptype_text = _property_type_vi(row.get("property_type"))
        title = row.get("title")
        title_text = f" - {title}" if isinstance(title, str) and title.strip() else ""
        lines.append("")
        lines.append(f"{i}. {ptype_text.capitalize()} ở {district}{title_text}")
        lines.append(
            f"Căn này có giá khoảng {_fmt_money(row.get('price_billion'))}, "
            f"diện tích {_fmt_area(row.get('area_m2'))}, {_fmt_structure(row)}. "
            f"Pháp lý: {_legal_vi(row.get('legal_class'))}. "
            f"Tiện ích/điểm nổi bật: {_amenities_vi(row.get('amenity_tags', []))}."
        )
        strengths = _customer_strengths(row, profile)
        if strengths:
            lines.append("Điểm đáng cân nhắc: " + "; ".join(strengths) + ".")
        cautions = _customer_cautions(row)
        if cautions:
            lines.append("Cần kiểm tra thêm: " + "; ".join(cautions) + ".")
        else:
            lines.append("Trước khi quyết định, vẫn nên kiểm tra lại sổ, hiện trạng nhà, quy hoạch và khả năng thương lượng giá.")

    lines.append("")
    lines.append(
        "Nếu anh/chị cho thêm ngân sách, khu vực mong muốn và mục đích mua để ở/đầu tư/cho thuê, "
        "hệ thống sẽ lọc sát hơn và tránh đưa các căn giá quá rộng so với khả năng tài chính."
    )
    return "\n".join(lines)
