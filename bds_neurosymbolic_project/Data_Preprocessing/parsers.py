"""Parser cho giá, diện tích, vị trí, pháp lý và tiện ích BĐS."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from Data_Preprocessing.text_utils import clean_text, norm_text


NUMBER_RE = r"\d+(?:[\.,]\d+)*"
MAX_TOTAL_PRICE_BILLION = 10000


def vn_float(x: Any) -> float:
    if x is None:
        return np.nan
    s = re.sub(r"[^0-9,\.\-]", "", clean_text(x).replace(" ", ""))
    if not s or not re.search(r"\d", s):
        return np.nan

    sign = "-" if s.startswith("-") else ""
    s = s.replace("-", "")

    # Nếu có cả . và , thì dấu đứng sau cùng là phần thập phân.
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        # 1.400, 12.500, 1.234.567 là định dạng hàng nghìn kiểu Việt Nam.
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            s = "".join(parts)
        elif len(parts) > 2 and all(len(part) == 3 for part in parts[1:]):
            s = "".join(parts)
        elif len(parts) > 2:
            s = "".join(parts[:-1]) + "." + parts[-1]
    else:
        parts = s.split(",")
        if len(parts) > 2 and all(len(part) == 3 for part in parts[1:]):
            s = "".join(parts)
        elif len(parts) > 1:
            s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(sign + s)
    except Exception:
        return np.nan


def first_number(text: Any) -> float:
    s = clean_text(text)
    m = re.search(rf"({NUMBER_RE})", s)
    if not m:
        return np.nan
    return vn_float(m.group(1))


def all_numbers(text: Any) -> list[float]:
    s = clean_text(text)
    nums = re.findall(NUMBER_RE, s)
    return [vn_float(n) for n in nums]


def parse_area_m2(raw_area: Any, text_all: str = "") -> float:
    s = norm_text(f"{clean_text(raw_area)} {text_all}")
    if not s:
        return np.nan

    m = re.search(rf"({NUMBER_RE})\s*(m2|m²|㎡)", s)
    if m:
        val = vn_float(m.group(1))
        return val if 10 <= val <= 2000 else np.nan

    # 5x20, 5.5 x 18
    m = re.search(rf"({NUMBER_RE})\s*(?:m)?\s*[x×]\s*({NUMBER_RE})", s)
    if m:
        a, b = vn_float(m.group(1)), vn_float(m.group(2))
        area = a * b
        return area if 10 <= area <= 2000 else np.nan

    # ngang/rộng 4m dài/sâu 18m
    m = re.search(
        rf"(?:ngang|rong|rộng)\s*({NUMBER_RE})\s*m?.{{0,25}}(?:dai|dài|sau|sâu)\s*({NUMBER_RE})",
        s,
    )
    if m:
        a, b = vn_float(m.group(1)), vn_float(m.group(2))
        area = a * b
        return area if 10 <= area <= 2000 else np.nan

    # Nếu raw là số thuần
    val = first_number(raw_area)
    if 10 <= val <= 2000:
        return val
    return np.nan


def parse_area_m2_with_method(raw_area: Any, text_all: str = "") -> tuple[float, str]:
    """Parse diện tích và trả kèm nguồn parse để audit."""
    raw_clean = clean_text(raw_area)
    raw_norm = norm_text(raw_clean)
    if raw_clean and raw_norm not in {"chua xac dinh", "khong ro", "nan"}:
        val = parse_area_m2(raw_clean, "")
        if not pd.isna(val):
            return val, "raw_area"

    val = parse_area_m2("", text_all)
    if not pd.isna(val):
        return val, "text_all"
    return np.nan, "missing"


def _price_text(raw: Any) -> str:
    s = norm_text(raw)
    return s.replace("m²", "m2").replace("㎡", "m2")


def _valid_total_price(value: float) -> bool:
    return not pd.isna(value) and 0 < value <= MAX_TOTAL_PRICE_BILLION


def _tail_million_to_billion(tail: str) -> float:
    if len(tail) >= 3:
        return int(tail) / 1000.0
    return int(tail) / 100.0


def _parse_price_from_text(text: Any, area_m2: float | None = None, allow_bare_number: bool = False) -> tuple[float, str]:
    s = _price_text(text)
    if not s:
        return np.nan, "missing"

    # Tổng giá dạng "4tỉ350" hoặc "4 tỷ 350".
    m = re.search(rf"\b({NUMBER_RE})\s*(?:ty|ti)\s*(\d{{2,3}})\b", s)
    if m:
        value = vn_float(m.group(1)) + _tail_million_to_billion(m.group(2))
        if _valid_total_price(value):
            return value, "total_price_billion_million_tail"

    # Tổng giá dạng "8,9 tỷ", "13.5 tỉ".
    m = re.search(rf"\b({NUMBER_RE})\s*(?:ty|ti)\b", s)
    if m:
        value = vn_float(m.group(1))
        if _valid_total_price(value):
            return value, "total_price_billion"

    # Đơn giá/m² chỉ dùng khi không có tổng giá rõ ràng.
    m = re.search(rf"\b({NUMBER_RE})\s*(?:trieu|tr)\s*/?\s*(?:m2|m vuong)\b", s)
    if m and area_m2 is not None and not pd.isna(area_m2) and area_m2 > 0:
        value = vn_float(m.group(1)) * area_m2 / 1000.0
        if _valid_total_price(value):
            return value, "unit_price_million_per_m2"

    # Tổng giá dưới 1 tỷ dạng "850 triệu"; loại các ngữ cảnh thuê/tháng/đơn giá.
    for m in re.finditer(rf"\b({NUMBER_RE})\s*(?:trieu|tr)\b", s):
        end = m.end()
        window = s[max(0, m.start() - 10): end + 18]
        if re.search(r"/\s*(?:m2|m|thang|th)|thang|/th|m2|m vuong", window):
            continue
        million_value = vn_float(m.group(1))
        if pd.isna(million_value) or million_value < 100:
            continue
        value = million_value / 1000.0
        if _valid_total_price(value):
            return value, "total_price_million"

    if allow_bare_number and re.fullmatch(NUMBER_RE, s):
        value = first_number(s)
        if _valid_total_price(value):
            return value, "raw_numeric_billion"

    return np.nan, "missing"


def parse_price_billion(raw_price: Any, text_all: str = "", area_m2: float | None = None) -> float:
    val, _ = parse_price_billion_with_method(raw_price, text_all, area_m2)
    return val


def parse_price_billion_with_method(raw_price: Any, text_all: str = "", area_m2: float | None = None) -> tuple[float, str]:
    """Parse giá bán về đơn vị tỷ và trả kèm nguồn/kiểu parse.

    Luật ưu tiên:
    1. Cột giá gốc nếu có tổng giá hoặc số tỷ rõ ràng.
    2. Mô tả/title nếu cột giá thiếu, thỏa thuận hoặc liên hệ.
    3. Đơn giá/m² chỉ dùng khi không có tổng giá.
    """
    raw_val, raw_method = _parse_price_from_text(raw_price, area_m2, allow_bare_number=True)
    if not pd.isna(raw_val):
        return raw_val, f"raw_price:{raw_method}"

    text_val, text_method = _parse_price_from_text(text_all, area_m2, allow_bare_number=False)
    if not pd.isna(text_val):
        return text_val, f"text_all:{text_method}"

    return np.nan, "missing"


def parse_int_value(raw: Any, text_all: str = "", keywords: list[str] | None = None) -> float:
    s_raw = clean_text(raw)
    val = first_number(s_raw)
    if not pd.isna(val):
        return int(round(val))
    if keywords:
        s = norm_text(text_all)
        for kw in keywords:
            m = re.search(rf"(\d+)\s*{kw}", s)
            if m:
                return int(m.group(1))
    return np.nan


def parse_meter(raw: Any, text_all: str = "", mode: str = "generic") -> float:
    s = norm_text(f"{clean_text(raw)} {text_all}")
    if mode == "frontage":
        pats = [rf"mat tien\s*({NUMBER_RE})", rf"ngang\s*({NUMBER_RE})", rf"rong\s*({NUMBER_RE})"]
    elif mode == "road":
        pats = [rf"duong\s*(?:vao)?\s*({NUMBER_RE})", rf"hem\s*(?:xe hoi)?\s*({NUMBER_RE})"]
    else:
        pats = [rf"({NUMBER_RE})\s*m"]
    for pat in pats:
        m = re.search(pat, s)
        if m:
            val = vn_float(m.group(1))
            return val if 0 < val <= 100 else np.nan
    return np.nan


DISTRICT_CANONICAL = {
    "quan 1": "Quận 1", "q1": "Quận 1", "q.1": "Quận 1",
    "quan 2": "Quận 2", "q2": "Quận 2", "q.2": "Quận 2",
    "quan 3": "Quận 3", "q3": "Quận 3", "q.3": "Quận 3",
    "quan 4": "Quận 4", "q4": "Quận 4", "q.4": "Quận 4",
    "quan 5": "Quận 5", "q5": "Quận 5", "q.5": "Quận 5",
    "quan 6": "Quận 6", "q6": "Quận 6", "q.6": "Quận 6",
    "quan 7": "Quận 7", "q7": "Quận 7", "q.7": "Quận 7",
    "quan 8": "Quận 8", "q8": "Quận 8", "q.8": "Quận 8",
    "quan 9": "Quận 9", "q9": "Quận 9", "q.9": "Quận 9",
    "quan 10": "Quận 10", "q10": "Quận 10", "q.10": "Quận 10",
    "quan 11": "Quận 11", "q11": "Quận 11", "q.11": "Quận 11",
    "quan 12": "Quận 12", "q12": "Quận 12", "q.12": "Quận 12",
    "binh thanh": "Bình Thạnh", "phu nhuan": "Phú Nhuận", "go vap": "Gò Vấp",
    "tan binh": "Tân Bình", "tan phu": "Tân Phú", "binh tan": "Bình Tân",
    "thu duc": "Thủ Đức", "nha be": "Nhà Bè", "binh chanh": "Bình Chánh",
    "hoc mon": "Hóc Môn", "cu chi": "Củ Chi", "can gio": "Cần Giờ",
}

CENTRAL_DISTRICTS = {"Quận 1", "Quận 3", "Quận 5", "Quận 10", "Bình Thạnh", "Phú Nhuận"}
DEVELOPING_DISTRICTS = {"Thủ Đức", "Quận 2", "Quận 7", "Quận 9", "Nhà Bè", "Bình Chánh"}
SUBURBAN_DISTRICTS = {"Hóc Môn", "Củ Chi", "Cần Giờ", "Bình Tân", "Quận 12"}


def extract_district(location_text: Any) -> str:
    s = norm_text(location_text)
    m = re.search(r"\b(?:quan|q\.?)\s*(\d{1,2})\b", s)
    if m:
        return f"Quận {int(m.group(1))}"
    for key, value in sorted(DISTRICT_CANONICAL.items(), key=lambda kv: -len(kv[0])):
        if key in s:
            return value
    return "Không rõ"


def extract_ward(location_text: Any) -> str:
    s = clean_text(location_text)
    m = re.search(r"(Phường|P\.|Xã|Thị trấn)\s*([^,\-]+)", s, flags=re.IGNORECASE)
    if m:
        prefix = m.group(1).replace("P.", "Phường")
        return f"{prefix} {m.group(2).strip()}"
    return "Không rõ"


def location_zone(district: str) -> str:
    if district in CENTRAL_DISTRICTS:
        return "central"
    if district in DEVELOPING_DISTRICTS:
        return "developing"
    if district in SUBURBAN_DISTRICTS:
        return "suburban"
    if district == "Không rõ" or not district:
        return "unknown"
    return "urban"


def infer_property_type(text_all: Any) -> str:
    s = norm_text(text_all)
    if any(k in s for k in ["can ho", "chung cu", "apartment"]):
        return "apartment"
    if any(k in s for k in ["biet thu", "villa"]):
        return "villa"
    if any(k in s for k in ["dat nen", "lo dat", "ban dat"]):
        return "land"
    if any(k in s for k in ["nha pho", "mat tien", "shophouse"]):
        return "townhouse"
    if any(k in s for k in ["nha rieng", "nha hem", "hem xe hoi", "nha"]):
        return "house"
    return "unknown"


def normalize_legal(raw_legal: Any, text_all: str = "") -> str:
    s = norm_text(f"{clean_text(raw_legal)} {text_all}")
    if not s:
        return "unknown"
    if any(k in s for k in ["tranh chap", "quy hoach treo", "dinh quy hoach"]):
        return "disputed_planning"
    if any(k in s for k in ["chua co so", "chua co", "dang cho", "cho so", "dang lam so", "chua hoan cong", "phap ly chua ro"]):
        return "pending"
    if any(k in s for k in ["hop dong mua ban", "hdmb", "vi bang"]):
        return "contract_based"
    if any(k in s for k in ["so hong", "so do", "shr", "so rieng", "co so"]):
        return "clear_ownership"
    if any(k in s for k in ["phap ly", "giay to"]):
        return "other"
    return "unknown"


def normalize_legal_with_method(raw_legal: Any, text_all: str = "") -> tuple[str, str]:
    raw_class = normalize_legal(raw_legal, "")
    if raw_class != "unknown":
        return raw_class, "raw_legal"
    text_class = normalize_legal("", text_all)
    if text_class != "unknown":
        return text_class, "text_all"
    return "unknown", "missing"


def legal_score(legal_class: str) -> float:
    return {
        "clear_ownership": 1.0,
        "contract_based": 0.6,
        "pending": 0.4,
        "other": 0.5,
        "disputed_planning": 0.0,
        "unknown": 0.2,
    }.get(legal_class, 0.2)

AMENITY_KEYWORDS = {
    "school": ["truong hoc", "truong quoc te", "mam non", "dai hoc", "cho con"],
    "hospital": ["benh vien", "phong kham", "y te"],
    "market": ["gan cho", "cach cho", "cho dan sinh", "cho truyen thong", "sieu thi", "bach hoa", "winmart", "coopmart"],
    "park": ["cong vien", "cay xanh"],
    "mall": ["trung tam thuong mai", "mall", "vincom", "lotte"],
    "business": ["kinh doanh", "mat tien", "van phong", "shophouse", "mo cua hang", "cua hang"],
    "rental": ["cho thue", "hdt", "hop dong thue", "dong tien"],
    "car_access": ["xe hoi", "oto", "o to", "duong rong", "hem xe hoi"],
    "furnished": ["noi that", "full noi that", "co ban"],
    "urgent_sale": ["ban gap", "can tien", "giam gia", "cat lo"],
    "planning_infra": ["quy hoach", "ha tang", "metro", "cao toc", "san bay", "thu thiem"],
    "river_view": ["view song", "ven song", "song"],
    "security": ["an ninh", "bao ve", "camera"],
}


def extract_amenity_tags(text_all: Any) -> list[str]:
    s = norm_text(text_all)
    tags: list[str] = []
    for tag, keywords in AMENITY_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            tags.append(tag)
    return sorted(set(tags))
