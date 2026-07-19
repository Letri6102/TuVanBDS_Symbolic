"""Chuẩn hóa dữ liệu crawl BĐS thành bảng sạch."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from Data_Preprocessing.text_utils import clean_text, compact_key, norm_text
from Data_Preprocessing.parsers import (
    extract_amenity_tags,
    extract_district,
    extract_ward,
    infer_property_type,
    legal_score,
    location_zone,
    MAX_TOTAL_PRICE_BILLION,
    normalize_legal_with_method,
    parse_area_m2_with_method,
    parse_int_value,
    parse_meter,
    parse_price_billion_with_method,
)


def get_first_existing_col(df: pd.DataFrame, candidates: list[str], default: Any = "") -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def infer_district_with_method(raw_location: Any, text_all: Any) -> tuple[str, str]:
    district = extract_district(raw_location)
    if district != "Không rõ":
        return district, "raw_location"
    district = extract_district(text_all)
    if district != "Không rõ":
        return district, "text_all"
    return "Không rõ", "missing"


def infer_ward_with_method(raw_location: Any, text_all: Any) -> tuple[str, str]:
    ward = extract_ward(raw_location)
    if ward != "Không rõ":
        return ward, "raw_location"
    ward = extract_ward(text_all)
    if ward != "Không rõ":
        return ward, "text_all"
    return "Không rõ", "missing"


def build_parse_warnings(row: pd.Series) -> list[str]:
    warnings: list[str] = []
    for field in ["price", "area", "location", "legal"]:
        method = str(row.get(f"{field}_parse_method", ""))
        if method == "missing":
            warnings.append(f"missing_{field}")
        elif method.startswith("text_all"):
            warnings.append(f"{field}_inferred_from_text")
    if row.get("legal_class") == "disputed_planning":
        warnings.append("legal_disputed_or_planning")
    return warnings


def standardize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    out["source"] = get_first_existing_col(raw, ["_source"], "unknown")
    out["source_file"] = get_first_existing_col(raw, ["_source_file"], "")
    out["source_sheet"] = get_first_existing_col(raw, ["_source_sheet"], "")
    out["source_row_id"] = get_first_existing_col(raw, ["_source_row_id"], np.nan)

    out["title"] = get_first_existing_col(raw, ["Tieu_De", "title", "Title", "TieuDe"])
    out["description"] = get_first_existing_col(raw, ["Mo_Ta", "description", "Description", "MoTa"])
    out["url"] = get_first_existing_col(raw, ["URL", "url", "Link", "link"])
    out["raw_location"] = get_first_existing_col(raw, ["Vi_Tri", "location", "Dia_Chi", "address"])
    out["raw_area"] = get_first_existing_col(raw, ["Dien_Tich", "area", "DienTich"])
    out["raw_price"] = get_first_existing_col(raw, ["Gia", "price", "Gia_Ban"])
    out["raw_amenities"] = get_first_existing_col(raw, ["Tien_Ich", "Tien_Iich", "amenities", "TienIch"])
    out["raw_bedrooms"] = get_first_existing_col(raw, ["So_PN", "bedrooms", "PN"])
    out["raw_bathrooms"] = get_first_existing_col(raw, ["So_PT", "bathrooms", "WC"])
    out["raw_floors"] = get_first_existing_col(raw, ["So_Tang", "floors", "Tang"])
    out["raw_frontage"] = get_first_existing_col(raw, ["Mat_Tien", "frontage", "Ngang"])
    out["raw_road_width"] = get_first_existing_col(raw, ["Duong_Vao", "road_width", "Duong"])
    out["raw_legal"] = get_first_existing_col(raw, ["Phap_Ly", "legal", "PhapLy"])
    out["raw_furniture"] = get_first_existing_col(raw, ["Noi_That", "furniture", "NoiThat"])

    out["text_all"] = (
        out["title"].fillna("").astype(str) + " "
        + out["description"].fillna("").astype(str) + " "
        + out["raw_amenities"].fillna("").astype(str) + " "
        + out["raw_location"].fillna("").astype(str) + " "
        + out["raw_furniture"].fillna("").astype(str)
    )

    area_results = [parse_area_m2_with_method(a, t) for a, t in zip(out["raw_area"], out["text_all"])]
    out["area_m2"] = [v for v, _ in area_results]
    out["area_parse_method"] = [m for _, m in area_results]

    price_results = [
        parse_price_billion_with_method(p, t, a)
        for p, t, a in zip(out["raw_price"], out["text_all"], out["area_m2"])
    ]
    out["price_billion"] = [v for v, _ in price_results]
    out["price_parse_method"] = [m for _, m in price_results]
    out["price_per_m2_million"] = np.where(
        (out["area_m2"].notna()) & (out["area_m2"] > 0) & out["price_billion"].notna(),
        out["price_billion"] * 1000 / out["area_m2"],
        np.nan,
    )

    out["bedrooms"] = [parse_int_value(v, t, ["pn", "phong ngu"]) for v, t in zip(out["raw_bedrooms"], out["text_all"])]
    out["bathrooms"] = [parse_int_value(v, t, ["wc", "phong tam", "toilet"]) for v, t in zip(out["raw_bathrooms"], out["text_all"])]
    out["floors"] = [parse_int_value(v, t, ["tang"]) for v, t in zip(out["raw_floors"], out["text_all"])]
    out["frontage_m"] = [parse_meter(v, t, "frontage") for v, t in zip(out["raw_frontage"], out["text_all"])]
    out["road_width_m"] = [parse_meter(v, t, "road") for v, t in zip(out["raw_road_width"], out["text_all"])]

    district_results = [infer_district_with_method(l, t) for l, t in zip(out["raw_location"], out["text_all"])]
    out["district"] = [v for v, _ in district_results]
    out["location_parse_method"] = [m for _, m in district_results]
    ward_results = [infer_ward_with_method(l, t) for l, t in zip(out["raw_location"], out["text_all"])]
    out["ward"] = [v for v, _ in ward_results]
    out["ward_parse_method"] = [m for _, m in ward_results]
    out["location_zone"] = out["district"].apply(location_zone)
    out["property_type"] = out["text_all"].apply(infer_property_type)
    legal_results = [normalize_legal_with_method(l, t) for l, t in zip(out["raw_legal"], out["text_all"])]
    out["legal_class"] = [v for v, _ in legal_results]
    out["legal_parse_method"] = [m for _, m in legal_results]
    out["legal_score"] = out["legal_class"].apply(legal_score)
    out["amenity_tags"] = out["text_all"].apply(extract_amenity_tags)
    out["parse_warnings"] = out.apply(build_parse_warnings, axis=1)

    return out


def make_fingerprint(row: pd.Series) -> str:
    key = "|".join(
        [
            compact_key(row.get("title", ""))[:80],
            compact_key(row.get("raw_location", ""))[:80],
            compact_key(row.get("text_all", ""))[:120],
            str(row.get("price_billion", "")),
            str(row.get("area_m2", "")),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def validate_numeric_range(row: pd.Series) -> list[str]:
    flags: list[str] = []
    price = row.get("price_billion", np.nan)
    area = row.get("area_m2", np.nan)
    ppm = row.get("price_per_m2_million", np.nan)

    if pd.isna(price):
        flags.append("missing_price")
    elif price <= 0 or price > MAX_TOTAL_PRICE_BILLION:
        flags.append("abnormal_price")

    if pd.isna(area):
        flags.append("missing_area")
    elif area < 10 or area > 2000:
        flags.append("abnormal_area")

    if pd.isna(ppm):
        flags.append("missing_price_per_m2")
    elif ppm < 5 or ppm > 1000:
        flags.append("abnormal_price_per_m2")

    if row.get("district", "Không rõ") == "Không rõ":
        flags.append("missing_district")
    if row.get("legal_class", "unknown") == "unknown":
        flags.append("missing_legal")
    if row.get("property_type", "unknown") == "unknown":
        flags.append("unknown_property_type")
    return flags


def data_quality_score(flags: list[str]) -> float:
    penalties = {
        "missing_price": 0.25,
        "abnormal_price": 0.30,
        "missing_area": 0.20,
        "abnormal_area": 0.25,
        "missing_price_per_m2": 0.10,
        "abnormal_price_per_m2": 0.20,
        "missing_district": 0.20,
        "missing_legal": 0.15,
        "unknown_property_type": 0.05,
    }
    score = 1.0 - sum(penalties.get(f, 0.05) for f in flags)
    return max(0.0, min(1.0, score))


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["url_norm"] = df["url"].fillna("").astype(str).str.strip()
    df["fingerprint"] = df.apply(make_fingerprint, axis=1)
    df["url_duplicate_count"] = 1
    has_url_mask = df["url_norm"] != ""
    df.loc[has_url_mask, "url_duplicate_count"] = df.loc[has_url_mask].groupby("url_norm")["url_norm"].transform("size")
    df["fingerprint_duplicate_count"] = df.groupby("fingerprint")["fingerprint"].transform("size")
    df["duplicate_count"] = np.where(
        has_url_mask,
        df["url_duplicate_count"],
        df["fingerprint_duplicate_count"],
    )

    has_url = df[df["url_norm"] != ""].drop_duplicates(subset=["url_norm"], keep="first")
    no_url = df[df["url_norm"] == ""].drop_duplicates(subset=["fingerprint"], keep="first")
    df = pd.concat([has_url, no_url], ignore_index=True)
    df = df.drop_duplicates(subset=["fingerprint"], keep="first").reset_index(drop=True)

    df["property_id"] = [f"BDS_{i+1:05d}" for i in range(len(df))]
    df["data_flags"] = df.apply(validate_numeric_range, axis=1)
    df["data_quality_score"] = df["data_flags"].apply(data_quality_score)

    # Không giữ url_norm/fingerprint như output chính nếu không cần.
    return df


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    std = standardize_dataframe(raw)
    return clean_and_deduplicate(std)
