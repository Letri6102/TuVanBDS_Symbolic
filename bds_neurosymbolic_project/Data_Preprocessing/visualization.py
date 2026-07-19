"""Tạo bảng thống kê và dashboard HTML đơn giản cho dữ liệu BĐS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def explode_count(df: pd.DataFrame, list_col: str, out_col_name: str) -> pd.DataFrame:
    rows = []
    if list_col not in df.columns:
        return pd.DataFrame(columns=[out_col_name, "count"])
    for _, row in df.iterrows():
        vals = row[list_col] if isinstance(row[list_col], list) else []
        for v in vals:
            rows.append({out_col_name: v})
    if not rows:
        return pd.DataFrame(columns=[out_col_name, "count"])
    return pd.DataFrame(rows)[out_col_name].value_counts().reset_index().rename(columns={"index": out_col_name, out_col_name: "count"})


def build_quality_audit(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    important_cols = [
        "property_id", "source", "source_sheet", "source_row_id", "title",
        "duplicate_count", "url_duplicate_count", "fingerprint_duplicate_count",
        "district", "location_parse_method", "location_zone", "property_type", "price_billion",
        "price_parse_method",
        "area_m2", "price_per_m2_million", "bedrooms", "bathrooms", "floors",
        "frontage_m", "road_width_m", "legal_class", "legal_parse_method", "legal_score",
        "data_quality_score", "risk_score", "parse_warnings", "amenity_tags",
        "symbolic_facts", "risk_flags", "triggered_rules",
    ]
    important_cols = [c for c in important_cols if c in df.columns]
    missing = pd.DataFrame({
        "column": important_cols,
        "missing_count": [df[c].isna().sum() for c in important_cols],
        "missing_ratio": [df[c].isna().mean() for c in important_cols],
    }).sort_values("missing_ratio", ascending=False)

    tables: dict[str, pd.DataFrame] = {"missing_summary": missing}
    for col in [
        "source", "district", "location_parse_method", "location_zone", "property_type",
        "price_parse_method", "area_parse_method", "legal_class", "legal_parse_method",
        "price_band", "area_band",
    ]:
        if col in df.columns:
            tables[f"count_{col}"] = df[col].fillna("unknown").value_counts().reset_index().rename(columns={"index": col, col: "count"})

    for list_col, out_name in [
        ("amenity_tags", "amenity_tag"),
        ("symbolic_facts", "symbolic_fact"),
        ("risk_flags", "risk_flag"),
        ("triggered_rules", "triggered_rule"),
        ("data_flags", "data_flag"),
        ("parse_warnings", "parse_warning"),
    ]:
        tables[f"count_{list_col}"] = explode_count(df, list_col, out_name)

    score_cols = [c for c in df.columns if c.endswith("_score") and pd.api.types.is_numeric_dtype(df[c])]
    if score_cols:
        tables["score_describe"] = df[score_cols].describe().T.reset_index().rename(columns={"index": "score"})
    return tables


def save_quality_audit(df: pd.DataFrame, output_dir: str | Path, filename: str = "data_quality_audit.xlsx") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    tables = build_quality_audit(df)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def build_static_dashboard_html(df: pd.DataFrame, output_dir: str | Path, filename: str = "dashboard.html") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(df)
    median_price = df["price_billion"].median() if "price_billion" in df else np.nan
    median_area = df["area_m2"].median() if "area_m2" in df else np.nan
    median_ppm = df["price_per_m2_million"].median() if "price_per_m2_million" in df else np.nan
    mean_quality = df["data_quality_score"].mean() if "data_quality_score" in df else np.nan

    def fmt(x, suffix=""):
        return "N/A" if pd.isna(x) else f"{x:,.2f}{suffix}"

    top_cols = [c for c in ["property_id", "title", "district", "price_billion", "area_m2", "legal_class", "data_quality_score", "risk_score"] if c in df.columns]
    top_html = df.sort_values("data_quality_score", ascending=False)[top_cols].head(20).to_html(index=False) if top_cols else ""
    html = f"""
<!doctype html><html><head><meta charset='utf-8'><title>BDS Symbolic Dashboard</title>
<style>body{{font-family:Arial;margin:30px;background:#f8fafc}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:white;padding:18px;border-radius:14px;box-shadow:0 2px 10px #ddd}}.label{{color:#64748b;font-size:14px}}.value{{font-size:28px;font-weight:700}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #e2e8f0;padding:8px;font-size:13px}}th{{background:#e2e8f0}}</style>
</head><body><h1>BDS Symbolic Dashboard</h1><div class='grid'>
<div class='card'><div class='label'>Tổng BĐS</div><div class='value'>{total}</div></div>
<div class='card'><div class='label'>Giá trung vị</div><div class='value'>{fmt(median_price, ' tỷ')}</div></div>
<div class='card'><div class='label'>Diện tích trung vị</div><div class='value'>{fmt(median_area, ' m²')}</div></div>
<div class='card'><div class='label'>Giá/m² trung vị</div><div class='value'>{fmt(median_ppm, ' tr/m²')}</div></div>
<div class='card'><div class='label'>Data quality TB</div><div class='value'>{fmt(mean_quality)}</div></div>
</div><h2>Top dữ liệu chất lượng cao</h2>{top_html}</body></html>
"""
    path = output_dir / filename
    path.write_text(html, encoding="utf-8")
    return path
