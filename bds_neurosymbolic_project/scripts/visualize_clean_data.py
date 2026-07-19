"""Visualization cho dataset BĐS sau chuẩn hoá.

Chạy:
    python scripts/visualize_clean_data.py

Output:
    outputs/visualizations/clean_data_dashboard.html
"""
from __future__ import annotations

import ast
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
try:
    import plotly.express as px
    import plotly.io as pio
except ModuleNotFoundError:
    px = None
    pio = None

from configs.settings import OUTPUT_DIR


LIST_COLUMNS = [
    "amenity_tags",
    "symbolic_facts",
    "risk_flags",
    "triggered_rules",
    "data_flags",
    "parse_warnings",
    "rule_evidence",
]


def parse_list_value(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_value)
    return df


def explode_counter(df: pd.DataFrame, col: str, name: str, top_n: int = 20) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[name, "count"])
    rows = []
    for values in df[col]:
        if isinstance(values, list):
            rows.extend(values)
    if not rows:
        return pd.DataFrame(columns=[name, "count"])
    return (
        pd.Series(rows)
        .value_counts()
        .head(top_n)
        .rename_axis(name)
        .reset_index(name="count")
    )


def top_categories(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    return (
        df[col]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .rename_axis(col)
        .reset_index(name="count")
    )


def fig_to_section(title: str, fig) -> str:
    return f"<section><h2>{title}</h2>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</section>"


def table_section(title: str, table: pd.DataFrame, max_rows: int = 20) -> str:
    if table.empty:
        body = "<p>Khong co du lieu.</p>"
    else:
        body = table.head(max_rows).to_html(index=False, escape=False)
    return f"<section><h2>{title}</h2>{body}</section>"


def bar_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df.empty or "count" not in df.columns:
        return df
    max_count = max(float(df["count"].max()), 1.0)
    out = df.copy()
    out["bar"] = [
        f"<div class='barwrap'><div class='bar' style='width:{float(v) / max_count * 100:.1f}%'></div><span>{v}</span></div>"
        for v in out["count"]
    ]
    return out[[label_col, "bar"]]


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T.reset_index().rename(columns={"index": "field"})


def build_basic_dashboard(df: pd.DataFrame) -> str:
    cards = {
        "Tong BDS": len(df),
        "Gia trung vi (ty)": round(float(df["price_billion"].median()), 2) if "price_billion" in df else None,
        "Dien tich trung vi (m2)": round(float(df["area_m2"].median()), 2) if "area_m2" in df else None,
        "Gia/m2 trung vi (trieu)": round(float(df["price_per_m2_million"].median()), 2) if "price_per_m2_million" in df else None,
        "Data quality TB": round(float(df["data_quality_score"].mean()), 3) if "data_quality_score" in df else None,
    }
    cards_html = "".join(
        f"<div class='card'><div class='label'>{k}</div><div class='value'>{v if v is not None else 'N/A'}</div></div>"
        for k, v in cards.items()
    )
    sections = [
        table_section("Thong ke numeric", numeric_summary(df, [
            "price_billion",
            "area_m2",
            "price_per_m2_million",
            "legal_score",
            "data_quality_score",
            "risk_score",
            "family_suitability_score",
            "business_potential_score",
            "rental_potential_score",
            "investment_potential_score",
        ]), 50)
    ]
    for title, col in [
        ("Nguon du lieu", "source_file"),
        ("Top quan/huyen", "district"),
        ("Loai bat dong san", "property_type"),
        ("Phap ly", "legal_class"),
        ("Price band", "price_band"),
        ("Area band", "area_band"),
        ("Data quality level", "data_quality_level"),
        ("Risk level", "risk_level"),
    ]:
        counts = top_categories(df, col, 20)
        sections.append(table_section(title, bar_table(counts, col), 20))

    for title, col, name in [
        ("Tien ich/thuoc tinh", "amenity_tags", "amenity_tag"),
        ("Symbolic facts", "symbolic_facts", "symbolic_fact"),
        ("Risk flags", "risk_flags", "risk_flag"),
        ("Parse warnings", "parse_warnings", "parse_warning"),
    ]:
        counts = explode_counter(df, col, name, 20)
        sections.append(table_section(title, bar_table(counts, name), 20))

    return base_html(cards_html, "".join(sections), "Clean BDS Data Dashboard - Basic")


def base_html(cards_html: str, sections_html: str, title: str) -> str:
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f6f8fb; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #64748b; margin-bottom: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin-bottom: 28px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .label {{ color: #64748b; font-size: 13px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .barwrap {{ display: grid; grid-template-columns: 1fr 64px; align-items: center; gap: 8px; }}
    .bar {{ height: 14px; background: #2563eb; border-radius: 4px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">Visualization cho dataset sau chuan hoa, dedup, symbolic reasoning va semantic training.</div>
  <div class="cards">{cards_html}</div>
  {sections_html}
</body>
</html>
"""


def build_dashboard(df: pd.DataFrame) -> str:
    if px is None or pio is None:
        return build_basic_dashboard(df)

    sections: list[str] = []

    cards = {
        "Tong BDS": len(df),
        "Gia trung vi (ty)": round(float(df["price_billion"].median()), 2) if "price_billion" in df else None,
        "Dien tich trung vi (m2)": round(float(df["area_m2"].median()), 2) if "area_m2" in df else None,
        "Gia/m2 trung vi (trieu)": round(float(df["price_per_m2_million"].median()), 2) if "price_per_m2_million" in df else None,
        "Data quality TB": round(float(df["data_quality_score"].mean()), 3) if "data_quality_score" in df else None,
    }
    cards_html = "".join(
        f"<div class='card'><div class='label'>{k}</div><div class='value'>{v if v is not None else 'N/A'}</div></div>"
        for k, v in cards.items()
    )

    if "source_file" in df.columns:
        fig = px.bar(top_categories(df, "source_file"), x="source_file", y="count", title="So dong theo file nguon")
        sections.append(fig_to_section("Nguon du lieu", fig))

    if "district" in df.columns:
        fig = px.bar(top_categories(df, "district", 20), x="district", y="count", title="Top quan/huyen")
        fig.update_layout(xaxis_tickangle=-35)
        sections.append(fig_to_section("Phan bo theo quan/huyen", fig))

    if "property_type" in df.columns:
        fig = px.bar(top_categories(df, "property_type"), x="property_type", y="count", title="Loai BDS")
        sections.append(fig_to_section("Loai bat dong san", fig))

    if "legal_class" in df.columns:
        fig = px.bar(top_categories(df, "legal_class"), x="legal_class", y="count", title="Nhom phap ly")
        sections.append(fig_to_section("Phap ly", fig))

    if "price_billion" in df.columns:
        price_df = df[df["price_billion"].notna() & (df["price_billion"] <= df["price_billion"].quantile(0.98))]
        fig = px.histogram(price_df, x="price_billion", nbins=40, title="Phan bo gia sau khi loai outlier 2% tren")
        sections.append(fig_to_section("Phan bo gia", fig))

    if "area_m2" in df.columns:
        area_df = df[df["area_m2"].notna() & (df["area_m2"] <= df["area_m2"].quantile(0.98))]
        fig = px.histogram(area_df, x="area_m2", nbins=40, title="Phan bo dien tich sau khi loai outlier 2% tren")
        sections.append(fig_to_section("Phan bo dien tich", fig))

    if {"area_m2", "price_billion"}.issubset(df.columns):
        scatter_df = df[
            df["area_m2"].notna()
            & df["price_billion"].notna()
            & (df["area_m2"] <= df["area_m2"].quantile(0.98))
            & (df["price_billion"] <= df["price_billion"].quantile(0.98))
        ].copy()
        hover_cols = [c for c in ["property_id", "title", "district", "legal_class", "data_quality_score"] if c in scatter_df.columns]
        fig = px.scatter(
            scatter_df,
            x="area_m2",
            y="price_billion",
            color="legal_class" if "legal_class" in scatter_df.columns else None,
            hover_data=hover_cols,
            title="Gia theo dien tich",
        )
        sections.append(fig_to_section("Tuong quan gia - dien tich", fig))

    if {"district", "price_billion"}.issubset(df.columns):
        top_districts = top_categories(df, "district", 10)["district"].tolist()
        box_df = df[df["district"].isin(top_districts) & df["price_billion"].notna()].copy()
        box_df = box_df[box_df["price_billion"] <= box_df["price_billion"].quantile(0.98)]
        fig = px.box(box_df, x="district", y="price_billion", title="Gia theo top quan/huyen")
        fig.update_layout(xaxis_tickangle=-35)
        sections.append(fig_to_section("So sanh gia theo khu vuc", fig))

    if "amenity_tags" in df.columns:
        fig = px.bar(explode_counter(df, "amenity_tags", "amenity_tag"), x="amenity_tag", y="count", title="Top tien ich/thuoc tinh")
        fig.update_layout(xaxis_tickangle=-35)
        sections.append(fig_to_section("Tien ich", fig))

    if "symbolic_facts" in df.columns:
        fig = px.bar(explode_counter(df, "symbolic_facts", "symbolic_fact"), x="symbolic_fact", y="count", title="Top symbolic facts")
        fig.update_layout(xaxis_tickangle=-35)
        sections.append(fig_to_section("Symbolic facts", fig))

    if "risk_flags" in df.columns:
        fig = px.bar(explode_counter(df, "risk_flags", "risk_flag"), x="risk_flag", y="count", title="Top risk flags")
        fig.update_layout(xaxis_tickangle=-35)
        sections.append(fig_to_section("Risk flags", fig))

    if "data_quality_score" in df.columns:
        fig = px.histogram(df, x="data_quality_score", nbins=20, title="Phan bo data quality score")
        sections.append(fig_to_section("Data quality", fig))

    return base_html(cards_html, "".join(sections), "Clean BDS Data Dashboard")


def main() -> None:
    data_path = OUTPUT_DIR / "bds_clean_symbolic.csv"
    if not data_path.exists():
        raise FileNotFoundError("Chua co outputs/bds_clean_symbolic.csv. Hay chay scripts/run_pipeline.py truoc.")
    out_dir = OUTPUT_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean_data(data_path)
    html = build_dashboard(df)
    out_path = out_dir / "clean_data_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
