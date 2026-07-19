"""Visualize a few Knowledge Graph samples as static HTML/SVG.

Usage:
    python scripts/visualize_kg_samples.py
    python scripts/visualize_kg_samples.py --top-n 5
    python scripts/visualize_kg_samples.py --property-ids BDS_00134 BDS_00202

Output:
    outputs/visualizations/kg_sample_graph.html
    outputs/visualizations/kg_sample_triples.csv
"""
from __future__ import annotations

import argparse
import ast
import html
import math
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from configs.settings import OUTPUT_DIR


IMPORTANT_PREDICATES = [
    "locatedInDistrict",
    "locatedInWard",
    "locatedInZone",
    "hasPropertyType",
    "hasLegalClass",
    "hasPriceBand",
    "hasAreaBand",
    "hasBedroomsBand",
    "hasRoadBand",
    "hasFrontageBand",
    "hasPriceBillion",
    "hasAreaM2",
    "hasPricePerM2Million",
    "hasLegalScore",
    "hasDataQualityScore",
    "hasFamilySuitabilityScore",
    "hasBusinessPotentialScore",
    "hasRentalPotentialScore",
    "hasInvestmentPotentialScore",
    "hasRiskScore",
    "hasAmenityTag",
    "inferredFact",
    "hasRiskFlag",
    "triggeredRule",
]


PREDICATE_LABELS = {
    "locatedInDistrict": "district",
    "locatedInWard": "ward",
    "locatedInZone": "zone",
    "hasPropertyType": "type",
    "hasLegalClass": "legal",
    "hasPriceBand": "price band",
    "hasAreaBand": "area band",
    "hasBedroomsBand": "bedrooms",
    "hasRoadBand": "road",
    "hasFrontageBand": "frontage",
    "hasPriceBillion": "price",
    "hasAreaM2": "area",
    "hasPricePerM2Million": "price/m2",
    "hasLegalScore": "legal score",
    "hasDataQualityScore": "data quality",
    "hasFamilySuitabilityScore": "family score",
    "hasBusinessPotentialScore": "business score",
    "hasRentalPotentialScore": "rental score",
    "hasInvestmentPotentialScore": "investment score",
    "hasRiskScore": "risk score",
    "hasAmenityTag": "amenity",
    "inferredFact": "fact",
    "hasRiskFlag": "risk",
    "triggeredRule": "rule",
}


PREDICATE_COLORS = {
    "locatedInDistrict": "#2563eb",
    "locatedInWard": "#2563eb",
    "locatedInZone": "#2563eb",
    "hasPropertyType": "#7c3aed",
    "hasLegalClass": "#16a34a",
    "hasPriceBand": "#ca8a04",
    "hasAreaBand": "#ca8a04",
    "hasBedroomsBand": "#ca8a04",
    "hasRoadBand": "#ca8a04",
    "hasFrontageBand": "#ca8a04",
    "hasPriceBillion": "#ea580c",
    "hasAreaM2": "#ea580c",
    "hasPricePerM2Million": "#ea580c",
    "hasLegalScore": "#16a34a",
    "hasDataQualityScore": "#0891b2",
    "hasFamilySuitabilityScore": "#0d9488",
    "hasBusinessPotentialScore": "#0d9488",
    "hasRentalPotentialScore": "#0d9488",
    "hasInvestmentPotentialScore": "#0d9488",
    "hasRiskScore": "#dc2626",
    "hasAmenityTag": "#4f46e5",
    "inferredFact": "#15803d",
    "hasRiskFlag": "#dc2626",
    "triggeredRule": "#64748b",
}


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def short_text(value, max_len: int = 34) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def fmt_object(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.2f}"
    text = str(value)
    try:
        num = float(text)
        if abs(num) >= 100:
            return f"{num:.1f}"
        return f"{num:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return text


def choose_property_ids(clean_df: pd.DataFrame, kg: pd.DataFrame, top_n: int) -> list[str]:
    if "property_id" not in clean_df.columns:
        return kg["subject"].drop_duplicates().head(top_n).tolist()

    df = clean_df.copy()
    for col in ["symbolic_facts", "risk_flags", "amenity_tags", "triggered_rules"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    df["kg_visual_score"] = (
        df["symbolic_facts"].apply(len) * 3
        + df["risk_flags"].apply(len) * 2
        + df["amenity_tags"].apply(len)
        + df["triggered_rules"].apply(len)
        + df.get("data_quality_score", 0).fillna(0)
    )
    return df.sort_values("kg_visual_score", ascending=False)["property_id"].head(top_n).tolist()


def filter_triples_for_property(kg: pd.DataFrame, property_id: str, max_triples: int) -> pd.DataFrame:
    sub = kg[(kg["subject"] == property_id) & (kg["predicate"].isin(IMPORTANT_PREDICATES))].copy()
    sub["predicate_order"] = sub["predicate"].apply(lambda p: IMPORTANT_PREDICATES.index(p) if p in IMPORTANT_PREDICATES else 999)
    sub = sub.sort_values(["predicate_order", "predicate", "object"]).drop(columns=["predicate_order"])
    return sub.head(max_triples)


def property_title(clean_df: pd.DataFrame, property_id: str) -> str:
    if "property_id" not in clean_df.columns or "title" not in clean_df.columns:
        return property_id
    rows = clean_df.loc[clean_df["property_id"] == property_id, "title"]
    if rows.empty or pd.isna(rows.iloc[0]):
        return property_id
    return str(rows.iloc[0])


def svg_for_property(property_id: str, title: str, triples: pd.DataFrame) -> str:
    width, height = 1180, 760
    cx, cy = width / 2, height / 2
    radius_x, radius_y = 450, 250
    rows = triples.to_dict("records")
    n = max(len(rows), 1)
    node_parts = []
    edge_parts = []

    center_label = html.escape(property_id)
    center_title = html.escape(short_text(title, 70))
    node_parts.append(
        f"""
        <circle cx="{cx}" cy="{cy}" r="58" fill="#111827"></circle>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="center-text">{center_label}</text>
        <text x="{cx}" y="{cy + 16}" text-anchor="middle" class="center-subtitle">{center_title}</text>
        """
    )

    for i, row in enumerate(rows):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + radius_x * math.cos(angle)
        y = cy + radius_y * math.sin(angle)
        pred = str(row["predicate"])
        obj = fmt_object(row["object"])
        color = PREDICATE_COLORS.get(pred, "#475569")
        label = html.escape(short_text(obj, 30))
        pred_label = html.escape(PREDICATE_LABELS.get(pred, pred))
        edge_label_x = cx + (x - cx) * 0.54
        edge_label_y = cy + (y - cy) * 0.54

        edge_parts.append(
            f"""
            <line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="{color}" stroke-width="1.5" opacity="0.62"></line>
            <text x="{edge_label_x}" y="{edge_label_y}" text-anchor="middle" class="edge-label" fill="{color}">{pred_label}</text>
            """
        )
        node_parts.append(
            f"""
            <circle cx="{x}" cy="{y}" r="42" fill="white" stroke="{color}" stroke-width="2"></circle>
            <text x="{x}" y="{y - 4}" text-anchor="middle" class="node-text">{label}</text>
            <text x="{x}" y="{y + 15}" text-anchor="middle" class="node-type" fill="{color}">{pred_label}</text>
            """
        )

    return f"""
    <section class="graph-card">
      <h2>{html.escape(property_id)} - {html.escape(short_text(title, 110))}</h2>
      <svg viewBox="0 0 {width} {height}" role="img">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#f8fafc"></rect>
        {''.join(edge_parts)}
        {''.join(node_parts)}
      </svg>
      <details>
        <summary>Triples trong subgraph</summary>
        {triples.to_html(index=False, escape=True)}
      </details>
    </section>
    """


def build_html(clean_df: pd.DataFrame, kg: pd.DataFrame, property_ids: list[str], max_triples: int) -> tuple[str, pd.DataFrame]:
    sample_triples = []
    sections = []
    for pid in property_ids:
        triples = filter_triples_for_property(kg, pid, max_triples=max_triples)
        if triples.empty:
            continue
        sample_triples.append(triples)
        sections.append(svg_for_property(pid, property_title(clean_df, pid), triples))

    sample_df = pd.concat(sample_triples, ignore_index=True) if sample_triples else pd.DataFrame(columns=kg.columns)
    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>KG Sample Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #eef2f7; color: #111827; }}
    h1 {{ margin-bottom: 6px; }}
    .subtitle {{ color: #64748b; margin-bottom: 24px; }}
    .graph-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px; margin-bottom: 22px; }}
    svg {{ width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 10px; }}
    .center-text {{ fill: white; font-size: 18px; font-weight: 700; }}
    .center-subtitle {{ fill: #d1d5db; font-size: 11px; }}
    .node-text {{ fill: #111827; font-size: 12px; font-weight: 700; }}
    .node-type {{ font-size: 10px; font-weight: 700; }}
    .edge-label {{ font-size: 10px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Knowledge Graph Sample Visualization</h1>
  <div class="subtitle">
    Moi graph hien thi mot BDS o trung tam va cac triples quan trong xung quanh: vi tri, loai hinh,
    phap ly, gia/dien tich, facts, risks va rules.
  </div>
  {''.join(sections)}
</body>
</html>
"""
    return html_doc, sample_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-triples", type=int, default=28)
    parser.add_argument("--property-ids", nargs="*", default=None)
    args = parser.parse_args()

    kg_path = OUTPUT_DIR / "bds_kg_triples.csv"
    clean_path = OUTPUT_DIR / "bds_clean_symbolic.csv"
    if not kg_path.exists():
        raise FileNotFoundError("Missing outputs/bds_kg_triples.csv. Run scripts/run_pipeline.py first.")
    if not clean_path.exists():
        raise FileNotFoundError("Missing outputs/bds_clean_symbolic.csv. Run scripts/run_pipeline.py first.")

    kg = pd.read_csv(kg_path)
    clean_df = pd.read_csv(clean_path)
    property_ids = args.property_ids or choose_property_ids(clean_df, kg, args.top_n)

    out_dir = OUTPUT_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_doc, sample_triples = build_html(clean_df, kg, property_ids, args.max_triples)
    html_path = out_dir / "kg_sample_graph.html"
    csv_path = out_dir / "kg_sample_triples.csv"
    html_path.write_text(html_doc, encoding="utf-8")
    sample_triples.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"DONE: {html_path}")
    print(f"Sample triples: {csv_path}")
    print(f"Property IDs: {', '.join(property_ids)}")


if __name__ == "__main__":
    main()
