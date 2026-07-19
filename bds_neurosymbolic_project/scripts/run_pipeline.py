"""Chạy toàn bộ pipeline: data -> symbolic -> rules -> KG -> recommendation mẫu."""
from __future__ import annotations

import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from configs.settings import INPUT_FILES, VALID_SHEETS, INPUT_SOURCES, RAW_DATA_DIR, OUTPUT_DIR, DEFAULT_TEST_QUERY
from Data_Preprocessing.io_utils import read_excel_sources, read_excel_source_map, ensure_dir
from Data_Preprocessing.standardizer import prepare_dataset
from Data_Preprocessing.visualization import save_quality_audit, build_static_dashboard_html
from Symbolic_Knowledge.feature_engineering import add_symbolic_features
from Symbolic_Knowledge.rule_engine import apply_rule_engine, save_rule_artifacts
from Symbolic_Knowledge.kg_builder import create_triples
from Symbolic_Knowledge.location_similarity import build_location_similarity_matrix
from Recommendation.advisor import build_advisory_response, build_customer_friendly_advice
from Recommendation.llm_agent import build_agent_report
from Recommendation.recommender import recommend
from Recommendation.semantic import fit_semantic_index, save_semantic_index, semantic_index_metadata


def main() -> None:
    out = ensure_dir(OUTPUT_DIR)
    raw = read_excel_source_map(INPUT_SOURCES, RAW_DATA_DIR)
    raw_rows = len(raw)
    bds = prepare_dataset(raw)
    bds = add_symbolic_features(bds)
    bds = apply_rule_engine(bds)

    non_empty_url = bds["url_norm"].fillna("").astype(str).str.strip() != "" if "url_norm" in bds.columns else pd.Series(False, index=bds.index)
    duplicate_audit = {
        "raw_rows": raw_rows,
        "rows_after_dedup": len(bds),
        "duplicates_removed": raw_rows - len(bds),
        "remaining_duplicate_url_non_empty": int(bds.loc[non_empty_url, "url_norm"].duplicated().sum()) if "url_norm" in bds.columns else None,
        "remaining_duplicate_fingerprint": int(bds["fingerprint"].duplicated().sum()) if "fingerprint" in bds.columns else None,
        "rows_with_duplicate_count_gt1": int((bds["duplicate_count"] > 1).sum()) if "duplicate_count" in bds.columns else None,
        "max_duplicate_count": int(bds["duplicate_count"].max()) if "duplicate_count" in bds.columns and len(bds) else 0,
        "source_file_counts_after_dedup": bds["source_file"].value_counts(dropna=False).to_dict() if "source_file" in bds.columns else {},
    }
    with open(out / "duplicate_audit.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_audit, f, ensure_ascii=False, indent=2)

    bds.to_csv(out / "bds_clean_symbolic.csv", index=False, encoding="utf-8-sig")
    bds.to_excel(out / "bds_clean_symbolic.xlsx", index=False)

    kg = create_triples(bds)
    kg.to_csv(out / "bds_kg_triples.csv", index=False, encoding="utf-8-sig")

    sim = build_location_similarity_matrix(bds["district"].dropna().unique().tolist())
    sim.to_csv(out / "location_similarity_matrix.csv", encoding="utf-8-sig")

    save_rule_artifacts(out)
    audit_path = save_quality_audit(bds, out)
    dashboard_path = build_static_dashboard_html(bds, out)

    semantic_index = fit_semantic_index(bds)
    save_semantic_index(semantic_index, out / "semantic_index.json", property_ids=bds["property_id"].tolist())
    with open(out / "semantic_index_metadata.json", "w", encoding="utf-8") as f:
        json.dump(semantic_index_metadata(semantic_index), f, ensure_ascii=False, indent=2)

    profile, recs = recommend(DEFAULT_TEST_QUERY, bds, top_k=5, relax=True, semantic_index=semantic_index)
    recs.to_excel(out / "sample_recommendation_results.xlsx", index=False)
    advice = build_advisory_response(DEFAULT_TEST_QUERY, profile, recs, bds)
    with open(out / "sample_advisory_response.txt", "w", encoding="utf-8") as f:
        f.write(advice)
    customer_advice = build_customer_friendly_advice(DEFAULT_TEST_QUERY, profile, recs)
    with open(out / "sample_customer_advice.txt", "w", encoding="utf-8") as f:
        f.write(customer_advice)
    agent_report = build_agent_report(DEFAULT_TEST_QUERY, profile, recs, use_llm=False)
    with open(out / "sample_agent_report.txt", "w", encoding="utf-8") as f:
        f.write(agent_report)
    with open(out / "sample_query_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(f"Rows: {len(bds)} | Triples: {len(kg)}")
    print(f"Duplicates removed: {duplicate_audit['duplicates_removed']} | Remaining URL duplicates: {duplicate_audit['remaining_duplicate_url_non_empty']} | Remaining fingerprint duplicates: {duplicate_audit['remaining_duplicate_fingerprint']}")
    print(f"Clean data: {out / 'bds_clean_symbolic.xlsx'}")
    print(f"KG triples: {out / 'bds_kg_triples.csv'}")
    print(f"Audit: {audit_path}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Recommendation sample: {out / 'sample_recommendation_results.xlsx'}")
    print(f"Advisory sample: {out / 'sample_advisory_response.txt'}")
    print(f"Customer advice sample: {out / 'sample_customer_advice.txt'}")
    print(f"Agent report sample: {out / 'sample_agent_report.txt'}")


if __name__ == "__main__":
    main()
