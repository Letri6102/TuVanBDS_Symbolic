"""Demo Streamlit tối giản cho hệ thống tư vấn BĐS Neuro-Symbolic."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from Recommendation.advisor import build_advisory_response, build_customer_friendly_advice
from Recommendation.llm_agent import build_agent_report
from Recommendation.recommender import recommend
from Recommendation.semantic import SemanticBackendError, fit_semantic_index, load_semantic_index
from configs.settings import DEFAULT_TEST_QUERY, OUTPUT_DIR


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["amenity_tags", "symbolic_facts", "risk_flags", "triggered_rules", "data_flags", "parse_warnings", "rule_evidence"]:
        if col in df.columns:
            def parse(x):
                if isinstance(x, list): return x
                if pd.isna(x): return []
                try: return ast.literal_eval(str(x))
                except Exception: return []
            df[col] = df[col].apply(parse)
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    path = OUTPUT_DIR / "bds_clean_symbolic.csv"
    if not path.exists():
        st.error("Chưa có outputs/bds_clean_symbolic.csv. Hãy chạy: python scripts/run_pipeline.py")
        return pd.DataFrame()
    return parse_list_columns(pd.read_csv(path))


@st.cache_resource
def load_trained_semantic_index(index_mtime_ns: int) -> dict:
    return load_semantic_index(OUTPUT_DIR / "semantic_index.json")


st.set_page_config(page_title="BDS Neuro-Symbolic Advisor", layout="wide")
st.title("BDS Neuro-Symbolic Advisor")

df = load_data()
query = st.text_input("Nhập nhu cầu BĐS", DEFAULT_TEST_QUERY)
top_k = st.slider("Top K", 3, 20, 5)
use_llm_agent = st.toggle("Dùng LLM agent nếu đã cấu hình API", value=False)

if st.button("Tư vấn") and not df.empty:
    semantic_index = None
    index_path = OUTPUT_DIR / "semantic_index.json"
    try:
        if index_path.exists():
            semantic_index = load_trained_semantic_index(index_path.stat().st_mtime_ns)
            expected_ids = semantic_index.get("property_ids")
            if expected_ids and "property_id" in df.columns and expected_ids != df["property_id"].tolist():
                st.warning("Semantic index không khớp dataset hiện tại, đang fit lại bằng PhoBERT.")
                semantic_index = fit_semantic_index(df)
        else:
            st.info("Chưa có semantic_index.json, đang fit PhoBERT index từ dataset hiện tại.")
            semantic_index = fit_semantic_index(df)
        profile, recs = recommend(query, df, top_k=top_k, relax=True, semantic_index=semantic_index)
    except SemanticBackendError as exc:
        st.error(str(exc))
        st.stop()
    st.subheader("Symbolic profile")
    st.json(profile)
    st.subheader("Gợi ý cho khách hàng")
    st.markdown(build_customer_friendly_advice(query, profile, recs).replace("\n", "\n\n"))
    st.subheader("Tư vấn tổng quan")
    st.markdown(build_advisory_response(query, profile, recs, df).replace("\n", "\n\n"))
    st.subheader("Bài phân tích Top 3")
    st.markdown(build_agent_report(query, profile, recs, use_llm=use_llm_agent).replace("\n", "\n\n"))
    st.subheader("Kết quả đề xuất")
    show_cols = [
        c for c in [
            "property_id", "source", "title", "district", "price_billion", "area_m2",
            "bedrooms", "bathrooms", "floors", "legal_class", "legal_score", "amenity_tags", "final_score", "symbolic_score", "semantic_score", "match_type",
            "semantic_backend", "activation_score", "symbolic_facts", "risk_flags", "explanation",
        ]
        if c in recs.columns
    ]
    st.dataframe(recs[show_cols], use_container_width=True)
