"""Cấu hình trung tâm cho dự án Neuro-Symbolic BĐS."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

# Semantic backend mặc định: PhoBERT embedding retrieval.
# Có thể tạm override bằng biến môi trường SEMANTIC_BACKEND=tfidf nếu cần chạy nhanh
# khi máy chưa cài torch/transformers.
SEMANTIC_BACKEND = "phobert"
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
PHOBERT_MAX_LENGTH = 256
PHOBERT_BATCH_SIZE = 16
PHOBERT_DEVICE = "auto"

INPUT_FILES = [
    "Crawl_data_refine.xlsx",
    "Crawl_data_batdongsan.xlsx",
    "Crawl_data_nhatot.xlsx",
    "Crawl_data.xlsx",
]

VALID_SHEETS = [
    "refined_apartments",
    "Batdongsan.com.vn",
    "nhatot.com",
]

# Ưu tiên đọc đúng sheet tương ứng với từng file để tránh nhân bản dữ liệu
INPUT_SOURCES = [
    {"file": "Crawl_data_refine.xlsx", "sheet": "refined_apartments", "source": "refine"},
    {"file": "Crawl_data_batdongsan.xlsx", "sheet": "Batdongsan.com.vn", "source": "batdongsan"},
    {"file": "Crawl_data_nhatot.xlsx", "sheet": "nhatot.com", "source": "nhatot"},
    {"file": "Crawl_data.xlsx", "sheet": "refined_apartments", "source": "crawl_data_refine"},
    {"file": "Crawl_data.xlsx", "sheet": "Batdongsan.com.vn", "source": "crawl_data_batdongsan"},
    {"file": "Crawl_data.xlsx", "sheet": "nhatot.com", "source": "crawl_data_nhatot"},
]

OUTPUT_FILES = {
    "clean_csv": "bds_clean_symbolic.csv",
    "clean_xlsx": "bds_clean_symbolic.xlsx",
    "kg_csv": "bds_kg_triples.csv",
    "rules_json": "symbolic_rule_base.json",
    "facts_json": "symbolic_fact_catalog.json",
    "risks_json": "risk_flag_catalog.json",
    "location_similarity_csv": "location_similarity_matrix.csv",
    "recommendation_sample_xlsx": "sample_recommendation_results.xlsx",
}

DEFAULT_TEST_QUERY = (
    "Tài chính 3 tỷ thì nên mua nhà như thế nào?"
)
