# BDS Neuro-Symbolic Recommendation Project

## 1. Cấu trúc thư mục

```text
bds_neurosymbolic_project/
├── configs/
│   └── settings.py
├── data/
│   └── raw/
│       ├── Crawl_data_batdongsan.xlsx
│       ├── Crawl_data_nhatot.xlsx
│       └── Crawl_data_refine.xlsx
├── Data_Preprocessing/
│   ├── io_utils.py
│   ├── text_utils.py
│   ├── parsers.py
│   ├── standardizer.py
│   └── visualization.py
├── Symbolic_Knowledge/
│   ├── feature_engineering.py
│   ├── rule_engine.py
│   ├── kg_builder.py
│   └── location_similarity.py
├── Recommendation/
│   ├── query_parser.py
│   ├── scoring.py
│   ├── explanation.py
│   └── recommender.py
├── Evaluation/
│   ├── test_suite.py
│   ├── baselines.py
│   └── metrics.py
├── scripts/
│   ├── run_pipeline.py
│   └── run_evaluation.py
├── outputs/
├── app_streamlit.py
└── requirements.txt
```

## 2. Chạy pipeline chính

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Output sẽ nằm trong `outputs/`:

```text
bds_clean_symbolic.csv / .xlsx
bds_kg_triples.csv
location_similarity_matrix.csv
symbolic_rule_base.json
symbolic_fact_catalog.json
risk_flag_catalog.json
data_quality_audit.xlsx
dashboard.html
sample_recommendation_results.xlsx
```

## 3. Chạy đánh giá baseline

```bash
python scripts/run_evaluation.py
```

Sinh ra:

```text
evaluation_metrics_summary.csv
evaluation_metrics_by_case.csv
evaluation_details.csv
```

## 4. Chạy giao diện demo

```bash
streamlit run app_streamlit.py
```

## 5. Ý nghĩa các module

### Data_Preprocessing
Chuẩn hóa dữ liệu crawl thô: text, giá, diện tích, vị trí, pháp lý, số phòng, mặt tiền, đường vào, tiện ích, lọc trùng và đánh giá chất lượng dữ liệu.

### Symbolic_Knowledge
Tạo symbolic features, symbolic scores, rule engine, risk flags, inferred facts, location similarity và Knowledge Graph triples.

### Recommendation
Parse câu hỏi tiếng Việt thành symbolic profile, tính fuzzy score, áp hard constraints có nới lỏng, xếp hạng Top-K và sinh giải thích.

### Evaluation
So sánh hệ thống đề xuất với baseline keyword, hard filter và TF-IDF bằng test queries heuristic.
