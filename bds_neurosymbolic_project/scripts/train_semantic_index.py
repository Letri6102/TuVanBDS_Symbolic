"""Train semantic index từ dataset sạch hiện có.

Mặc định dùng backend PhoBERT theo configs/settings.py. Có thể tạm chạy nhanh
bằng backend legacy: SEMANTIC_BACKEND=tfidf python scripts/train_semantic_index.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from configs.settings import OUTPUT_DIR
from Recommendation.semantic import fit_semantic_index, save_semantic_index, semantic_index_metadata


LIST_COLUMNS = [
    "amenity_tags",
    "symbolic_facts",
    "risk_flags",
    "triggered_rules",
    "data_flags",
    "parse_warnings",
    "rule_evidence",
]


def _parse_list_value(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _load_clean_dataset() -> pd.DataFrame:
    path = OUTPUT_DIR / "bds_clean_symbolic.csv"
    if not path.exists():
        raise FileNotFoundError("Chưa có outputs/bds_clean_symbolic.csv. Hãy chạy scripts/run_pipeline.py trước.")
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_value)
    return df


def main() -> None:
    out = OUTPUT_DIR
    df = _load_clean_dataset()
    semantic_index = fit_semantic_index(df)
    save_semantic_index(semantic_index, out / "semantic_index.json", property_ids=df["property_id"].tolist())
    metadata = semantic_index_metadata(semantic_index)
    with open(out / "semantic_index_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("DONE")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Semantic index: {out / 'semantic_index.json'}")


if __name__ == "__main__":
    main()
