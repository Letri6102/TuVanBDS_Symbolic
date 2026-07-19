"""Chạy đánh giá baseline và proposed model."""
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from configs.settings import OUTPUT_DIR
from Evaluation.test_suite import TEST_QUERIES
from Evaluation.baselines import evaluate_methods
from Evaluation.metrics import summarize_metrics


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    import ast
    for col in ["amenity_tags", "symbolic_facts", "risk_flags", "triggered_rules", "data_flags", "parse_warnings", "rule_evidence"]:
        if col in df.columns:
            def parse(x):
                if isinstance(x, list): return x
                if pd.isna(x): return []
                try: return ast.literal_eval(str(x))
                except Exception: return []
            df[col] = df[col].apply(parse)
    return df


def main() -> None:
    df = pd.read_csv(OUTPUT_DIR / "bds_clean_symbolic.csv")
    df = _parse_list_columns(df)
    metrics, details = evaluate_methods(TEST_QUERIES, df, top_k=5)
    summary = summarize_metrics(metrics.to_dict("records"), k=5)
    metrics.to_csv(OUTPUT_DIR / "evaluation_metrics_by_case.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUTPUT_DIR / "evaluation_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "evaluation_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
