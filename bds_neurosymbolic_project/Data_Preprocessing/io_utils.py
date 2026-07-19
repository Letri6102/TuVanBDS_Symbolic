"""Đọc dữ liệu Excel crawl và hỗ trợ lưu output."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def find_file(filename: str, roots: Iterable[str | Path]) -> Path | None:
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        direct = root / filename
        if direct.exists():
            return direct
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    return None


def read_excel_sources(input_files: list[str], valid_sheets: list[str], data_dir: str | Path) -> pd.DataFrame:
    """Đọc nhiều file Excel, chỉ lấy các sheet hợp lệ. Không thêm cột source vào output."""
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for fname in input_files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[WARN] Không tìm thấy file: {fpath}")
            continue
        xls = pd.ExcelFile(fpath)
        for sheet in xls.sheet_names:
            if sheet in valid_sheets:
                temp = pd.read_excel(fpath, sheet_name=sheet)
                frames.append(temp)
                print(f"[LOAD] {fname} | sheet={sheet} | rows={len(temp)}")

    if not frames:
        raise FileNotFoundError("Không đọc được dữ liệu nào. Kiểm tra data/raw và tên sheet.")
    return pd.concat(frames, ignore_index=True)


def read_excel_source_map(input_sources: list[dict], data_dir: str | Path) -> pd.DataFrame:
    """Đọc chính xác từng file-sheet theo cấu hình INPUT_SOURCES.

    Mỗi dòng được gắn metadata nguồn để quy về một unified dataset có thể audit.
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for cfg in input_sources:
        fname = cfg["file"]
        sheet = cfg["sheet"]
        source = cfg.get("source", sheet)
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[WARN] Không tìm thấy file: {fpath}")
            continue
        xls = pd.ExcelFile(fpath)
        if sheet not in xls.sheet_names:
            print(f"[WARN] File {fname} không có sheet {sheet}. Sheets={xls.sheet_names}")
            continue
        temp = pd.read_excel(fpath, sheet_name=sheet)
        temp.insert(0, "_source_row_id", range(1, len(temp) + 1))
        temp.insert(0, "_source_sheet", sheet)
        temp.insert(0, "_source_file", fname)
        temp.insert(0, "_source", source)
        frames.append(temp)
        print(f"[LOAD] {fname} | sheet={sheet} | rows={len(temp)}")
    if not frames:
        raise FileNotFoundError("Không đọc được dữ liệu nào. Kiểm tra data/raw và INPUT_SOURCES.")
    return pd.concat(frames, ignore_index=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, output_dir: str | Path, name_without_ext: str) -> None:
    out = ensure_dir(output_dir)
    df.to_csv(out / f"{name_without_ext}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(out / f"{name_without_ext}.xlsx", index=False)
