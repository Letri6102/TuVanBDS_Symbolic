"""Tiện ích xử lý text tiếng Việt."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        # pandas NaN
        if x != x:
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(x)).strip()


def remove_accents(text: Any) -> str:
    text = clean_text(text)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def norm_text(text: Any) -> str:
    return remove_accents(text).lower().strip()


def contains_any(text: Any, keywords: list[str]) -> bool:
    s = norm_text(text)
    return any(k in s for k in keywords)


def compact_key(text: Any) -> str:
    s = norm_text(text)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")
