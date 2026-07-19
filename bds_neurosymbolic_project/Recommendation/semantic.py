"""Semantic retrieval bằng PhoBERT embeddings.

Backend mặc định dùng PhoBERT để mã hoá query và tin đăng BĐS thành vector,
sau đó xếp hạng bằng cosine similarity. TF-IDF chỉ còn là backend legacy để
debug nhanh khi môi trường chưa có torch/transformers.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.settings import (
    MODEL_CACHE_DIR,
    PHOBERT_BATCH_SIZE,
    PHOBERT_DEVICE,
    PHOBERT_MAX_LENGTH,
    PHOBERT_MODEL_NAME,
    SEMANTIC_BACKEND,
)
from Data_Preprocessing.text_utils import clean_text, norm_text


class SemanticBackendError(RuntimeError):
    """Lỗi cấu hình/runtime của semantic backend."""


def _semantic_backend() -> str:
    return os.environ.get("SEMANTIC_BACKEND", SEMANTIC_BACKEND).strip().lower()


def _model_name() -> str:
    return os.environ.get("PHOBERT_MODEL_NAME", PHOBERT_MODEL_NAME).strip()


def _device_name() -> str:
    return os.environ.get("PHOBERT_DEVICE", PHOBERT_DEVICE).strip().lower()


def _batch_size() -> int:
    return int(os.environ.get("PHOBERT_BATCH_SIZE", PHOBERT_BATCH_SIZE))


def _max_length() -> int:
    return int(os.environ.get("PHOBERT_MAX_LENGTH", PHOBERT_MAX_LENGTH))


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm_text(text))


def property_semantic_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""),
        row.get("description", ""),
        row.get("raw_location", ""),
        row.get("district", ""),
        row.get("location_zone", ""),
        row.get("property_type", ""),
        row.get("legal_class", ""),
        row.get("price_band", ""),
        row.get("area_band", ""),
        " ".join(row.get("amenity_tags", []) if isinstance(row.get("amenity_tags", []), list) else []),
        " ".join(row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []),
        " ".join(row.get("risk_flags", []) if isinstance(row.get("risk_flags", []), list) else []),
    ]
    return clean_text(" ".join(map(str, parts)))


def _segment_vietnamese(text: Any) -> str:
    text = clean_text(text)
    if not text:
        return ""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except Exception:
        return text


def _resolve_device(torch_module: Any, requested: str):
    if requested and requested != "auto":
        return torch_module.device(requested)
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


@lru_cache(maxsize=2)
def _load_phobert(model_name: str, cache_dir: str, requested_device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SemanticBackendError(
            "Thiếu dependency cho PhoBERT. Hãy cài: pip install torch transformers sentencepiece underthesea"
        ) from exc

    device = _resolve_device(torch, requested_device)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path), use_fast=False)
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_path))
    model.to(device)
    model.eval()
    return tokenizer, model, torch, str(device)


def _mean_pool(last_hidden_state: Any, attention_mask: Any, torch_module: Any):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_texts_phobert(texts: list[str]) -> np.ndarray:
    model_name = _model_name()
    tokenizer, model, torch, device_name = _load_phobert(model_name, str(MODEL_CACHE_DIR), _device_name())
    device = torch.device(device_name)
    segmented = [_segment_vietnamese(text) for text in texts]
    vectors = []

    with torch.no_grad():
        for start in range(0, len(segmented), _batch_size()):
            batch = segmented[start : start + _batch_size()]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=_max_length(),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"], torch)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy())

    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def _fit_phobert_index(df: pd.DataFrame) -> dict[str, Any]:
    texts = [property_semantic_text(row) for _, row in df.iterrows()]
    vectors = _embed_texts_phobert(texts)
    return {
        "method": "phobert_embedding_retrieval",
        "backend": "phobert",
        "model_name": _model_name(),
        "doc_count": len(texts),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and len(vectors) else 0,
        "max_length": _max_length(),
        "batch_size": _batch_size(),
        "vectors": vectors,
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _fit_tfidf_index(df: pd.DataFrame) -> dict[str, Any]:
    docs = [tokenize(property_semantic_text(row)) for _, row in df.iterrows()]
    doc_freq: Counter[str] = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))
    n_docs = len(docs)
    idf = {token: math.log((1 + n_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}
    vectors = [_tfidf_vector(tokens, idf) for tokens in docs]
    return {
        "method": "tfidf_semantic_retrieval_legacy",
        "backend": "tfidf",
        "idf": idf,
        "vectors": vectors,
        "doc_count": n_docs,
        "vocab_size": len(idf),
    }


def fit_semantic_index(df: pd.DataFrame, backend: str | None = None) -> dict[str, Any]:
    backend = (backend or _semantic_backend()).lower()
    if backend == "phobert":
        return _fit_phobert_index(df)
    if backend == "tfidf":
        return _fit_tfidf_index(df)
    raise SemanticBackendError(f"Semantic backend không hỗ trợ: {backend}")


def _cosine_dense(query_vector: np.ndarray, vectors: np.ndarray) -> list[float]:
    if vectors.size == 0 or query_vector.size == 0:
        return [0.0] * len(vectors)
    q = query_vector.reshape(-1)
    q_norm = np.linalg.norm(q)
    v_norms = np.linalg.norm(vectors, axis=1)
    denom = np.maximum(v_norms * q_norm, 1e-9)
    return ((vectors @ q) / denom).astype(float).tolist()


def semantic_similarity_scores(query: str, df: pd.DataFrame, index: dict[str, Any] | None = None) -> list[float]:
    if index is None:
        index = fit_semantic_index(df)

    backend = index.get("backend", "tfidf")
    if backend == "phobert":
        query_vector = _embed_texts_phobert([query])
        vectors = np.asarray(index["vectors"], dtype=np.float32)
        return _cosine_dense(query_vector[0], vectors)

    query_vec = _tfidf_vector(tokenize(query), index["idf"])
    return [_cosine_sparse(query_vec, vector) for vector in index["vectors"]]


def semantic_index_metadata(index: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "backend",
        "model_name",
        "doc_count",
        "embedding_dim",
        "vocab_size",
        "max_length",
        "batch_size",
    ]
    return {key: index[key] for key in keys if key in index}


def save_semantic_index(index: dict[str, Any], path: str | Path, property_ids: list[str] | None = None) -> None:
    artifact = semantic_index_metadata(index)
    if property_ids is not None:
        artifact["property_ids"] = property_ids
    if index.get("backend") == "phobert":
        artifact["vectors"] = np.asarray(index["vectors"], dtype=np.float32).tolist()
    else:
        artifact["idf"] = index.get("idf", {})
        artifact["vectors"] = index.get("vectors", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False)


def load_semantic_index(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    if artifact.get("backend") == "phobert":
        artifact["vectors"] = np.asarray(artifact.get("vectors", []), dtype=np.float32)
    return artifact
