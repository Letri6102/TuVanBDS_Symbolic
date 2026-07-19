"""Ma trận tương đồng vị trí cho spreading activation."""
from __future__ import annotations

import pandas as pd

from Data_Preprocessing.parsers import location_zone

LOCATION_SIMILARITY_MANUAL = {
    ("Quận 3", "Quận 1"): 0.90,
    ("Quận 3", "Quận 10"): 0.85,
    ("Quận 3", "Phú Nhuận"): 0.75,
    ("Quận 1", "Bình Thạnh"): 0.75,
    ("Quận 1", "Quận 4"): 0.75,
    ("Quận 10", "Quận 5"): 0.80,
    ("Quận 10", "Quận 11"): 0.75,
    ("Bình Thạnh", "Phú Nhuận"): 0.75,
    ("Bình Thạnh", "Thủ Đức"): 0.60,
    ("Quận 2", "Thủ Đức"): 0.75,
    ("Quận 7", "Nhà Bè"): 0.75,
    ("Bình Tân", "Tân Phú"): 0.70,
    ("Tân Bình", "Phú Nhuận"): 0.75,
    ("Gò Vấp", "Phú Nhuận"): 0.65,
    ("Gò Vấp", "Tân Bình"): 0.65,
}


def district_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.2
    if a == b:
        return 1.0
    if a == "Không rõ" or b == "Không rõ":
        return 0.2
    if (a, b) in LOCATION_SIMILARITY_MANUAL:
        return LOCATION_SIMILARITY_MANUAL[(a, b)]
    if (b, a) in LOCATION_SIMILARITY_MANUAL:
        return LOCATION_SIMILARITY_MANUAL[(b, a)]
    za, zb = location_zone(a), location_zone(b)
    if za == zb:
        return 0.65
    if {za, zb} == {"central", "developing"}:
        return 0.50
    if {za, zb} == {"developing", "suburban"}:
        return 0.45
    return 0.30


def build_location_similarity_matrix(districts: list[str]) -> pd.DataFrame:
    districts = sorted(set(d for d in districts if d))
    matrix = pd.DataFrame(index=districts, columns=districts, dtype=float)
    for a in districts:
        for b in districts:
            matrix.loc[a, b] = district_similarity(a, b)
    return matrix
