"""Regression tests for Vietnamese real-estate parsers."""
from __future__ import annotations

import math
import unittest

import pandas as pd

from Data_Preprocessing.parsers import (
    parse_area_m2_with_method,
    parse_price_billion_with_method,
    vn_float,
)
from Data_Preprocessing.standardizer import clean_and_deduplicate, standardize_dataframe
from Recommendation.query_parser import parse_user_query


class VietnameseNumberParserTest(unittest.TestCase):
    def assertFloatEqual(self, actual: float, expected: float) -> None:
        self.assertFalse(math.isnan(actual))
        self.assertAlmostEqual(actual, expected)

    def test_dot_thousand_separator_is_not_lost(self) -> None:
        self.assertFloatEqual(vn_float("1.400"), 1400.0)
        self.assertFloatEqual(vn_float("1.400,5"), 1400.5)
        self.assertFloatEqual(vn_float("1.234.567"), 1234567.0)

    def test_decimal_prices_still_parse_as_decimals(self) -> None:
        self.assertFloatEqual(vn_float("4.35"), 4.35)
        self.assertFloatEqual(vn_float("4,35"), 4.35)
        self.assertFloatEqual(vn_float("108.5"), 108.5)

    def test_price_parser_keeps_large_billion_prices(self) -> None:
        value, method = parse_price_billion_with_method("1.400 tỷ")
        self.assertFloatEqual(value, 1400.0)
        self.assertEqual(method, "raw_price:total_price_billion")

        value, _ = parse_price_billion_with_method("1400 triệu")
        self.assertFloatEqual(value, 1.4)

        value, _ = parse_price_billion_with_method("4 tỷ 350")
        self.assertFloatEqual(value, 4.35)

    def test_area_and_query_parser_share_fixed_number_parser(self) -> None:
        area, method = parse_area_m2_with_method("1.416 m²")
        self.assertFloatEqual(area, 1416.0)
        self.assertEqual(method, "raw_area")

        profile = parse_user_query("Tài chính 1.400 tỷ mua nhà")
        self.assertFloatEqual(profile["budget_billion"], 1400.0)
        profile = parse_user_query("Tài chính 1.400,5 tỷ mua nhà")
        self.assertFloatEqual(profile["budget_billion"], 1400.5)

    def test_standardizer_outputs_large_price_billion(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "Tieu_De": "Nhà trung tâm giá 1.400 tỷ",
                    "Mo_Ta": "Diện tích 200 m2, pháp lý sổ hồng",
                    "Gia": "1.400 tỷ",
                    "Dien_Tich": "200 m²",
                    "Vi_Tri": "Quận 1, Hồ Chí Minh",
                    "Phap_Ly": "Sổ hồng",
                }
            ]
        )
        standardized = standardize_dataframe(raw)
        self.assertFloatEqual(float(standardized.loc[0, "price_billion"]), 1400.0)
        cleaned = clean_and_deduplicate(standardized)
        self.assertNotIn("abnormal_price", cleaned.loc[0, "data_flags"])


if __name__ == "__main__":
    unittest.main()
