"""Bộ test query mẫu cho đánh giá heuristic."""
TEST_QUERIES = [
    {
        "case_id": "Q01_FAMILY_Q3_8B",
        "query": "Tìm nhà quanh quận 3 khoảng 8 tỷ, 2PN trở lên, pháp lý rõ, phù hợp gia đình",
        "expected_facts": ["suitable_for_family", "low_legal_risk"],
        "avoid_risks": ["need_legal_verification", "low_data_confidence"],
    },
    {
        "case_id": "Q02_BUSINESS_10B",
        "query": "Tìm nhà mặt tiền kinh doanh dưới 10 tỷ, đường xe hơi, pháp lý rõ",
        "expected_facts": ["suitable_for_business", "good_car_access"],
        "avoid_risks": ["need_legal_verification"],
    },
    {
        "case_id": "Q03_RENTAL_Q7",
        "query": "Tìm căn hộ quận 7 để cho thuê, có nội thất, giá hợp lý",
        "expected_facts": ["suitable_for_rental"],
        "avoid_risks": ["possibly_overpriced"],
    },
    {
        "case_id": "Q04_INVESTMENT_TD",
        "query": "Tìm bất động sản Thủ Đức để đầu tư, pháp lý rõ, có tiềm năng tăng giá",
        "expected_facts": ["suitable_for_investment", "low_legal_risk"],
        "avoid_risks": ["need_legal_verification"],
    },
]
