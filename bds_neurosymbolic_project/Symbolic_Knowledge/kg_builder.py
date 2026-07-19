"""Sinh Knowledge Graph dạng triples subject-predicate-object."""
from __future__ import annotations

import pandas as pd


def _add(triples: list[list[object]], s: object, p: str, o: object) -> None:
    if o is None:
        return
    try:
        if pd.isna(o):
            return
    except Exception:
        pass
    triples.append([s, p, o])


def create_triples(df: pd.DataFrame) -> pd.DataFrame:
    triples: list[list[object]] = []
    for _, row in df.iterrows():
        pid = row.get("property_id")
        if not pid:
            continue
        _add(triples, pid, "type", "Property")
        for col, pred in {
            "source": "hasSource",
            "source_sheet": "hasSourceSheet",
            "title": "hasTitle",
            "district": "locatedInDistrict",
            "ward": "locatedInWard",
            "location_zone": "locatedInZone",
            "property_type": "hasPropertyType",
            "price_band": "hasPriceBand",
            "area_band": "hasAreaBand",
            "bedrooms_band": "hasBedroomsBand",
            "road_band": "hasRoadBand",
            "frontage_band": "hasFrontageBand",
            "legal_class": "hasLegalClass",
        }.items():
            _add(triples, pid, pred, row.get(col))

        for col, pred in {
            "price_billion": "hasPriceBillion",
            "area_m2": "hasAreaM2",
            "price_per_m2_million": "hasPricePerM2Million",
            "bedrooms": "hasBedrooms",
            "bathrooms": "hasBathrooms",
            "floors": "hasFloors",
            "frontage_m": "hasFrontageM",
            "road_width_m": "hasRoadWidthM",
            "legal_score": "hasLegalScore",
            "family_suitability_score": "hasFamilySuitabilityScore",
            "business_potential_score": "hasBusinessPotentialScore",
            "rental_potential_score": "hasRentalPotentialScore",
            "investment_potential_score": "hasInvestmentPotentialScore",
            "risk_score": "hasRiskScore",
            "data_quality_score": "hasDataQualityScore",
        }.items():
            _add(triples, pid, pred, row.get(col))

        for tag in row.get("amenity_tags", []) if isinstance(row.get("amenity_tags", []), list) else []:
            _add(triples, pid, "hasAmenityTag", tag)
        for fact in row.get("symbolic_facts", []) if isinstance(row.get("symbolic_facts", []), list) else []:
            _add(triples, pid, "inferredFact", fact)
        for risk in row.get("risk_flags", []) if isinstance(row.get("risk_flags", []), list) else []:
            _add(triples, pid, "hasRiskFlag", risk)
        for rule in row.get("triggered_rules", []) if isinstance(row.get("triggered_rules", []), list) else []:
            _add(triples, pid, "triggeredRule", rule)

    return pd.DataFrame(triples, columns=["subject", "predicate", "object"])
