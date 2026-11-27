"""Reproduce the baseline DREAMS proximity and signature behavior."""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreams_app.core.extra import (  # noqa: E402
    categorical_proximity,
    composite_proximity,
    cultural_similarity,
    linguistic_similarity,
    normalize_geographic_distance,
)
from dreams_app.core.extra.place_emotion_signature import (  # noqa: E402
    CHIME_DIMENSIONS,
    PlaceEmotionSignature,
    build_place_signature,
    chime_proximity,
    detect_recovery_anchors,
)
from dreams_app.location_proximity import calculate_distance  # noqa: E402

GEO_ONLY_WEIGHTS: Dict[str, float] = {"geo": 1.0, "cat": 0.0, "ling": 0.0, "cult": 0.0}
CATEGORY_ONLY_WEIGHTS: Dict[str, float] = {
    "geo": 0.0,
    "cat": 1.0,
    "ling": 0.0,
    "cult": 0.0,
}
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "research" / "results" / "baseline_output.txt"


def build_baseline_places() -> List[Dict]:
    """Return 12 hand-written places across Anchorage, Seattle, and Lisbon."""
    return [
        {
            "id": "anc_church_recovery",
            "name": "St. John's Community Church",
            "city": "Anchorage",
            "latitude": 61.2176,
            "longitude": -149.8997,
            "type": "church",
            "language": "english",
            "cultural_tags": ["christian", "community", "alaskan"],
        },
        {
            "id": "anc_clinic_north",
            "name": "North Star Free Clinic",
            "city": "Anchorage",
            "latitude": 61.2195,
            "longitude": -149.8940,
            "type": "clinic",
            "language": "english",
            "cultural_tags": ["healthcare", "community", "alaskan"],
        },
        {
            "id": "anc_park_town",
            "name": "Town Square Park",
            "city": "Anchorage",
            "latitude": 61.2178,
            "longitude": -149.8950,
            "type": "park",
            "language": "english",
            "cultural_tags": ["outdoor", "public", "alaskan"],
        },
        {
            "id": "anc_cafe_kaladi",
            "name": "Kaladi Brothers Cafe",
            "city": "Anchorage",
            "latitude": 61.2165,
            "longitude": -149.8880,
            "type": "cafe",
            "language": "english",
            "cultural_tags": ["coffee", "social", "alaskan"],
        },
        {
            "id": "sea_church_grief",
            "name": "St. James Memorial Chapel",
            "city": "Seattle",
            "latitude": 47.6074,
            "longitude": -122.3257,
            "type": "church",
            "language": "english",
            "cultural_tags": ["christian", "memorial", "pacific_northwest"],
        },
        {
            "id": "sea_hospital_harbor",
            "name": "Harborview Medical Hospital",
            "city": "Seattle",
            "latitude": 47.6040,
            "longitude": -122.3240,
            "type": "hospital",
            "language": "english",
            "cultural_tags": ["healthcare", "emergency", "pacific_northwest"],
        },
        {
            "id": "sea_park_cal",
            "name": "Cal Anderson Park",
            "city": "Seattle",
            "latitude": 47.6175,
            "longitude": -122.3190,
            "type": "park",
            "language": "english",
            "cultural_tags": ["outdoor", "public", "pacific_northwest"],
        },
        {
            "id": "sea_cafe_victrola",
            "name": "Victrola Coffee Cafe",
            "city": "Seattle",
            "latitude": 47.6130,
            "longitude": -122.3220,
            "type": "cafe",
            "language": "english",
            "cultural_tags": ["coffee", "social", "pacific_northwest"],
        },
        {
            "id": "lis_church_se",
            "name": "Lisbon Cathedral",
            "city": "Lisbon",
            "latitude": 38.7098,
            "longitude": -9.1326,
            "type": "church",
            "language": "portuguese",
            "cultural_tags": ["christian", "catholic", "european"],
        },
        {
            "id": "lis_hospital_jose",
            "name": "Hospital de Sao Jose",
            "city": "Lisbon",
            "latitude": 38.7176,
            "longitude": -9.1370,
            "type": "hospital",
            "language": "portuguese",
            "cultural_tags": ["healthcare", "public", "european"],
        },
        {
            "id": "lis_park_eduardo",
            "name": "Parque Eduardo VII",
            "city": "Lisbon",
            "latitude": 38.7285,
            "longitude": -9.1520,
            "type": "park",
            "language": "portuguese",
            "cultural_tags": ["outdoor", "public", "european"],
        },
        {
            "id": "lis_cafe_brasileira",
            "name": "Cafe A Brasileira",
            "city": "Lisbon",
            "latitude": 38.7106,
            "longitude": -9.1422,
            "type": "cafe",
            "language": "portuguese",
            "cultural_tags": ["coffee", "historic", "european"],
        },
    ]


def build_baseline_visits() -> List[Dict]:
    """Return hand-made CHIME visit records showing within-type place differences."""
    return [
        {
            "place_id": "anc_church_recovery",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.85,
                "Hope": 0.90,
                "Identity": 0.70,
                "Meaning": 0.85,
                "Empowerment": 0.75,
            },
        },
        {
            "place_id": "anc_church_recovery",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.80,
                "Hope": 0.85,
                "Identity": 0.75,
                "Meaning": 0.80,
                "Empowerment": 0.70,
            },
        },
        {
            "place_id": "anc_church_recovery",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.90,
                "Hope": 0.85,
                "Identity": 0.75,
                "Meaning": 0.90,
                "Empowerment": 0.80,
            },
        },
        {
            "place_id": "sea_church_grief",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.25,
                "Hope": 0.20,
                "Identity": 0.30,
                "Meaning": 0.65,
                "Empowerment": 0.15,
            },
        },
        {
            "place_id": "sea_church_grief",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.20,
                "Hope": 0.15,
                "Identity": 0.25,
                "Meaning": 0.60,
                "Empowerment": 0.20,
            },
        },
        {
            "place_id": "sea_church_grief",
            "place_type": "church",
            "chime": {
                "Connectedness": 0.30,
                "Hope": 0.25,
                "Identity": 0.35,
                "Meaning": 0.70,
                "Empowerment": 0.25,
            },
        },
        {
            "place_id": "anc_clinic_north",
            "place_type": "clinic",
            "chime": {
                "Connectedness": 0.55,
                "Hope": 0.70,
                "Identity": 0.45,
                "Meaning": 0.50,
                "Empowerment": 0.65,
            },
        },
        {
            "place_id": "anc_clinic_north",
            "place_type": "clinic",
            "chime": {
                "Connectedness": 0.60,
                "Hope": 0.75,
                "Identity": 0.50,
                "Meaning": 0.55,
                "Empowerment": 0.70,
            },
        },
        {
            "place_id": "sea_hospital_harbor",
            "place_type": "hospital",
            "chime": {
                "Connectedness": 0.30,
                "Hope": 0.40,
                "Identity": 0.25,
                "Meaning": 0.35,
                "Empowerment": 0.30,
            },
        },
        {
            "place_id": "sea_hospital_harbor",
            "place_type": "hospital",
            "chime": {
                "Connectedness": 0.35,
                "Hope": 0.45,
                "Identity": 0.30,
                "Meaning": 0.40,
                "Empowerment": 0.35,
            },
        },
        {
            "place_id": "anc_park_town",
            "place_type": "park",
            "chime": {
                "Connectedness": 0.65,
                "Hope": 0.75,
                "Identity": 0.60,
                "Meaning": 0.70,
                "Empowerment": 0.75,
            },
        },
        {
            "place_id": "sea_park_cal",
            "place_type": "park",
            "chime": {
                "Connectedness": 0.70,
                "Hope": 0.80,
                "Identity": 0.65,
                "Meaning": 0.72,
                "Empowerment": 0.78,
            },
        },
        {
            "place_id": "anc_cafe_kaladi",
            "place_type": "cafe",
            "chime": {
                "Connectedness": 0.75,
                "Hope": 0.65,
                "Identity": 0.60,
                "Meaning": 0.55,
                "Empowerment": 0.60,
            },
        },
        {
            "place_id": "lis_cafe_brasileira",
            "place_type": "cafe",
            "chime": {
                "Connectedness": 0.80,
                "Hope": 0.70,
                "Identity": 0.65,
                "Meaning": 0.60,
                "Empowerment": 0.65,
            },
        },
    ]


def compute_pairwise_geo_km(places: Sequence[Dict]) -> np.ndarray:
    """Compute pairwise Haversine distance matrix in kilometers."""
    n = len(places)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist_km = (
                calculate_distance(
                    places[i]["latitude"],
                    places[i]["longitude"],
                    places[j]["latitude"],
                    places[j]["longitude"],
                )
                / 1000.0
            )
            distances[i, j] = dist_km
            distances[j, i] = dist_km
    return distances


def compute_proximity_matrix(
    places: Sequence[Dict],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute the N x N composite_proximity matrix using DREAMS existing functions."""
    n = len(places)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                geo_prox = normalize_geographic_distance(0.0)
            else:
                dist_km = (
                    calculate_distance(
                        places[i]["latitude"],
                        places[i]["longitude"],
                        places[j]["latitude"],
                        places[j]["longitude"],
                    )
                    / 1000.0
                )
                geo_prox = normalize_geographic_distance(dist_km)
            p_i = {**places[i], "geo_proximity": geo_prox}
            p_j = {**places[j], "geo_proximity": geo_prox}
            score = composite_proximity(p_i, p_j, weights=weights)
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix


def build_type_signatures(visits: Sequence[Dict]) -> Dict[str, PlaceEmotionSignature]:
    """Aggregate visit CHIME dicts into one PlaceEmotionSignature per place_type."""
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for visit in visits:
        grouped.setdefault(visit["place_type"], []).append(visit["chime"])
    return {
        place_type: build_place_signature(place_type, chime_list)
        for place_type, chime_list in grouped.items()
    }


def build_instance_signatures(
    visits: Sequence[Dict],
) -> Dict[str, PlaceEmotionSignature]:
    """Aggregate visit CHIME dicts per physical place_id for comparison only."""
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for visit in visits:
        grouped.setdefault(visit["place_id"], []).append(visit["chime"])
    return {
        place_id: build_place_signature(place_id, chime_list)
        for place_id, chime_list in grouped.items()
    }


def get_signature_for_place(
    place: Dict,
    type_signatures: Dict[str, PlaceEmotionSignature],
) -> PlaceEmotionSignature:
    """Look up the current type-level PlaceEmotionSignature for a place."""
    return type_signatures[place["type"]]


def format_matrix(title: str, ids: Sequence[str], matrix: np.ndarray) -> str:
    """Render a symmetric proximity matrix as an aligned plain-text table."""
    short = [pid[:9] for pid in ids]
    col_w = 10
    lines = [title]
    header = f"{'':10s}" + "".join(f"{label:>{col_w}s}" for label in short)
    lines.append(header)
    for row_label, row in zip(short, matrix):
        cells = "".join(f"{val:{col_w}.3f}" for val in row)
        lines.append(f"{row_label:10s}{cells}")
    return "\n".join(lines)


def _format_signature(label: str, sig: PlaceEmotionSignature) -> str:
    dims = ", ".join(
        f"{dim[:4]}={sig.chime_vector[dim]:.3f}" for dim in CHIME_DIMENSIONS
    )
    return (
        f"{label} (place_type={sig.place_type}, visits={sig.visit_count}, "
        f"volatility={sig.volatility:.3f}): [{dims}]"
    )


def _pair_breakdown(p1: Dict, p2: Dict) -> str:
    dist_km = (
        calculate_distance(
            p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]
        )
        / 1000.0
    )
    geo = normalize_geographic_distance(dist_km)
    cat = categorical_proximity(p1["type"], p2["type"])
    ling = linguistic_similarity(p1["language"], p2["language"])
    cult = cultural_similarity(p1["cultural_tags"], p2["cultural_tags"])
    comp_default = composite_proximity(
        {**p1, "geo_proximity": geo},
        {**p2, "geo_proximity": geo},
    )
    comp_geo = composite_proximity(
        {**p1, "geo_proximity": geo},
        {**p2, "geo_proximity": geo},
        weights=GEO_ONLY_WEIGHTS,
    )
    comp_cat = composite_proximity(
        {**p1, "geo_proximity": geo},
        {**p2, "geo_proximity": geo},
        weights=CATEGORY_ONLY_WEIGHTS,
    )
    return (
        f"  {p1['id']} ({p1['city']}, {p1['type']}) vs "
        f"{p2['id']} ({p2['city']}, {p2['type']}):\n"
        f"    distance_km={dist_km:.2f} km -> geo={geo:.3f}, cat={cat:.3f}, "
        f"ling={ling:.3f}, cult={cult:.3f}\n"
        f"    composite_default={comp_default:.3f} | geo_only={comp_geo:.3f} | "
        f"category_only={comp_cat:.3f}"
    )


def run_baseline(output_path: Optional[Path] = None) -> str:
    """Execute the baseline reproduction and return the formatted report."""
    places = build_baseline_places()
    visits = build_baseline_visits()
    place_by_id = {p["id"]: p for p in places}
    ids = [p["id"] for p in places]

    mat_default = compute_proximity_matrix(places)
    mat_geo = compute_proximity_matrix(places, weights=GEO_ONLY_WEIGHTS)
    mat_cat = compute_proximity_matrix(places, weights=CATEGORY_ONLY_WEIGHTS)

    type_sigs = build_type_signatures(visits)
    inst_sigs = build_instance_signatures(visits)
    anchors = detect_recovery_anchors(type_sigs, threshold=0.6, min_visits=2)

    anc_church = place_by_id["anc_church_recovery"]
    sea_church = place_by_id["sea_church_grief"]
    sig_anc_via_type = get_signature_for_place(anc_church, type_sigs)
    sig_sea_via_type = get_signature_for_place(sea_church, type_sigs)

    sections = [
        "=== DREAMS BASELINE PROXIMITY & SIGNATURE REPORT ===",
        f"Places: {len(places)} | Visits: {len(visits)} | Place Types: {len(type_sigs)}",
        "",
        format_matrix(
            "1. Pairwise composite_proximity (default weights {'geo':0.3,'cat':0.4,'ling':0.15,'cult':0.15})",
            ids,
            mat_default,
        ),
        "",
        format_matrix(
            "2a. Pairwise composite_proximity (geography only: {'geo':1,'cat':0,'ling':0,'cult':0})",
            ids,
            mat_geo,
        ),
        "",
        format_matrix(
            "2b. Pairwise composite_proximity (category only: {'geo':0,'cat':1,'ling':0,'cult':0})",
            ids,
            mat_cat,
        ),
        "",
        "3. Type-Level PlaceEmotionSignature Summary",
    ]

    for pt in sorted(type_sigs):
        sections.append("  " + _format_signature(pt, type_sigs[pt]))
    sections.append(f"  Recovery anchors (type-level, threshold=0.6): {anchors}")
    sections.append("  Type-lookup check for two different churches:")
    sections.append(
        "    " + _format_signature("anc_church_recovery ->", sig_anc_via_type)
    )
    sections.append(
        "    " + _format_signature("sea_church_grief    ->", sig_sea_via_type)
    )
    sections.append(
        "    Identical type signature? "
        f"{sig_anc_via_type == sig_sea_via_type} | "
        f"chime_proximity={chime_proximity(sig_anc_via_type, sig_sea_via_type):.3f}"
    )

    sections.extend(
        [
            "",
            "4. Three Concrete Research-Gap Examples",
            "Example A: Far apart, same place type (high cat, near-zero geo):",
            _pair_breakdown(anc_church, place_by_id["lis_church_se"]),
            "",
            "Example B: Nearby in the same city, different place types (high geo, zero cat):",
            _pair_breakdown(anc_church, place_by_id["anc_clinic_north"]),
            "",
            "Example C: Type-level signature hides place-specific emotional divergence:",
            "  Assigned by current type-level model (both return identical signature):",
            "    " + _format_signature("anc_church_recovery", sig_anc_via_type),
            "    " + _format_signature("sea_church_grief   ", sig_sea_via_type),
            "  Actual per-place visit profiles pooled into church:",
            "    "
            + _format_signature(
                "anc_church_recovery (actual)", inst_sigs["anc_church_recovery"]
            ),
            "    "
            + _format_signature(
                "sea_church_grief    (actual)", inst_sigs["sea_church_grief"]
            ),
            "  Actual per-place chime_proximity(anc_church_recovery, sea_church_grief) = "
            f"{chime_proximity(inst_sigs['anc_church_recovery'], inst_sigs['sea_church_grief']):.3f} "
            "(vs 1.000 under type pooling)",
        ]
    )

    report = "\n".join(sections) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8", newline="\n")
    return report


def main() -> None:
    report = run_baseline(output_path=DEFAULT_OUTPUT_PATH)
    print(report, end="")


if __name__ == "__main__":
    main()
