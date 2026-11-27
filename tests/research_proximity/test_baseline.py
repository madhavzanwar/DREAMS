"""Deterministic unit tests for baseline proximity and signature reproduction."""

from copy import deepcopy

import numpy as np
import pytest

from dreams_app.core.extra.place_emotion_signature import chime_proximity
from research.proximity_validation.baseline import (
    CATEGORY_ONLY_WEIGHTS,
    GEO_ONLY_WEIGHTS,
    build_baseline_places,
    build_baseline_visits,
    build_instance_signatures,
    build_type_signatures,
    compute_proximity_matrix,
    get_signature_for_place,
)


def test_matrices_are_symmetric():
    places = build_baseline_places()
    for weights in (None, GEO_ONLY_WEIGHTS, CATEGORY_ONLY_WEIGHTS):
        matrix = compute_proximity_matrix(places, weights=weights)
        assert matrix.shape == (len(places), len(places))
        assert np.allclose(matrix, matrix.T)
    mat_default = compute_proximity_matrix(places)
    # Spot-check Example A (anc_church_recovery vs lis_church_se) != 1.0 or 0.0
    assert 0.4 < mat_default[0, 8] < 0.5


def test_matrix_diagonals_equal_one():
    places = build_baseline_places()
    for weights in (None, GEO_ONLY_WEIGHTS, CATEGORY_ONLY_WEIGHTS):
        matrix = compute_proximity_matrix(places, weights=weights)
        assert np.allclose(np.diag(matrix), 1.0)


def test_geo_only_matrix_depends_only_on_coordinates():
    places = build_baseline_places()
    baseline_geo = compute_proximity_matrix(places, weights=GEO_ONLY_WEIGHTS)

    mutated = deepcopy(places)
    for idx, place in enumerate(mutated):
        place["type"] = f"synthetic_type_{idx}"
        place["language"] = f"lang_{idx}"
        place["cultural_tags"] = [f"tag_{idx}"]
    mutated_geo = compute_proximity_matrix(mutated, weights=GEO_ONLY_WEIGHTS)
    assert np.allclose(baseline_geo, mutated_geo)

    shifted = deepcopy(places)
    shifted[0]["latitude"] = 0.0
    shifted[0]["longitude"] = 0.0
    shifted_geo = compute_proximity_matrix(shifted, weights=GEO_ONLY_WEIGHTS)
    assert not np.allclose(baseline_geo, shifted_geo)


def test_category_only_matrix_depends_only_on_type():
    places = build_baseline_places()
    baseline_cat = compute_proximity_matrix(places, weights=CATEGORY_ONLY_WEIGHTS)

    mutated = deepcopy(places)
    for idx, place in enumerate(mutated):
        place["latitude"] = float((idx * 13) % 80 - 40)
        place["longitude"] = float((idx * 29) % 160 - 80)
        place["language"] = f"lang_{idx}"
        place["cultural_tags"] = [f"tag_{idx}"]
    mutated_cat = compute_proximity_matrix(mutated, weights=CATEGORY_ONLY_WEIGHTS)
    assert np.allclose(baseline_cat, mutated_cat)

    retyped = deepcopy(places)
    retyped[0]["type"] = "park"
    retyped_cat = compute_proximity_matrix(retyped, weights=CATEGORY_ONLY_WEIGHTS)
    assert not np.allclose(baseline_cat, retyped_cat)


def test_same_type_places_get_identical_signature_despite_visit_differences():
    places = build_baseline_places()
    visits = build_baseline_visits()
    type_sigs = build_type_signatures(visits)
    inst_sigs = build_instance_signatures(visits)

    churches = [p for p in places if p["type"] == "church"]
    assert len(churches) >= 2

    sig_first = get_signature_for_place(churches[0], type_sigs)
    sig_second = get_signature_for_place(churches[1], type_sigs)
    assert sig_first == sig_second
    assert chime_proximity(sig_first, sig_second) == pytest.approx(1.0)

    inst_anc = inst_sigs["anc_church_recovery"]
    inst_sea = inst_sigs["sea_church_grief"]
    assert inst_anc.chime_vector["Hope"] > inst_sea.chime_vector["Hope"] + 0.5
    assert chime_proximity(inst_anc, inst_sea) < 1.0
