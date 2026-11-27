"""Deterministic unit tests for proximity and similarity dimensions."""

import numpy as np
import pytest

from dreams_app.core.extra.proximity_calculator import (
    categorical_proximity,
    composite_proximity,
    normalize_geographic_distance,
)
from dreams_app.location_proximity import calculate_distance
from research.proximity_validation.distances import (
    REPO_DEFAULT_WEIGHTS,
    category_tree_similarity,
    composite_similarity,
    cultural_proximity,
    geographic_proximity,
    linguistic_proximity,
    pairwise_matrix,
    semantic_name_similarity,
)
from research.proximity_validation.generate import GeneratorConfig, generate_dataset


def _sample_places():
    return [
        {
            "id": "p0",
            "name": "St. Mary's Church",
            "type": "church",
            "latitude": 61.2176,
            "longitude": -149.8997,
            "language": "english",
            "cultural_tags": ["christian", "community", "alaskan"],
        },
        {
            "id": "p1",
            "name": "Holy Trinity Church",
            "type": "church",
            "latitude": 47.6074,
            "longitude": -122.3257,
            "language": "english",
            "cultural_tags": ["christian", "historic"],
        },
        {
            "id": "p2",
            "name": "Al-Noor Mosque",
            "type": "mosque",
            "latitude": 45.5017,
            "longitude": -73.5673,
            "language": "french",
            "cultural_tags": ["spiritual", "community"],
        },
        {
            "id": "p3",
            "name": "Central City Library",
            "type": "library",
            "latitude": 61.2190,
            "longitude": -149.8950,
            "language": "english",
            "cultural_tags": ["learning", "public"],
        },
        {
            "id": "p4",
            "name": "Lisbon General Hospital",
            "type": "hospital",
            "latitude": 38.7176,
            "longitude": -9.1370,
            "language": "portuguese",
            "cultural_tags": [],
        },
        {
            "id": "p5",
            "name": "Neighborhood Free Clinic",
            "type": "clinic",
            "latitude": 38.7200,
            "longitude": -9.1400,
            "language": "portuguese",
            "cultural_tags": ["care"],
        },
    ]


def test_geographic_symmetry_identity_and_range():
    places = _sample_places()
    mat = pairwise_matrix(places, dimension="geo")
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.all((mat > 0.0) & (mat <= 1.0))


def test_category_tree_symmetry_identity_and_range():
    places = _sample_places()
    mat = pairwise_matrix(places, dimension="category_tree")
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.all((mat >= 0.0) & (mat <= 1.0))


def test_category_tree_wu_palmer_ordering():
    same_type = category_tree_similarity({"type": "church"}, {"type": "church"})
    same_group = category_tree_similarity({"type": "church"}, {"type": "mosque"})
    diff_group = category_tree_similarity({"type": "church"}, {"type": "library"})
    assert same_type == pytest.approx(1.0)
    assert same_group > diff_group
    assert same_type > same_group > diff_group >= 0.0


def test_category_tree_vs_repo_categorical_proximity():
    assert category_tree_similarity(
        {"type": "hospital"}, {"type": "clinic"}
    ) == pytest.approx(categorical_proximity("hospital", "clinic"))
    assert category_tree_similarity(
        {"type": "church"}, {"type": "temple"}
    ) == pytest.approx(categorical_proximity("church", "temple"))
    for a, b in (
        ("church", "mosque"),
        ("park", "garden"),
        ("clinic", "pharmacy"),
        ("library", "school"),
        ("cafe", "bar"),
    ):
        assert categorical_proximity(a, b) == 0.0
        assert category_tree_similarity({"type": a}, {"type": b}) > 0.0


def test_semantic_name_fallback_church_higher_than_library():
    church_a = {"name": "St. Mary's Church"}
    church_b = {"name": "Holy Trinity Church"}
    library = {"name": "Central City Library"}

    sim_churches = semantic_name_similarity(church_a, church_b, backend="tfidf")
    sim_church_lib = semantic_name_similarity(church_a, library, backend="tfidf")
    assert 0.0 <= sim_church_lib < sim_churches <= 1.0


def test_semantic_name_symmetry_identity_and_range():
    places = _sample_places()
    mat = pairwise_matrix(places, dimension="semantic_name", semantic_backend="tfidf")
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.all((mat >= 0.0) & (mat <= 1.0))


def test_linguistic_and_cultural_symmetry_identity_and_range():
    places = _sample_places()
    ling_mat = pairwise_matrix(places, dimension="ling")
    cult_mat = pairwise_matrix(places, dimension="cult")

    assert np.allclose(ling_mat, ling_mat.T)
    assert np.allclose(np.diag(ling_mat), 1.0)
    assert np.all((ling_mat == 0.0) | (ling_mat == 1.0))

    assert np.allclose(cult_mat, cult_mat.T)
    assert np.allclose(np.diag(cult_mat), 1.0)
    assert np.all((cult_mat >= 0.0) & (cult_mat <= 1.0))
    assert cultural_proximity(places[4], places[4]) == 1.0


def test_composite_reproduces_repo_default_weights_across_pairs():
    places = _sample_places()
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (1, 3)]
    for i, j in pairs:
        p1, p2 = places[i], places[j]
        dist_km = (
            calculate_distance(
                p1["latitude"],
                p1["longitude"],
                p2["latitude"],
                p2["longitude"],
            )
            / 1000.0
        )
        geo_prox = normalize_geographic_distance(dist_km)
        expected = composite_proximity(
            {**p1, "geo_proximity": geo_prox},
            {**p2, "geo_proximity": geo_prox},
            weights=REPO_DEFAULT_WEIGHTS,
        )
        actual = composite_similarity(p1, p2, weights=REPO_DEFAULT_WEIGHTS)
        assert actual == pytest.approx(expected, abs=1e-12)


def test_composite_renormalizes_with_missing_dimensions():
    p1 = {
        "name": "St. Mary's Church",
        "type": "church",
        "latitude": 61.2176,
        "longitude": -149.8997,
        "language": "english",
        "cultural_tags": [],
    }
    p2 = {
        "name": "Holy Trinity Church",
        "type": "church",
        "latitude": 47.6074,
        "longitude": -122.3257,
        "language": None,
        "cultural_tags": ["christian"],
    }
    weights = {"geo": 0.3, "category_tree": 0.4, "ling": 0.15, "cult": 0.15}
    geo = geographic_proximity(p1, p2)
    expected = (0.3 * geo + 0.4 * 1.0) / (0.3 + 0.4)
    assert composite_similarity(p1, p2, weights=weights) == pytest.approx(expected)


def test_composite_with_place_records_and_semantic_dimension():
    ds = generate_dataset(GeneratorConfig(n_places=10, n_users=5), seed=4)
    subset = ds.places[:6]
    weights = {
        "geo": 0.25,
        "category_tree": 0.35,
        "semantic_name": 0.2,
        "ling": 0.1,
        "cult": 0.1,
    }
    mat = pairwise_matrix(
        subset,
        dimension="composite",
        weights=weights,
        semantic_backend="tfidf",
    )
    assert mat.shape == (6, 6)
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.all((mat >= 0.0) & (mat <= 1.0))


def test_invalid_inputs_raise_clear_errors():
    valid = _sample_places()[0]
    with pytest.raises(ValueError, match="latitude"):
        geographic_proximity({**valid, "latitude": 120.0}, valid)
    with pytest.raises(ValueError, match="Unknown place type"):
        category_tree_similarity({"type": "spaceship"}, valid)
    with pytest.raises(ValueError, match="backend"):
        semantic_name_similarity(valid, valid, backend="nonexistent")
    with pytest.raises(ValueError, match="language"):
        linguistic_proximity({"language": ""}, valid)
    with pytest.raises(ValueError, match="cultural_tags"):
        cultural_proximity({"cultural_tags": None}, valid)
    with pytest.raises(ValueError, match="Unknown proximity dimension"):
        composite_similarity(valid, valid, weights={"bogus": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        composite_similarity(valid, valid, weights={"geo": -0.5})
    with pytest.raises(ValueError, match="non-empty"):
        pairwise_matrix([])
    with pytest.raises(ValueError, match="Unknown dimension"):
        pairwise_matrix([valid], dimension="unknown_dim")


def test_semantic_auto_fallback_handles_missing_package(monkeypatch):
    monkeypatch.setattr(
        "research.proximity_validation.distances._sentence_transformers_available",
        lambda: False,
    )
    sim = semantic_name_similarity(
        {"name": "St. Mary's Church"},
        {"name": "Holy Trinity Church"},
        backend="auto",
    )
    sim_tfidf = semantic_name_similarity(
        {"name": "St. Mary's Church"},
        {"name": "Holy Trinity Church"},
        backend="tfidf",
    )
    assert sim == pytest.approx(sim_tfidf)
