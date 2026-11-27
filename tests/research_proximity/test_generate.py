"""Deterministic unit tests for the synthetic data generator."""

from collections import Counter
import time

import numpy as np
import pytest

from dreams_app.core.extra.place_emotion_signature import (
    CHIME_DIMENSIONS,
    build_place_signature,
)
from dreams_app.core.extra.proximity_trajectory import (
    PlaceVisit,
    build_place_trajectories,
)
from research.proximity_validation.generate import (
    LEAF_TYPES,
    PLACE_TAXONOMY_TREE,
    TAXONOMY_DEPTHS,
    TAXONOMY_NODES,
    GeneratorConfig,
    export_chime_by_type,
    export_to_place_visits,
    generate_dataset,
)


def _same_type_same_city_fraction(places) -> float:
    same_city = 0
    total_pairs = 0
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            if places[i].type == places[j].type:
                total_pairs += 1
                if places[i].city == places[j].city:
                    same_city += 1
    assert total_pairs > 0
    return same_city / total_pairs


def test_determinism_same_and_different_seeds():
    cfg = GeneratorConfig()
    ds_a = generate_dataset(cfg, seed=42)
    ds_b = generate_dataset(cfg, seed=42)
    ds_other = generate_dataset(cfg, seed=99)

    assert ds_a.places == ds_b.places
    assert ds_a.visits == ds_b.visits
    assert ds_a.effects == ds_b.effects
    assert ds_a.effects.category_effect != ds_other.effects.category_effect
    assert ds_a.visits != ds_other.visits


def test_dataset_shapes_and_place_attributes():
    cfg = GeneratorConfig(n_places=60, n_cities=4, n_users=40)
    ds = generate_dataset(cfg, seed=7)

    assert len(ds.places) == 60
    cities = {p.city for p in ds.places}
    assert len(cities) == 4
    for place in ds.places:
        assert place.id.startswith("place_")
        assert place.name
        assert place.type in LEAF_TYPES
        assert place.group in PLACE_TAXONOMY_TREE
        assert -90.0 <= place.latitude <= 90.0
        assert -180.0 <= place.longitude <= 180.0
        assert place.language
        assert 0 <= len(place.cultural_tags) <= 3

    assert len(ds.effects.user_effect) == 40
    assert len(ds.visits) > 0


def test_chime_vectors_sum_to_one():
    ds = generate_dataset(GeneratorConfig(), seed=11)
    for visit in ds.visits:
        assert tuple(visit.chime.keys()) == tuple(CHIME_DIMENSIONS)
        vals = list(visit.chime.values())
        assert all(0.0 < v < 1.0 for v in vals)
        assert sum(vals) == pytest.approx(1.0, abs=1e-6)


def test_taxonomy_tree_depths_and_structure():
    assert TAXONOMY_DEPTHS["root"] == 0
    assert TAXONOMY_NODES["root"].parent is None
    assert len(PLACE_TAXONOMY_TREE) == 5

    for group_name, leaves in PLACE_TAXONOMY_TREE.items():
        assert TAXONOMY_DEPTHS[group_name] == 1
        assert TAXONOMY_NODES[group_name].parent == "root"
        assert len(leaves) == 3
        for leaf in leaves:
            assert TAXONOMY_DEPTHS[leaf] == 2
            assert TAXONOMY_NODES[leaf].parent == group_name
            assert TAXONOMY_NODES[leaf].children == ()


def test_geo_category_correlation_changes_same_city_clustering():
    cfg_indep = GeneratorConfig(n_places=120, n_cities=4, geo_category_correlation=0.0)
    cfg_clust = GeneratorConfig(n_places=120, n_cities=4, geo_category_correlation=1.0)

    ds_indep = generate_dataset(cfg_indep, seed=123)
    ds_clust = generate_dataset(cfg_clust, seed=123)

    frac_indep = _same_type_same_city_fraction(ds_indep.places)
    frac_clust = _same_type_same_city_fraction(ds_clust.places)

    assert frac_clust == pytest.approx(1.0)
    assert 0.05 <= frac_indep <= 0.45
    assert frac_clust - frac_indep > 0.5


def test_strength_parameters_drive_between_type_and_between_place_variance():
    def _mean_between_type_variance(ds) -> float:
        by_type = {}
        for visit in ds.visits:
            by_type.setdefault(visit.place_type, []).append(
                [visit.chime[d] for d in CHIME_DIMENSIONS]
            )
        type_means = np.array([np.mean(rows, axis=0) for rows in by_type.values()])
        return float(np.mean(np.var(type_means, axis=0)))

    cfg_null = GeneratorConfig(
        n_places=30,
        n_users=40,
        mean_visits_per_user=12.0,
        category_strength=0.0,
        group_strength=0.0,
        place_strength=0.0,
        user_strength=0.0,
        noise_sd=0.25,
    )
    cfg_strong_cat = GeneratorConfig(
        n_places=30,
        n_users=40,
        mean_visits_per_user=12.0,
        category_strength=2.5,
        group_strength=0.0,
        place_strength=0.0,
        user_strength=0.0,
        noise_sd=0.25,
    )

    var_null = _mean_between_type_variance(generate_dataset(cfg_null, seed=5))
    var_strong = _mean_between_type_variance(generate_dataset(cfg_strong_cat, seed=5))
    assert var_strong > 5.0 * var_null


def test_export_to_repo_place_visits_and_signatures():
    ds = generate_dataset(GeneratorConfig(), seed=21)
    place_visits = export_to_place_visits(ds)

    assert len(place_visits) == len(ds.visits)
    assert all(isinstance(pv, PlaceVisit) for pv in place_visits)

    trajectories = build_place_trajectories(place_visits)
    counts_by_type = Counter(v.place_type for v in ds.visits)
    assert set(trajectories.keys()) == set(counts_by_type.keys())
    for place_type, count in counts_by_type.items():
        assert trajectories[place_type].visit_count == count

    grouped = export_chime_by_type(ds)
    for place_type, chime_list in grouped.items():
        sig = build_place_signature(place_type, chime_list)
        assert sig.visit_count == len(chime_list)


def test_generation_speed_under_one_second():
    cfg = GeneratorConfig()
    start = time.perf_counter()
    ds = generate_dataset(cfg, seed=0)
    elapsed = time.perf_counter() - start

    assert len(ds.places) == 60
    assert elapsed < 1.0
