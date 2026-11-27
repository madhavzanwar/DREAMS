"""Deterministic unit tests for evaluation, cross-validation, ANOVA, and sweep."""

import csv

import numpy as np
import pytest

from research.proximity_validation.evaluate import (
    MODEL_FEATURES,
    compute_variance_decomposition,
    emotional_distance,
    evaluate_dataset,
    iter_leave_place_out_splits,
)
from research.proximity_validation.generate import GeneratorConfig, generate_dataset
from research.proximity_validation.run_experiments import (
    format_summary_markdown,
    run_sweep,
    write_experiments_csv,
)


def test_jsd_symmetric_zero_for_identical_and_bounded():
    p = [0.4, 0.2, 0.15, 0.15, 0.1]
    q = [0.1, 0.1, 0.5, 0.2, 0.1]
    assert emotional_distance(p, p) == 0.0
    d_pq = emotional_distance(p, q)
    d_qp = emotional_distance(q, p)
    assert d_pq == pytest.approx(d_qp)
    assert 0.0 < d_pq <= 1.0


def test_leave_place_out_splits_have_no_train_test_leakage():
    n_places = 10
    tri_i, tri_j = np.triu_indices(n_places, k=1)
    for held_out, train_mask, test_mask in iter_leave_place_out_splits(
        tri_i, tri_j, n_places
    ):
        assert np.sum(test_mask) == n_places - 1
        assert np.sum(train_mask) == (n_places - 1) * (n_places - 2) // 2
        assert np.all((tri_i[train_mask] != held_out) & (tri_j[train_mask] != held_out))
        assert np.all((tri_i[test_mask] == held_out) | (tri_j[test_mask] == held_out))
        assert not np.any(train_mask & test_mask)


def test_eligible_places_filtered_by_min_visits():
    cfg = GeneratorConfig(n_places=20, n_users=30, mean_visits_per_user=6.0)
    ds = generate_dataset(cfg, seed=10)
    res_low = evaluate_dataset(
        ds,
        min_visits=2,
        n_permutations=25,
        n_bootstrap=25,
        semantic_backend="tfidf",
        seed=10,
    )
    res_high = evaluate_dataset(
        ds,
        min_visits=8,
        n_permutations=25,
        n_bootstrap=25,
        semantic_backend="tfidf",
        seed=10,
    )
    assert res_low.n_eligible_places >= res_high.n_eligible_places


def test_high_category_zero_corr_cat_weight_exceeds_geo_and_m1_beats_m0():
    cfg = GeneratorConfig(
        n_places=24,
        n_cities=4,
        n_users=30,
        mean_visits_per_user=12.0,
        geo_category_correlation=0.0,
        category_strength=2.5,
        group_strength=0.0,
        place_strength=0.0,
        user_strength=0.0,
        noise_sd=0.2,
    )
    ds = generate_dataset(cfg, seed=7)
    res = evaluate_dataset(
        ds,
        min_visits=5,
        n_permutations=40,
        n_bootstrap=30,
        semantic_backend="tfidf",
        seed=7,
    )
    assert res.models["M1"].spearman > res.models["M0"].spearman + 0.3
    assert res.m4_weights["category_tree"] > abs(res.m4_weights["geo"])
    assert (res.models["M4"].r2 or 0.0) > 0.2


def test_m5_nnls_weights_non_negative_and_predictive():
    cfg = GeneratorConfig(
        n_places=45,
        n_cities=4,
        n_users=40,
        mean_visits_per_user=12.0,
        geo_category_correlation=0.0,
        category_strength=2.0,
        group_strength=1.0,
        place_strength=0.0,
        user_strength=0.2,
        noise_sd=0.2,
    )
    ds = generate_dataset(cfg, seed=14)
    res = evaluate_dataset(
        ds,
        min_visits=5,
        n_permutations=30,
        n_bootstrap=30,
        semantic_backend="tfidf",
        seed=14,
        include_m5=True,
    )
    assert set(res.m5_weights.keys()) == set(MODEL_FEATURES)
    assert all(w >= 0.0 for w in res.m5_weights.values())
    assert res.models["M5"].spearman > 0.25


def test_ablations_identify_dominant_category_dimension():
    cfg = GeneratorConfig(
        n_places=45,
        n_cities=4,
        n_users=40,
        mean_visits_per_user=12.0,
        geo_category_correlation=0.0,
        category_strength=2.0,
        group_strength=1.0,
        place_strength=0.0,
        user_strength=0.0,
        noise_sd=0.2,
    )
    ds = generate_dataset(cfg, seed=42)
    res = evaluate_dataset(
        ds,
        min_visits=5,
        n_permutations=30,
        n_bootstrap=25,
        semantic_backend="tfidf",
        seed=42,
    )
    drop_cat = res.ablations["category_tree"].delta_spearman
    drop_geo = res.ablations["geo"].delta_spearman
    assert drop_cat < -0.05
    assert drop_cat < drop_geo


def test_zero_effect_strengths_do_not_beat_permutation_null():
    cfg = GeneratorConfig(
        n_places=24,
        n_cities=4,
        n_users=30,
        mean_visits_per_user=12.0,
        geo_category_correlation=0.0,
        category_strength=0.0,
        group_strength=0.0,
        place_strength=0.0,
        user_strength=0.0,
        noise_sd=0.5,
    )
    ds = generate_dataset(cfg, seed=19)
    res = evaluate_dataset(
        ds,
        min_visits=5,
        n_permutations=100,
        n_bootstrap=30,
        semantic_backend="tfidf",
        seed=19,
    )
    for model_name in ("M0", "M1", "M2", "M4"):
        assert res.models[model_name].p_value > 0.05


def test_variance_shares_sum_to_one_and_track_place_strength():
    cfg_low = GeneratorConfig(
        n_places=30,
        n_users=30,
        mean_visits_per_user=12.0,
        category_strength=1.0,
        group_strength=0.5,
        place_strength=0.0,
        user_strength=0.2,
        noise_sd=0.2,
    )
    cfg_high = GeneratorConfig(
        n_places=30,
        n_users=30,
        mean_visits_per_user=12.0,
        category_strength=1.0,
        group_strength=0.5,
        place_strength=2.0,
        user_strength=0.2,
        noise_sd=0.2,
    )
    v_low = compute_variance_decomposition(generate_dataset(cfg_low, seed=3))
    v_high = compute_variance_decomposition(generate_dataset(cfg_high, seed=3))

    for var in (v_low, v_high):
        total = (
            var.group_share
            + var.type_within_group_share
            + var.place_within_type_share
            + var.residual_share
        )
        assert total == pytest.approx(1.0, abs=1e-9)
        assert var.type_total_share == pytest.approx(
            var.group_share + var.type_within_group_share, abs=1e-9
        )

    assert v_high.place_within_type_share > v_low.place_within_type_share + 0.15


def test_evaluation_determinism_given_seed():
    cfg = GeneratorConfig(
        n_places=24, n_users=25, mean_visits_per_user=10.0, category_strength=1.5
    )
    ds_a = generate_dataset(cfg, seed=42)
    ds_b = generate_dataset(cfg, seed=42)
    res_a = evaluate_dataset(
        ds_a,
        min_visits=5,
        n_permutations=40,
        n_bootstrap=30,
        semantic_backend="tfidf",
        seed=42,
    )
    res_b = evaluate_dataset(
        ds_b,
        min_visits=5,
        n_permutations=40,
        n_bootstrap=30,
        semantic_backend="tfidf",
        seed=42,
    )
    assert res_a == res_b


def test_sweep_and_outputs_in_tmp_dir(tmp_path):
    rows = run_sweep(
        geo_correlations=(0.0, 1.0),
        category_strengths=(0.0, 1.0),
        place_strengths=(0.0,),
        seeds=(0,),
        n_permutations=25,
        n_bootstrap=20,
        semantic_backend="tfidf",
        min_visits=4,
        base_kwargs={"n_places": 24, "n_users": 25, "mean_visits_per_user": 10.0},
    )
    assert len(rows) == 4
    csv_file = tmp_path / "experiments.csv"
    write_experiments_csv(rows, csv_file)
    with csv_file.open("r", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    assert len(reader) == 4

    summary = format_summary_markdown(rows)
    assert "Expected Sanity Checks" in summary
    assert "Informative Findings" in summary
