"""Evaluation pipeline for place-level proximity, weight learning, and ANOVA decomposition."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreams_app.core.extra.place_emotion_signature import (  # noqa: E402
    CHIME_DIMENSIONS,
    PlaceEmotionSignature,
    build_place_signature,
)
from research.proximity_validation.distances import (  # noqa: E402
    REPO_DEFAULT_WEIGHTS,
    pairwise_dimension_matrices,
    pairwise_matrix,
)
from research.proximity_validation.generate import (  # noqa: E402
    PlaceRecord,
    SyntheticDataset,
    SyntheticVisit,
)

MODEL_FEATURES: Tuple[str, ...] = (
    "geo",
    "category_tree",
    "semantic_name",
    "ling",
    "cult",
)


@dataclass(frozen=True)
class ModelMetrics:
    """Out-of-fold evaluation summary for one proximity model."""

    name: str
    spearman: float
    r2: Optional[float]
    ci_low: float
    ci_high: float
    p_value: Optional[float]
    null_mean: Optional[float]
    null_sd: Optional[float]


@dataclass(frozen=True)
class AblationResult:
    """Leave-one-feature-out impact relative to full M4 Ridge."""

    dropped_dimension: str
    spearman: float
    r2: float
    delta_spearman: float
    delta_r2: float


@dataclass(frozen=True)
class VarianceDecompositionResult:
    """Nested visit-level CHIME ANOVA decomposition vs planted generator strengths."""

    n_visits: int
    n_places: int
    n_types: int
    n_groups: int
    group_share: float
    type_within_group_share: float
    type_total_share: float
    place_within_type_share: float
    residual_share: float
    planted_group_share: float
    planted_category_share: float
    planted_place_share: float
    planted_residual_share: float


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation report on a synthetic dataset."""

    n_eligible_places: int
    n_pairs: int
    models: Dict[str, ModelMetrics]
    m4_weights: Dict[str, float]
    m5_weights: Dict[str, float]
    ablations: Dict[str, AblationResult]
    variance: VarianceDecompositionResult


@dataclass(frozen=True)
class PairDataset:
    """Precomputed pair indices, dimension matrix, and target vector."""

    places: List[PlaceRecord]
    profiles: np.ndarray
    jsd_matrix: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    features: Dict[str, np.ndarray]
    m3_composite: np.ndarray
    target: np.ndarray


def build_eligible_place_profiles(
    places: Sequence[PlaceRecord],
    visits: Sequence[SyntheticVisit],
    min_visits: int = 5,
) -> Tuple[List[PlaceRecord], Dict[str, PlaceEmotionSignature], np.ndarray]:
    """Build per-place CHIME mean signatures for places with >= min_visits visits."""
    visits_by_place: Dict[str, List[Dict[str, float]]] = {}
    for visit in visits:
        visits_by_place.setdefault(visit.place_id, []).append(visit.chime)

    eligible: List[PlaceRecord] = []
    signatures: Dict[str, PlaceEmotionSignature] = {}
    vectors: List[List[float]] = []

    for place in sorted(places, key=lambda p: p.id):
        place_visits = visits_by_place.get(place.id, [])
        if len(place_visits) < min_visits:
            continue
        sig = build_place_signature(place.id, place_visits)
        eligible.append(place)
        signatures[place.id] = sig
        vectors.append([sig.chime_vector[dim] for dim in CHIME_DIMENSIONS])

    if not vectors:
        return [], {}, np.zeros((0, len(CHIME_DIMENSIONS)), dtype=float)
    return eligible, signatures, np.asarray(vectors, dtype=float)


def emotional_distance(profile_a: Sequence[float], profile_b: Sequence[float]) -> float:
    """Return Jensen-Shannon distance between two CHIME probability distributions."""
    vec_a = np.asarray(profile_a, dtype=float)
    vec_b = np.asarray(profile_b, dtype=float)
    if vec_a.shape != vec_b.shape or vec_a.ndim != 1:
        raise ValueError("CHIME profiles must be 1D vectors of matching length")
    if np.any(vec_a < 0.0) or np.any(vec_b < 0.0):
        raise ValueError("CHIME profiles must be non-negative")
    sum_a = float(np.sum(vec_a))
    sum_b = float(np.sum(vec_b))
    if sum_a <= 0.0 or sum_b <= 0.0:
        raise ValueError("CHIME profiles must have positive total mass")
    norm_a = vec_a / sum_a
    norm_b = vec_b / sum_b
    if np.allclose(norm_a, norm_b, atol=1e-15, rtol=0.0):
        return 0.0
    dist = float(jensenshannon(norm_a, norm_b, base=2.0))
    return 0.0 if math.isnan(dist) else float(min(1.0, max(0.0, dist)))


def compute_jsd_matrix(profiles: np.ndarray) -> np.ndarray:
    """Compute pairwise Jensen-Shannon distance matrix across place profiles."""
    n = len(profiles)
    jsd_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = emotional_distance(profiles[i], profiles[j])
            jsd_mat[i, j] = dist
            jsd_mat[j, i] = dist
    return jsd_mat


def _rowwise_rank_stats(
    mat_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered ranks, row norms, and constant-row mask for a 2D array."""
    is_const = np.ptp(mat_2d, axis=1) == 0.0
    ranks = rankdata(mat_2d, axis=1)
    centered = ranks - np.mean(ranks, axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    is_const = is_const | (norms <= 1e-15)
    return centered, norms, is_const


def _per_fold_spearman(pred_2d: np.ndarray, y_2d: np.ndarray) -> np.ndarray:
    """Return length-M array of Spearman correlations, one per held-out place fold."""
    dx, nx, cx = _rowwise_rank_stats(pred_2d)
    dy, ny, cy = _rowwise_rank_stats(y_2d)
    bad = cx | cy
    denom = np.maximum(nx * ny, 1e-15)
    corrs = np.where(bad, 0.0, np.sum(dx * dy, axis=1) / denom)
    return np.clip(corrs, -1.0, 1.0)


def mean_fold_spearman(pred_2d: np.ndarray, y_2d: np.ndarray) -> float:
    """Mean Spearman correlation across leave-place-out folds (rows = folds)."""
    return float(np.mean(_per_fold_spearman(pred_2d, y_2d)))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation returning 0.0 when either input is constant."""
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    dx = rx - np.mean(rx)
    dy = ry - np.mean(ry)
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(dx, dy) / denom, -1.0, 1.0))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2 safely for small/constant targets."""
    if len(y_true) < 2:
        return 0.0
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - (ss_res / ss_tot))


def build_pair_dataset(
    places: Sequence[PlaceRecord],
    visits: Sequence[SyntheticVisit],
    min_visits: int = 5,
    semantic_backend: str = "auto",
) -> PairDataset:
    """Assemble eligible place pairs, proximity feature arrays, and target -JSD."""
    eligible, _, profiles = build_eligible_place_profiles(
        places, visits, min_visits=min_visits
    )
    n_places = len(eligible)
    if n_places < 4:
        raise ValueError(
            f"Need at least 4 places with >= {min_visits} visits for leave-place-out CV, got {n_places}"
        )

    dim_mats = pairwise_dimension_matrices(
        eligible,
        dimensions=MODEL_FEATURES,
        semantic_backend=semantic_backend,
    )
    m3_mat = pairwise_matrix(
        eligible,
        dimension="composite",
        weights=REPO_DEFAULT_WEIGHTS,
        semantic_backend=semantic_backend,
    )
    jsd_mat = compute_jsd_matrix(profiles)

    tri_i, tri_j = np.triu_indices(n_places, k=1)
    features = {dim: dim_mats[dim][tri_i, tri_j] for dim in MODEL_FEATURES}
    m3_vec = m3_mat[tri_i, tri_j]
    target = -jsd_mat[tri_i, tri_j]

    return PairDataset(
        places=eligible,
        profiles=profiles,
        jsd_matrix=jsd_mat,
        pair_i=tri_i,
        pair_j=tri_j,
        features=features,
        m3_composite=m3_vec,
        target=target,
    )


def iter_leave_place_out_splits(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_places: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (held_out_place_idx, train_mask, test_mask) with zero place leakage."""
    for p_idx in range(n_places):
        test_mask = (pair_i == p_idx) | (pair_j == p_idx)
        train_mask = ~test_mask
        yield p_idx, train_mask, test_mask


def _prepare_fold_cache(
    x_mat: np.ndarray,
    splits: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute standardized train/test blocks and closed-form Ridge projection."""
    cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    eye = ridge_alpha * np.eye(x_mat.shape[1], dtype=float)
    for _, train_mask, test_mask in splits:
        x_train = x_mat[train_mask]
        x_test = x_mat[test_mask]
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        safe_std = np.where(std <= 1e-12, 1.0, std)
        active = std > 1e-12
        x_tr_s = np.where(active, (x_train - mean) / safe_std, 0.0)
        x_te_s = np.where(active, (x_test - mean) / safe_std, 0.0)
        proj = np.linalg.solve(x_tr_s.T @ x_tr_s + eye, x_tr_s.T)
        cache.append((train_mask, test_mask, x_tr_s, x_te_s, proj))
    return cache


def _predict_lpo_ridge_2d(
    y: np.ndarray,
    fold_cache: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_test_2d, pred_2d, mean_standardized_coefs)."""
    y_rows: List[np.ndarray] = []
    p_rows: List[np.ndarray] = []
    fold_coefs: List[np.ndarray] = []
    for train_mask, test_mask, _, x_te_s, proj in fold_cache:
        y_train = y[train_mask]
        y_mean = float(np.mean(y_train))
        coef = proj @ (y_train - y_mean)
        y_rows.append(y[test_mask])
        p_rows.append(y_mean + x_te_s @ coef)
        fold_coefs.append(coef)
    return np.asarray(y_rows), np.asarray(p_rows), np.mean(fold_coefs, axis=0)


def _predict_lpo_nnls_2d(
    y: np.ndarray,
    fold_cache: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_rows: List[np.ndarray] = []
    p_rows: List[np.ndarray] = []
    fold_coefs: List[np.ndarray] = []
    for train_mask, test_mask, x_tr_s, x_te_s, _ in fold_cache:
        y_train = y[train_mask]
        y_mean = float(np.mean(y_train))
        coef, _ = nnls(x_tr_s, y_train - y_mean)
        y_rows.append(y[test_mask])
        p_rows.append(y_mean + x_te_s @ coef)
        fold_coefs.append(coef)
    return np.asarray(y_rows), np.asarray(p_rows), np.mean(fold_coefs, axis=0)


def _permutation_nulls(
    pair_data: PairDataset,
    fold_cache_m4: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    test_index_2d: np.ndarray,
    fixed_rank_stats: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_permutations: int,
    rng: np.random.Generator,
    include_m5: bool = False,
) -> Dict[str, np.ndarray]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    n_places = len(pair_data.places)
    nulls: Dict[str, List[float]] = {name: [] for name in (*fixed_rank_stats, "M4")}
    if include_m5:
        nulls["M5"] = []

    for _ in range(n_permutations):
        perm = rng.permutation(n_places)
        perm_jsd = pair_data.jsd_matrix[np.ix_(perm, perm)]
        y_perm = -perm_jsd[pair_data.pair_i, pair_data.pair_j]
        y_perm_2d = y_perm[test_index_2d]
        dy, ny, cy = _rowwise_rank_stats(y_perm_2d)

        for name, (dx, nx, cx) in fixed_rank_stats.items():
            nulls[name].append(_mean_corr_from_row_stats(dx, nx, cx, dy, ny, cy))

        _, pred_m4_2d, _ = _predict_lpo_ridge_2d(y_perm, fold_cache_m4)
        dx4, nx4, cx4 = _rowwise_rank_stats(pred_m4_2d)
        nulls["M4"].append(_mean_corr_from_row_stats(dx4, nx4, cx4, dy, ny, cy))

        if include_m5:
            _, pred_m5_2d, _ = _predict_lpo_nnls_2d(y_perm, fold_cache_m4)
            dx5, nx5, cx5 = _rowwise_rank_stats(pred_m5_2d)
            nulls["M5"].append(_mean_corr_from_row_stats(dx5, nx5, cx5, dy, ny, cy))

    return {k: np.asarray(v, dtype=float) for k, v in nulls.items()}


def _mean_corr_from_row_stats(
    dx: np.ndarray,
    nx: np.ndarray,
    cx: np.ndarray,
    dy: np.ndarray,
    ny: np.ndarray,
    cy: np.ndarray,
) -> float:
    bad = cx | cy
    denom = np.maximum(nx * ny, 1e-15)
    corrs = np.where(bad, 0.0, np.sum(dx * dy, axis=1) / denom)
    return float(np.mean(np.clip(corrs, -1.0, 1.0)))


def _bootstrap_fold_cis(
    fold_corrs: Mapping[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    first = next(iter(fold_corrs.values()))
    n_places = len(first)
    boot_idx = rng.integers(0, n_places, size=(n_bootstrap, n_places))
    cis: Dict[str, Tuple[float, float]] = {}
    for name, corrs in fold_corrs.items():
        means = np.mean(corrs[boot_idx], axis=1)
        cis[name] = (
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        )
    return cis


def compute_variance_decomposition(
    dataset: SyntheticDataset,
) -> VarianceDecompositionResult:
    """Decompose visit CHIME variance into group, type|group, place|type, and residual."""
    visits = dataset.visits
    if not visits:
        raise ValueError("dataset has no visits")

    y_all = np.asarray(
        [[visit.chime[dim] for dim in CHIME_DIMENSIONS] for visit in visits],
        dtype=float,
    )
    grand_mean = np.mean(y_all, axis=0)
    ss_total = float(np.sum((y_all - grand_mean) ** 2))

    by_group: Dict[str, List[int]] = {}
    by_type: Dict[str, List[int]] = {}
    by_place: Dict[str, List[int]] = {}
    type_to_group: Dict[str, str] = {}
    place_to_type: Dict[str, str] = {}

    for idx, visit in enumerate(visits):
        by_group.setdefault(visit.place_group, []).append(idx)
        by_type.setdefault(visit.place_type, []).append(idx)
        by_place.setdefault(visit.place_id, []).append(idx)
        type_to_group[visit.place_type] = visit.place_group
        place_to_type[visit.place_id] = visit.place_type

    group_means = {g: np.mean(y_all[idxs], axis=0) for g, idxs in by_group.items()}
    type_means = {t: np.mean(y_all[idxs], axis=0) for t, idxs in by_type.items()}
    place_means = {p: np.mean(y_all[idxs], axis=0) for p, idxs in by_place.items()}

    ss_group = float(
        sum(
            len(idxs) * np.sum((group_means[g] - grand_mean) ** 2)
            for g, idxs in by_group.items()
        )
    )
    ss_type_within_group = float(
        sum(
            len(idxs) * np.sum((type_means[t] - group_means[type_to_group[t]]) ** 2)
            for t, idxs in by_type.items()
        )
    )
    ss_place_within_type = float(
        sum(
            len(idxs) * np.sum((place_means[p] - type_means[place_to_type[p]]) ** 2)
            for p, idxs in by_place.items()
        )
    )
    ss_residual = float(
        sum(np.sum((y_all[idxs] - place_means[p]) ** 2) for p, idxs in by_place.items())
    )

    denom = ss_total if ss_total > 1e-15 else 1.0
    cfg = dataset.config
    planted_terms = np.array(
        [
            cfg.group_strength**2,
            cfg.category_strength**2,
            cfg.place_strength**2,
            cfg.user_strength**2 + cfg.noise_sd**2,
        ],
        dtype=float,
    )
    planted_sum = float(np.sum(planted_terms))
    planted_shares = planted_terms / planted_sum if planted_sum > 0 else np.zeros(4)

    return VarianceDecompositionResult(
        n_visits=len(visits),
        n_places=len(by_place),
        n_types=len(by_type),
        n_groups=len(by_group),
        group_share=ss_group / denom,
        type_within_group_share=ss_type_within_group / denom,
        type_total_share=(ss_group + ss_type_within_group) / denom,
        place_within_type_share=ss_place_within_type / denom,
        residual_share=ss_residual / denom,
        planted_group_share=float(planted_shares[0]),
        planted_category_share=float(planted_shares[1]),
        planted_place_share=float(planted_shares[2]),
        planted_residual_share=float(planted_shares[3]),
    )


def evaluate_dataset(
    dataset: SyntheticDataset,
    min_visits: int = 5,
    ridge_alpha: float = 1.0,
    n_permutations: int = 500,
    n_bootstrap: int = 200,
    semantic_backend: str = "auto",
    seed: int = 42,
    include_m5: bool = True,
    permutation_include_m5: bool = True,
) -> EvaluationResult:
    """Run leave-place-out CV, ablations, significance tests, and nested ANOVA."""
    pair_data = build_pair_dataset(
        dataset.places,
        dataset.visits,
        min_visits=min_visits,
        semantic_backend=semantic_backend,
    )
    n_places = len(pair_data.places)
    splits = list(
        iter_leave_place_out_splits(pair_data.pair_i, pair_data.pair_j, n_places)
    )
    test_index_2d = np.asarray([np.flatnonzero(tm) for _, _, tm in splits], dtype=int)
    y_2d = pair_data.target[test_index_2d]

    x_full = np.column_stack([pair_data.features[dim] for dim in MODEL_FEATURES])
    fold_cache_m4 = _prepare_fold_cache(x_full, splits, ridge_alpha=ridge_alpha)

    fixed_vecs = {
        "M0": pair_data.features["geo"],
        "M1": pair_data.features["category_tree"],
        "M2": pair_data.features["semantic_name"],
        "M3": pair_data.m3_composite,
    }

    fold_results: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]] = {}
    fixed_rank_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, vec in fixed_vecs.items():
        pred_2d = vec[test_index_2d]
        fold_results[name] = (y_2d, pred_2d, None)
        fixed_rank_stats[name] = _rowwise_rank_stats(pred_2d)

    yt4_2d, yp4_2d, m4_coefs = _predict_lpo_ridge_2d(pair_data.target, fold_cache_m4)
    fold_results["M4"] = (yt4_2d, yp4_2d, safe_r2(yt4_2d.ravel(), yp4_2d.ravel()))

    m5_weights: Dict[str, float] = {}
    if include_m5:
        yt5_2d, yp5_2d, m5_coefs = _predict_lpo_nnls_2d(pair_data.target, fold_cache_m4)
        fold_results["M5"] = (yt5_2d, yp5_2d, safe_r2(yt5_2d.ravel(), yp5_2d.ravel()))
        m5_weights = {dim: float(w) for dim, w in zip(MODEL_FEATURES, m5_coefs)}

    fold_corrs = {
        name: _per_fold_spearman(yp_2d, yt_2d)
        for name, (yt_2d, yp_2d, _) in fold_results.items()
    }
    rng = np.random.default_rng(seed)
    nulls = _permutation_nulls(
        pair_data,
        fold_cache_m4,
        test_index_2d,
        fixed_rank_stats,
        n_permutations=n_permutations,
        rng=rng,
        include_m5=include_m5 and permutation_include_m5,
    )
    cis = _bootstrap_fold_cis(fold_corrs, n_bootstrap=n_bootstrap, rng=rng)

    models: Dict[str, ModelMetrics] = {}
    for name, (_, _, r2_val) in fold_results.items():
        rho = float(np.mean(fold_corrs[name]))
        null_arr = nulls.get(name)
        if null_arr is None:
            p_val: Optional[float] = None
            null_mean: Optional[float] = None
            null_sd: Optional[float] = None
        else:
            p_val = float((1 + np.sum(null_arr >= rho)) / (len(null_arr) + 1))
            null_mean = float(np.mean(null_arr))
            null_sd = float(np.std(null_arr))
        models[name] = ModelMetrics(
            name=name,
            spearman=rho,
            r2=r2_val,
            ci_low=cis[name][0],
            ci_high=cis[name][1],
            p_value=p_val,
            null_mean=null_mean,
            null_sd=null_sd,
        )

    m4_spearman = models["M4"].spearman
    m4_r2 = models["M4"].r2 or 0.0
    ablations: Dict[str, AblationResult] = {}
    for drop_idx, drop_dim in enumerate(MODEL_FEATURES):
        keep_cols = [idx for idx in range(len(MODEL_FEATURES)) if idx != drop_idx]
        x_sub = x_full[:, keep_cols]
        sub_cache = _prepare_fold_cache(x_sub, splits, ridge_alpha=ridge_alpha)
        yt_sub, yp_sub, _ = _predict_lpo_ridge_2d(pair_data.target, sub_cache)
        sp_sub = mean_fold_spearman(yp_sub, yt_sub)
        r2_sub = safe_r2(yt_sub.ravel(), yp_sub.ravel())
        ablations[drop_dim] = AblationResult(
            dropped_dimension=drop_dim,
            spearman=sp_sub,
            r2=r2_sub,
            delta_spearman=sp_sub - m4_spearman,
            delta_r2=r2_sub - m4_r2,
        )

    return EvaluationResult(
        n_eligible_places=n_places,
        n_pairs=len(pair_data.target),
        models=models,
        m4_weights={dim: float(w) for dim, w in zip(MODEL_FEATURES, m4_coefs)},
        m5_weights=m5_weights,
        ablations=ablations,
        variance=compute_variance_decomposition(dataset),
    )
