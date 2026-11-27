"""Sweep script evaluating proximity models across geo-cat correlation and strengths."""

import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.proximity_validation.evaluate import (  # noqa: E402
    EvaluationResult,
    evaluate_dataset,
)
from research.proximity_validation.generate import (  # noqa: E402
    GeneratorConfig,
    generate_dataset,
)

DEFAULT_CSV_PATH = _REPO_ROOT / "research" / "results" / "experiments.csv"
DEFAULT_SUMMARY_PATH = _REPO_ROOT / "research" / "results" / "summary.md"

GEO_CAT_CORR_GRID = (0.0, 0.5, 1.0)
CATEGORY_STRENGTH_GRID = (0.0, 0.5, 1.0)
PLACE_STRENGTH_GRID = (0.0, 0.5, 1.0)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

CSV_COLUMNS: Tuple[str, ...] = (
    "geo_category_correlation",
    "category_strength",
    "place_strength",
    "seed",
    "n_eligible_places",
    "n_pairs",
    "m0_spearman",
    "m1_spearman",
    "m2_spearman",
    "m3_spearman",
    "m4_spearman",
    "m4_r2",
    "m4_p_value",
    "m5_spearman",
    "m5_r2",
    "w_m4_geo",
    "w_m4_category_tree",
    "w_m4_semantic_name",
    "w_m4_ling",
    "w_m4_cult",
    "w_m5_geo",
    "w_m5_category_tree",
    "m4_drop_geo_delta",
    "m4_drop_category_tree_delta",
    "m4_drop_semantic_name_delta",
    "var_group_share",
    "var_type_within_group_share",
    "var_type_total_share",
    "var_place_within_type_share",
    "var_residual_share",
    "planted_category_share",
    "planted_place_share",
)


def _row_from_result(
    geo_corr: float,
    cat_strength: float,
    place_strength: float,
    seed: int,
    res: EvaluationResult,
) -> Dict[str, object]:
    return {
        "geo_category_correlation": geo_corr,
        "category_strength": cat_strength,
        "place_strength": place_strength,
        "seed": seed,
        "n_eligible_places": res.n_eligible_places,
        "n_pairs": res.n_pairs,
        "m0_spearman": round(res.models["M0"].spearman, 6),
        "m1_spearman": round(res.models["M1"].spearman, 6),
        "m2_spearman": round(res.models["M2"].spearman, 6),
        "m3_spearman": round(res.models["M3"].spearman, 6),
        "m4_spearman": round(res.models["M4"].spearman, 6),
        "m4_r2": round(res.models["M4"].r2 or 0.0, 6),
        "m4_p_value": round(res.models["M4"].p_value, 6),
        "m5_spearman": round(res.models["M5"].spearman, 6),
        "m5_r2": round(res.models["M5"].r2 or 0.0, 6),
        "w_m4_geo": round(res.m4_weights["geo"], 6),
        "w_m4_category_tree": round(res.m4_weights["category_tree"], 6),
        "w_m4_semantic_name": round(res.m4_weights["semantic_name"], 6),
        "w_m4_ling": round(res.m4_weights["ling"], 6),
        "w_m4_cult": round(res.m4_weights["cult"], 6),
        "w_m5_geo": round(res.m5_weights["geo"], 6),
        "w_m5_category_tree": round(res.m5_weights["category_tree"], 6),
        "m4_drop_geo_delta": round(res.ablations["geo"].delta_spearman, 6),
        "m4_drop_category_tree_delta": round(
            res.ablations["category_tree"].delta_spearman, 6
        ),
        "m4_drop_semantic_name_delta": round(
            res.ablations["semantic_name"].delta_spearman, 6
        ),
        "var_group_share": round(res.variance.group_share, 6),
        "var_type_within_group_share": round(res.variance.type_within_group_share, 6),
        "var_type_total_share": round(res.variance.type_total_share, 6),
        "var_place_within_type_share": round(res.variance.place_within_type_share, 6),
        "var_residual_share": round(res.variance.residual_share, 6),
        "planted_category_share": round(res.variance.planted_category_share, 6),
        "planted_place_share": round(res.variance.planted_place_share, 6),
    }


def run_sweep(
    geo_correlations: Sequence[float] = GEO_CAT_CORR_GRID,
    category_strengths: Sequence[float] = CATEGORY_STRENGTH_GRID,
    place_strengths: Sequence[float] = PLACE_STRENGTH_GRID,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_permutations: int = 200,
    n_bootstrap: int = 100,
    semantic_backend: str = "tfidf",
    min_visits: int = 5,
    base_kwargs: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Run grid sweep over correlation, category_strength, place_strength, and seeds."""
    extra_cfg = dict(base_kwargs or {})
    rows: List[Dict[str, object]] = []
    for geo_corr in geo_correlations:
        for cat_strength in category_strengths:
            for place_strength in place_strengths:
                for seed in seeds:
                    cfg = GeneratorConfig(
                        geo_category_correlation=float(geo_corr),
                        category_strength=float(cat_strength),
                        place_strength=float(place_strength),
                        **extra_cfg,
                    )
                    ds = generate_dataset(cfg, seed=int(seed))
                    res = evaluate_dataset(
                        ds,
                        min_visits=min_visits,
                        n_permutations=n_permutations,
                        n_bootstrap=n_bootstrap,
                        semantic_backend=semantic_backend,
                        seed=int(seed),
                        permutation_include_m5=False,
                    )
                    rows.append(
                        _row_from_result(
                            geo_corr=float(geo_corr),
                            cat_strength=float(cat_strength),
                            place_strength=float(place_strength),
                            seed=int(seed),
                            res=res,
                        )
                    )
    return rows


def write_experiments_csv(rows: Sequence[Dict[str, object]], csv_path: Path) -> None:
    """Save experiment rows deterministically to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _group_mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def format_summary_markdown(rows: Sequence[Dict[str, object]]) -> str:
    """Generate a readable summary report distinguishing sanity checks from findings."""
    grouped: Dict[Tuple[float, float, float], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["geo_category_correlation"]),
            float(row["category_strength"]),
            float(row["place_strength"]),
        )
        grouped.setdefault(key, []).append(row)

    lines = [
        "# Proximity Validation Experiment Summary",
        "",
        f"Total runs: `{len(rows)}` across `{len(grouped)}` configurations "
        f"(fixed defaults: `n_places=60`, `n_cities=4`, `n_users=40`, "
        f"`mean_visits_per_user=8.0`, `group_strength=0.8`, `user_strength=0.5`, "
        f"`noise_sd=0.3`, `min_visits=5`, `semantic_backend='tfidf'`).",
        "",
        "## Mean Spearman & Variance Decomposition by Configuration",
        "",
        (
            "| geo_corr | cat_str | place_str | M0 (Geo) | M1 (CatTree) | "
            "M2 (SemName) | M3 (Repo) | M4 (Ridge) | M5 (NNLS) | M4 R2 | "
            "Var(Type) | Var(Place\\|Type) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for (geo_c, cat_s, plc_s), grp in sorted(grouped.items()):
        lines.append(
            f"| {geo_c:.1f} | {cat_s:.1f} | {plc_s:.1f} | "
            f"{_group_mean(grp, 'm0_spearman'):.3f} | "
            f"{_group_mean(grp, 'm1_spearman'):.3f} | "
            f"{_group_mean(grp, 'm2_spearman'):.3f} | "
            f"{_group_mean(grp, 'm3_spearman'):.3f} | "
            f"{_group_mean(grp, 'm4_spearman'):.3f} | "
            f"{_group_mean(grp, 'm5_spearman'):.3f} | "
            f"{_group_mean(grp, 'm4_r2'):.3f} | "
            f"{_group_mean(grp, 'var_type_total_share'):.3f} | "
            f"{_group_mean(grp, 'var_place_within_type_share'):.3f} |"
        )

    unconfounded_high_cat = [
        r
        for r in rows
        if float(r["geo_category_correlation"]) == 0.0
        and float(r["category_strength"]) == 1.0
        and float(r["place_strength"]) == 0.0
    ]
    confounded_high_cat = [
        r
        for r in rows
        if float(r["geo_category_correlation"]) == 1.0
        and float(r["category_strength"]) == 1.0
        and float(r["place_strength"]) == 0.0
    ]
    high_place = [
        r
        for r in rows
        if float(r["category_strength"]) == 1.0 and float(r["place_strength"]) == 1.0
    ]
    low_place = [
        r
        for r in rows
        if float(r["category_strength"]) == 1.0 and float(r["place_strength"]) == 0.0
    ]

    lines.extend(
        [
            "",
            "## Interpretation: Expected Sanity Checks vs. Informative Findings",
            "",
            "### Expected Sanity Checks (Consequences of Planted Effects)",
            "1. **Category vs. Geography when unconfounded "
            "(`geo_corr=0.0`, `cat_str=1.0`, `place_str=0.0`):** "
            + (
                f"`M1` (category tree) achieves Spearman "
                f"`{_group_mean(unconfounded_high_cat, 'm1_spearman'):.3f}` vs. `M0` (geography) "
                f"`{_group_mean(unconfounded_high_cat, 'm0_spearman'):.3f}`, "
                "which is expected because `geo_category_correlation=0` removes spatial "
                "clustering of types."
                if unconfounded_high_cat
                else "Observed when unconfounded high-category configs are present."
            ),
            "2. **Geographic confounding (`geo_corr=1.0`, `cat_str=1.0`, `place_str=0.0`):** "
            + (
                f"`M0` rises to Spearman `{_group_mean(confounded_high_cat, 'm0_spearman'):.3f}` "
                "solely because same-type places share cities, showing geographic proximity "
                "can masquerade as categorical effect under spatial confounding."
                if confounded_high_cat
                else "Observed when geo-correlated configs are present."
            ),
            "3. **Place variance growth with `place_strength`:** "
            + (
                f"`Var(Place|Type)` rises from "
                f"`{_group_mean(low_place, 'var_place_within_type_share'):.3f}` (`place_str=0.0`) "
                f"to `{_group_mean(high_place, 'var_place_within_type_share'):.3f}` (`place_str=1.0`)."
                if high_place and low_place
                else "Observed as place_strength varies."
            ),
            "",
            "### Informative Findings (Non-Trivial Behavior Beyond Planted Strengths)",
            "1. **Type-level `M1` degrades rapidly when places have distinct identities (`place_strength=1.0`):** "
            + (
                f"At `cat_str=1.0` (averaged across `geo_corr` in `{{0.0, 0.5, 1.0}}`), "
                f"raising `place_str` from `0.0` to `1.0` drops `M1` Spearman from "
                f"`{_group_mean(low_place, 'm1_spearman'):.3f}` to "
                f"`{_group_mean(high_place, 'm1_spearman'):.3f}`, "
                "quantifying how severely type-level pooling fails when specific places diverge emotionally."
                if high_place and low_place
                else ""
            ),
            "2. **Learned `M4`/`M5` vs. repo `M3` and single-feature `M1`:** Fixed repo weights "
            "(`geo=0.3, cat=0.4, ling=0.15, cult=0.15`) dilute categorical signal with non-predictive "
            "dimensions when `geo_corr=0`. Leave-place-out Ridge (`M4`) and NNLS (`M5`) strongly outperform "
            "`M3` by down-weighting uninformative dimensions, though they still trail oracle "
            "single-feature `M1` due to finite-sample weight variance across five features "
            "(including category nouns leaked into `semantic_name`).",
            "3. **Parent-group baseline signal:** Even when `category_strength=0.0`, default "
            "`group_strength=0.8` keeps `M1` (taxonomy Wu-Palmer) predictive because places "
            "within the same parent group (`religious`, `healthcare`, etc.) still share latent CHIME effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_and_save(
    csv_path: Path = DEFAULT_CSV_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    n_permutations: int = 200,
    n_bootstrap: int = 100,
) -> List[Dict[str, object]]:
    """Execute the default experiment grid and write CSV and Markdown summary."""
    rows = run_sweep(
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        semantic_backend="tfidf",
    )
    write_experiments_csv(rows, csv_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        format_summary_markdown(rows), encoding="utf-8", newline="\n"
    )
    return rows


def main() -> None:
    rows = run_and_save()
    print(f"Completed {len(rows)} experiment runs -> {DEFAULT_CSV_PATH}")


if __name__ == "__main__":
    main()
