# Independent Review of Proximity Validation Work

Scope: Skeptical review of [baseline.py](proximity_validation/baseline.py), [generate.py](proximity_validation/generate.py), [distances.py](proximity_validation/distances.py), [evaluate.py](proximity_validation/evaluate.py), [run_experiments.py](proximity_validation/run_experiments.py), their unit tests under `tests/research_proximity/`, and prior conclusions in [NOTES.md](NOTES.md) and [summary.md](results/summary.md).

---

## 1. Real Bugs Found and Fixed (With Regression Tests)

1. **Corpus-IDF Leakage & Asymmetric `semantic_name` Matrix (`distances.py`):**
   - **Bug:** `semantic_name_similarity(a, b, backend="tfidf")` fit `TfidfVectorizer` on `[a.name, b.name]` alone, whereas `pairwise_matrix(places, dimension="semantic_name", semantic_backend="tfidf")` fit a single corpus-wide `TfidfVectorizer` across all `N` places. As a result, `pairwise_matrix(places, "semantic_name")` disagreed with both `semantic_name_similarity` and `pairwise_matrix(places, "composite", weights={"semantic_name": 1.0})`. In `evaluate.py`, this also allowed held-out place names to alter training-fold IDF weights.
   - **Fix:** Implemented vectorized exact 2-document TF-IDF (`_tfidf_pairwise_2doc_matrix` using `CountVectorizer` + exact pairwise IDF weights `1.0` for shared 3-grams and `1 + ln(3/2)` for single-doc 3-grams), matching pairwise `semantic_name_similarity` to `0.00e+00` without cross-place IDF leakage.
   - **Test added:** `test_semantic_matrix_matches_pairwise_and_composite_weights` in `tests/research_proximity/test_distances.py`.

2. **Silent String Coercion in `composite_similarity(..., weights={'cult': ...})` (`distances.py`):**
   - **Bug:** `_dimension_score` passed any truthy `cultural_tags` directly to `list(c1)`, so a malformed string like `"christian"` was silently split into characters `['c', 'h', 'r', ...]` instead of raising `ValueError`.
   - **Fix:** Validated non-`None` tag containers with `_extract_tags` inside `_dimension_score`.
   - **Test added:** `test_composite_rejects_string_cultural_tags` in `tests/research_proximity/test_distances.py`.

3. **Misleading `p_value = 1.0` Sentinel When M5 Permutations Are Skipped (`evaluate.py`):**
   - **Bug:** Passing `permutation_include_m5=False` (used in `run_experiments.py` sweeps for speed) populated `ModelMetrics` for `M5` with synthetic `p_value=1.0`, `null_mean=0.0`, and `null_sd=0.0` instead of `None`, making untested models look like tested nulls.
   - **Fix:** Changed `ModelMetrics.p_value`, `null_mean`, and `null_sd` to `Optional[float]` returning `None` when skipped.
   - **Test added:** `test_skipped_m5_permutation_sets_none_instead_of_fake_p_value` in `tests/research_proximity/test_evaluate.py`.

4. **Missing Generator Parameter Validation & Incomplete Test Assertions (`generate.py`, `tests/`):**
   - **Bug in `generate.py`:** `GeneratorConfig.__post_init__` did not reject negative `city_spread_deg` or `horizon_days < 1`.
   - **Hollow/weak test coverage:** `test_strength_parameters_drive_between_type_and_between_place_variance` (`test_generate.py`) only asserted between-type variance and never verified `place_strength`; `test_matrices_are_symmetric` (`test_baseline.py`) did not spot-check off-diagonal `default` values against an all-ones dummy; `test_zero_effect_strengths_do_not_beat_permutation_null` (`test_evaluate.py`) did not check `null_mean` proximity to zero or check `M5`.
   - **Fix:** Added validation in `GeneratorConfig.__post_init__` and added the missing assertions in all three test modules.

5. **Overstated Claim Corrected (`run_experiments.py` summary generation):**
   - **Claim mismatch:** Earlier text stated Ridge (`M4`) and NNLS (`M5`) down-weight noise dimensions *"without overfitting"*. In `summary.md`, `M4` (`0.212`) and `M5` (`0.225`) beat the hand-weighted repo `M3` (`0.092`) at `geo_corr=0.0, cat_str=1.0, place_str=0.0`, **but still trail univariate `M1` (`0.316`)** because estimating five regression weights on finite samples introduces variance across correlated features (`semantic_name` vs. `category_tree`).
   - **Fix:** Replaced that claim in `run_experiments.py` with the exact comparison against both `M3` and `M1`.

---

## 2. Structural & Methodological Limitations (Not Code Bugs, Must Be Stated)

1. **Shared Taxonomy & Template-Name Circularity:**
   `category_tree_similarity` scores pairs over the same 3-level tree (`root -> group -> leaf`) used by `generate.py` to generate logits. In addition, synthetic names like `"St. Mary's Church"` embed the literal category noun in the string, giving `M2` (`semantic_name`) positive Spearman (~`0.10–0.21`) solely through lexical category leakage rather than independent semantic name signal, and masking `category_tree` importance during single-feature ablation.
2. **Place-Level Bootstrap Independence Approximation:**
   To keep runtime under ~2–3 seconds per dataset and avoid upward in-sample bias, `_bootstrap_fold_cis` resamples the `M` already-evaluated per-place LPO fold Spearman correlations rather than refitting all `M` LPO folds on each bootstrap resample. Because every pair `(i, j)` participates in both fold `i` and fold `j`, the `M` fold scores are positively dependent, so reported 95% CIs are slightly narrower than full bootstrap refits.
3. **Finite-Sample Inflation in Unadjusted Nested SS (`Var(Place|Type)`):**
   With mean ~`5.3` visits per place (and some places with only `5` or fewer visits), sample place means absorb visit-level estimation noise. Consequently, `compute_variance_decomposition` reports `var_place_within_type_share ≈ 0.077` even when `place_strength = 0.0` (consistent with finite-sample `(N_places - N_types) / (N_visits - 1)` degrees-of-freedom noise rather than true place effects).
4. **Leave-Place-Out Pooling Pitfall Documented & Resolved:**
   Naive concatenation of out-of-fold predictions (`y_mean[p] + x_test @ coef`) across folds caused negative null Spearman (`-0.16` to `-0.28`) because excluding place `p` shifts `mean(y_train)` inversely to `p`'s average JSD. Evaluating Spearman per fold (`mean_fold_spearman`) eliminates this intercept artifact (`null_mean ≈ 0.0`).
