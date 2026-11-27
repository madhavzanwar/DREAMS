# Proximity Validation Research Log

Branch: `research/proximity-validation` (branched from `upstream/main` at `cf19c15`)

## Goal
Validate whether multi-dimensional place proximity predicts emotional similarity in DREAMS, focusing on three questions:
1. Do categorical and semantic/linguistic/cultural proximity predict emotional similarity better than geographic distance alone?
2. Can the `composite_proximity` dimension weights (`geo=0.3`, `cat=0.4`, `ling=0.15`, `cult=0.15` in `dreams_app/core/extra/proximity_calculator.py`) be learned from data instead of hand-set?
3. How much of a place's CHIME emotional profile is place-specific ("this particular church") versus shared across its place type ("any church"), since `dreams_app/core/extra/place_emotion_signature.py` currently builds one `PlaceEmotionSignature` per `place_type`?

## Baseline Repo Facts Verified
- `dreams_app/core/extra/proximity_calculator.py`: `composite_proximity` uses default weights `{'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}` and renormalizes over available dimensions.
- `dreams_app/core/extra/place_emotion_signature.py`: `build_place_signature(place_type, chime_results)` aggregates CHIME vectors (`Connectedness`, `Hope`, `Identity`, `Meaning`, `Empowerment`) keyed only by `place_type`; `chime_proximity` computes cosine similarity between two `PlaceEmotionSignature` vectors.
- `dreams_app/core/extra/location_extractor.py`: `reverse_geocode` extracts `display_name`, `place_category` (OSM `type`), `place_type` (OSM `category`), and `address`; `format_location_text` strips city/state/country before 384-dim `all-MiniLM-L6-v2` embedding.

## Commands Run
- `git fetch upstream && git checkout -b research/proximity-validation upstream/main` (at `cf19c15`).
- Read docs and modules for Task 1:
  - `README.md`, `SETUP.md`, `ARCHITECTURE.md`, `LOCATION_PROXIMITY_SUMMARY.md`, `location_proximity/README.md`, `location_proximity/RESEARCH.md`
  - `dreams_app/core/extra/proximity_calculator.py`, `dreams_app/core/extra/place_emotion_signature.py`, `dreams_app/core/extra/proximity_trajectory.py`, `dreams_app/core/graph/time_aware_proximity.py`, `dreams_app/core/graph/emotion_proximity.py`, `dreams_app/core/graph/episode_proximity.py`, `dreams_app/core/graph/place_narrative_bridge.py`
  - Also inspected `dreams_app/location_proximity.py`, `dreams_app/core/extra/location_extractor.py`, `dreams_app/core/extra/places365_classifier.py`, and discussions `#71`, `#273`, `#285`, `#294`
  - `PlaceVisit` is defined at `dreams_app/core/extra/proximity_trajectory.py` (L14–26) with fields `place_type`, `timestamp`, and `chime`.
  - `git grep -n -E "composite_proximity|chime_proximity|build_place_trajectories" -- dreams_app/` confirmed none of these three functions are called outside `dreams_app/core/extra/` and `dreams_app/core/graph/place_narrative_bridge.py`.
- Checked `location_proximity/demo.py`:
  - `python location_proximity/demo.py` failed with `[Errno 2] No such file or directory`.
  - `git log -n 2 --oneline -- location_proximity/` confirms `demo.py` does not exist in git history; only `location_proximity/README.md` and `location_proximity/RESEARCH.md` exist in that directory.
- Environment & dependencies (scratch Python 3.13.9 `.venv` with `--system-site-packages` plus `pymongo`, `Flask-Login`, `pytest-mock`, `pytest-cov`, `mongomock`, `hdbscan`):
  - `python -c "from data_integrity.validator import validate_dependencies; print(validate_dependencies())"` returned `{'valid': False, 'missing': ['wordcloud', 'datasets', 'spacy', 'setfit']}` (those four packages from `requirements.txt` are not installed; none caused failures in the test run below).
- Ran `pytest --ignore=tests/test_places365_classifier.py -v --tb=short` (with `HF_HUB_OFFLINE=1` to avoid downloading `j-hartmann/emotion-english-distilroberta-base`):
  - Collected `514 items`: **`511 passed, 3 skipped, 1 warning in 93.31s (0:01:33)`**.
  - Failures: `0`.
  - Skipped (`3`):
    - `tests/test_fl.py::test_fl_loop` (`MongoDB is not running on lo...`).
    - `tests/test_fl_bookkeeping.py::test_claim_timestamp_is_utc_whatever_the_server_timezone` (`tzset is not available on this platform` on Windows).
    - `tests/test_fl_bookkeeping.py::test_stale_claims_are_released_even_in_a_negative_utc_offset` (`tzset is not available on this platform` on Windows).
  - Note: Running tests generates an untracked `data/` directory at repo root; delete it before committing so generated data is never committed.
- Wrote `research/PROXIMITY_MAP.md` covering sections `(a)`–`(e)` with file/line citations and no proposed solutions.
- Created baseline reproduction script (`research/proximity_validation/baseline.py`, plus package markers `research/__init__.py` and `research/proximity_validation/__init__.py`) and deterministic unit tests (`tests/research_proximity/test_baseline.py`).
- Ran `python research/proximity_validation/baseline.py` on 12 hand-written places (Anchorage, Seattle, Lisbon) and 14 CHIME visits across 5 place types (`cafe`, `church`, `clinic`, `hospital`, `park`), saving output to `research/results/baseline_output.txt`:
  - Example A (`anc_church_recovery` vs `lis_church_se`, `8351.00 km`): `geo=0.000`, `cat=1.000`, `ling=0.000`, `cult=0.200`, `composite_default=0.430` (`geo_only=0.000`, `category_only=1.000`).
  - Example B (`anc_church_recovery` vs `anc_clinic_north`, `0.37 km`): `geo=0.998`, `cat=0.000`, `ling=1.000`, `cult=0.500`, `composite_default=0.524` (`geo_only=0.998`, `category_only=0.000`).
  - Example C (`anc_church_recovery` vs `sea_church_grief`): type pooling assigns both identical signatures (`visits=6`, `Hope=0.533`, `Connectedness=0.550`, `volatility=0.261`, `chime_proximity=1.000`) and prevents `'church'` from appearing in `detect_recovery_anchors(threshold=0.6)`, hiding actual per-place divergence (`anc_church_recovery` Hope `0.867` / volatility `0.035` vs `sea_church_grief` Hope `0.200` / volatility `0.041`, per-place `chime_proximity=0.890`).
  - Minor existing edge behavior observed: `chime_proximity(sig, sig)` for the pooled `'church'` vector returned `1.0000000000000002` (unclamped cosine similarity in `place_emotion_signature.py`), so `tests/research_proximity/test_baseline.py` compares with `pytest.approx(1.0)`.
- Formatted with `black research/ tests/research_proximity/`, verified `flake8 research/ tests/research_proximity/ --count --max-complexity=10 --max-line-length=127 --statistics` returned `0`, and verified `pytest tests/research_proximity/test_baseline.py -v` -> **`5 passed in 0.16s`**.
- Implemented deterministic synthetic data generator (`research/proximity_validation/generate.py`) with:
  - 3-level taxonomy tree (`root` depth 0 -> 5 parent groups depth 1 -> 15 leaf place types depth 2) in `PLACE_TAXONOMY_TREE` / `TAXONOMY_NODES` / `TAXONOMY_DEPTHS`.
  - `GeneratorConfig` (defaults `n_places=60`, `n_cities=4`, `n_users=40`, `mean_visits_per_user=8.0`, `geo_category_correlation=0.0`, `category_strength=1.0`, `group_strength=0.8`, `place_strength=0.8`, `user_strength=0.5`, `noise_sd=0.3`).
  - Seeded ground-truth effect tables (`category_effect`, `group_effect`, `place_effect`, `user_effect`) over `CHIME_DIMENSIONS` (`dreams_app.core.extra.place_emotion_signature`), additive logit + Gaussian noise + softmax CHIME visit generation, and export helpers `export_to_place_visits` (`dreams_app.core.extra.proximity_trajectory.PlaceVisit`) and `export_chime_by_type`.
- Documented the generator and synthetic-data scope limitations in `research/proximity_validation/README.md` (`## Synthetic data`).
- Added 8 deterministic tests in `tests/research_proximity/test_generate.py`.
- Formatted with `black`, ran `flake8 research/ tests/research_proximity/ --count --max-complexity=10 --max-line-length=127 --statistics` (`0` issues), ran `pytest tests/research_proximity/ -v` -> **`13 passed in 0.35s`** (`8` new + `5` baseline), and benchmarked default `generate_dataset(GeneratorConfig(), seed=42)`: `places=60, visits=348, elapsed=0.0131s`.
- Implemented proximity dimension module (`research/proximity_validation/distances.py`) with pure validated functions returning `[0, 1]` and matrix helpers `pairwise_matrix` / `pairwise_dimension_matrices`:
  - `geographic_proximity`: Haversine via `dreams_app.location_proximity.calculate_distance` + `dreams_app.core.extra.proximity_calculator.normalize_geographic_distance`.
  - `category_tree_similarity`: Wu-Palmer similarity over `generate.TAXONOMY_NODES` (`1.0` same type > `0.5` same group > `0.0` different group), explicitly contrasted with the repo's 3-pair `categorical_proximity`.
  - `semantic_name_similarity`: place-name cosine similarity supporting `backend='auto' | 'tfidf' | 'sentence_transformers'` with deterministic char 3-gram TF-IDF fallback (`TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3))`).
  - `linguistic_proximity` & `cultural_proximity`: reuse `linguistic_similarity` and `cultural_similarity` from `dreams_app.core.extra.proximity_calculator`.
  - `composite_similarity`: combines any subset of dimensions and renormalizes over active dimensions identically to `composite_proximity` (reproduced exactly across 6 pairs under `REPO_DEFAULT_WEIGHTS`).
- Added 12 deterministic tests in `tests/research_proximity/test_distances.py` (using `backend='tfidf'` / fallback), formatted with `black`, verified `flake8` returned `0`, and ran `pytest tests/research_proximity/ -v` -> **`25 passed in 0.42s`** (`12` in `test_distances.py`, `8` in `test_generate.py`, `5` in `test_baseline.py`).
- Implemented evaluation pipeline (`research/proximity_validation/evaluate.py`):
  - Aggregates CHIME profiles per `place_id` (`min_visits=5`) via `build_place_signature` and measures emotional distance via `scipy.spatial.distance.jensenshannon(..., base=2.0)`.
  - Evaluates models `M0` (geo), `M1` (category_tree Wu-Palmer), `M2` (semantic_name), `M3` (repo default composite), `M4` (Ridge on standardized features), and `M5` (NNLS with centered target intercept) under strict leave-place-out CV (`iter_leave_place_out_splits`), aggregating Spearman across held-out-place folds to avoid leave-one-out intercept anti-correlation, plus pooled OOF $R^2$, leave-one-feature-out ablations, place-profile permutation nulls (`n_permutations`), place-level bootstrap 95% CIs, and nested visit-level CHIME ANOVA (`group`, `type|group`, `place|type`, `residual`).
  - Default single-run check (`GeneratorConfig()`, `seed=42`, `min_visits=5`, `n_permutations=500`, `n_bootstrap=200`, `semantic_backend='tfidf'`): completed in **`2.173s`** (`44` eligible places, `946` pairs):
    - `M0`: Spearman `-0.018` (CI `[-0.079, 0.035]`, `p=0.6467`, `null=-0.000`)
    - `M1`: Spearman `0.279` (CI `[0.218, 0.345]`, `p=0.0020`, `null=-0.000`)
    - `M2`: Spearman `0.104` (CI `[0.053, 0.150]`, `p=0.0020`, `null=0.002`)
    - `M3`: Spearman `0.056` (CI `[-0.003, 0.104]`, `p=0.0838`, `null=-0.000`)
    - `M4`: Spearman `0.162` (CI `[0.107, 0.218]`, `R^2=0.069`, `p=0.0020`, `null=0.016`), weights `geo=0.0015, category_tree=0.0518, semantic_name=0.0011, ling=-0.0039, cult=0.0058`
    - `M5`: Spearman `0.205` (CI `[0.146, 0.268]`, `R^2=0.073`, `p=0.0020`, `null=0.006`), weights `geo=0.0, category_tree=0.0517, semantic_name=0.0013, ling=0.0, cult=0.0058`
    - Ablation `delta_spearman` from `M4`: dropping `category_tree` drops Spearman by `-0.0814` (vs `+0.0134` geo, `+0.0203` semantic_name, `+0.0066` ling, `-0.0023` cult).
    - Nested ANOVA on `seed=42`: `group=0.252`, `type|group=0.322` (`type_total=0.575`), `place|type=0.269`, `residual=0.156` (planted: `group=0.244`, `category=0.382`, `place=0.244`, `residual=0.130`).
- Implemented sweep script (`research/proximity_validation/run_experiments.py`) across `geo_category_correlation` in `{0, 0.5, 1}`, `category_strength` in `{0, 0.5, 1}`, `place_strength` in `{0, 0.5, 1}` with `5` seeds (`135` runs, `n_permutations=200`, `n_bootstrap=100`, `semantic_backend='tfidf'`), writing `research/results/experiments.csv` and `research/results/summary.md` in **`53.1s`**:
  - Unconfounded (`geo_corr=0.0, cat_str=1.0, place_str=0.0`): mean `M1=0.316` vs `M0=-0.030`, `M4=0.212` (`R^2=0.192`), `M5=0.225`.
  - Confounded (`geo_corr=1.0, cat_str=1.0, place_str=0.0`): `M0` rises to `0.062` (spatial confounding), `M1=0.374`, `M4=0.306` (`R^2=0.244`).
  - Place-specific divergence (`cat_str=1.0`): raising `place_str` from `0.0` to `1.0` increases `Var(Place|Type)` from `0.036` to `0.323` and drops mean `M1` Spearman from `0.343` to `0.201`.
- Added 10 deterministic tests in `tests/research_proximity/test_evaluate.py`; formatted with `black`, `flake8` returned `0`, and `pytest tests/research_proximity/ -v` -> **`35 passed in 2.47s`**.
- Completed independent review (`research/REVIEW.md`) and fixed bugs + added regression tests:
  - Fixed `distances.py::_semantic_name_matrix` so `pairwise_matrix(..., 'semantic_name', semantic_backend='tfidf')` uses exact vectorized 2-doc char-3-gram TF-IDF cosine (`max_diff=0.00e+00` vs pairwise `semantic_name_similarity`, zero cross-place IDF leakage), plus validated non-string `cultural_tags` in `_dimension_score`.
  - Typed `ModelMetrics.p_value`, `null_mean`, `null_sd` in `evaluate.py` as `Optional[float]` (`None` when permutation testing is skipped via `permutation_include_m5=False`).
  - Added `city_spread_deg >= 0` and `horizon_days >= 1` validation in `generate.py`, plus regression tests across `test_baseline.py`, `test_generate.py`, `test_distances.py`, and `test_evaluate.py` (`38 passed in 2.62s` on `tests/research_proximity/`).
  - Regenerated `research/results/experiments.csv` and `research/results/summary.md` (`135` runs in `54.62s`); verified `seed=42` 500-permutation default run (`2.519s`, `M1=0.279`, `M2=0.097`, `M4=0.165`, `M5=0.214`).
- Wrote `research/proximity_validation/README.md`, `research/RESULTS.md`, and `research/PROPOSAL_NOTES.md`.
- Verified `black --check research/ tests/research_proximity/` (`11 files would be left unchanged`), `flake8` (`0` issues), and full suite `pytest --ignore=tests/test_places365_classifier.py -v --tb=short` (with `HF_HUB_OFFLINE=1`): **`549 passed, 3 skipped, 1 warning in 87.62s`** (cleaned up generated `data/`).

## Open Questions
- Are `wordcloud`, `datasets`, `spacy`, or `setfit` needed for upcoming research experiments, or should `research/proximity_validation/` stick strictly to `numpy`, `scipy`, `scikit-learn`, and stdlib (already available)?
- What synthetic/benchmark dataset schema and evaluation protocol should be implemented first under `research/proximity_validation/`?
- What baseline statistical / optimization methods (`scipy`, `scikit-learn`) should be used for weight learning and place-vs-type variance decomposition?
