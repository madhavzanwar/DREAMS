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

## Open Questions
- Are `wordcloud`, `datasets`, `spacy`, or `setfit` needed for upcoming research experiments, or should `research/proximity_validation/` stick strictly to `numpy`, `scipy`, `scikit-learn`, and stdlib (already available)?
- What synthetic/benchmark dataset schema and evaluation protocol should be implemented first under `research/proximity_validation/`?
- What baseline statistical / optimization methods (`scipy`, `scikit-learn`) should be used for weight learning and place-vs-type variance decomposition?
