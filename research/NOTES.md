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

## Open Questions
- Are `wordcloud`, `datasets`, `spacy`, or `setfit` needed for upcoming research experiments, or should `research/proximity_validation/` stick strictly to `numpy`, `scipy`, `scikit-learn`, and stdlib (already available)?
- What synthetic/benchmark dataset schema and evaluation protocol should be implemented first under `research/proximity_validation/`?
- What baseline statistical / optimization methods (`scipy`, `scikit-learn`) should be used for weight learning and place-vs-type variance decomposition?
