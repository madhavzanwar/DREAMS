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

## Open Questions
- What synthetic/benchmark dataset format and evaluation protocol should be implemented first under `research/proximity_validation/`?
- What baseline statistical / optimization methods (`scipy`, `scikit-learn`) should be used for weight learning and place-vs-type variance decomposition?
