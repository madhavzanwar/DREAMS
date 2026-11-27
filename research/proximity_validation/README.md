# Multi-Dimensional Place Proximity Validation

Experimental code on branch `research/proximity-validation` testing three questions for [KathiraveluLab/DREAMS](https://github.com/KathiraveluLab/DREAMS):

1. Do categorical, taxonomy, and semantic proximity predict emotional similarity between physical places better than geographic distance alone?
2. Can dimension weights be learned from emotional similarity data instead of keeping the hand-set defaults in `dreams_app/core/extra/proximity_calculator.py` (`geo=0.3, cat=0.4, ling=0.15, cult=0.15`)?
3. How much visit-level CHIME variation belongs to an individual physical place ("this particular church") versus its place type ("any church")?

## Why the Existing Code Cannot Answer This

Inspection of the codebase (see [PROXIMITY_MAP.md](../PROXIMITY_MAP.md)) identified four blocking gaps:

- `composite_proximity` (`dreams_app/core/extra/proximity_calculator.py`) hard-codes `{'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}` and omits empirical validation against emotional outcomes.
- `categorical_proximity` uses a 3-pair table (`clinic/hospital`, `church/temple`, `cafe/restaurant`), scoring `0.0` for all other related places (for example `church/mosque` or `park/garden`).
- `PlaceVisit` (`dreams_app/core/extra/proximity_trajectory.py`) and `PlaceEmotionSignature` (`dreams_app/core/extra/place_emotion_signature.py`) aggregate solely by `place_type` and store no `place_id` or coordinates, so distinct places of the same type always receive the same signature.
- `reverse_geocode` (`dreams_app/core/extra/location_extractor.py`) forces English and does not extract `language` or `cultural_tags`, and the proximity modules are not called by the ingestion pipeline or analytics API.

## Method Summary

1. **Baseline Reproduction ([baseline.py](baseline.py)):** Computes pairwise default, geo-only, and category-only proximity matrices on 12 sample places across Anchorage, Seattle, and Lisbon, and shows that pooling CHIME visits by `place_type` erases real place-level differences between two churches.
2. **Synthetic Benchmark ([generate.py](generate.py)):** Generates places over a 3-level taxonomy tree (`depth=0..2`), Poisson user visit sequences, and softmax CHIME distributions from known additive logits, with `geo_category_correlation` in `[0, 1]` to control spatial confounding.
3. **Proximity Measures ([distances.py](distances.py)):** Combines Haversine + repo `normalize_geographic_distance`, Wu-Palmer taxonomy similarity (`category_tree_similarity`), place-name cosine similarity (`all-MiniLM-L6-v2` if installed/available, else deterministic 2-document char 3-gram TF-IDF), repo `linguistic_similarity` and `cultural_similarity`, and renormalization matching `composite_proximity`.
4. **Evaluation & ANOVA ([evaluate.py](evaluate.py), [run_experiments.py](run_experiments.py)):** Aggregates CHIME profiles per `place_id` (`min_visits >= 5`), defines target similarity as `-JSD` (`scipy.spatial.distance.jensenshannon` base 2), evaluates `M0`–`M5` under strict leave-place-out cross-validation (mean fold Spearman + pooled OOF $R^2$), runs feature ablations, place-profile permutation tests (`p`-values), place-fold bootstrap CIs, and nested visit-level ANOVA (`group`, `type|group`, `place|type`, `residual`).

## Synthetic data

Because no annotated user recovery dataset is publicly available without IRB approval, [generate.py](generate.py) provides a deterministic synthetic generator (`generate_dataset(config, seed)`) with planted ground-truth emotional effects.

### Taxonomy Tree

- `root` (`depth=0`)
  - `religious` (`depth=1`): `church`, `temple`, `mosque` (`depth=2`)
  - `healthcare` (`depth=1`): `hospital`, `clinic`, `pharmacy` (`depth=2`)
  - `recreation` (`depth=1`): `park`, `garden`, `beach` (`depth=2`)
  - `food` (`depth=1`): `restaurant`, `cafe`, `bar` (`depth=2`)
  - `civic` (`depth=1`): `library`, `school`, `community_center` (`depth=2`)

### Places and Geographic–Categorical Coupling

By default (`GeneratorConfig`), `n_places=60` places are created across `n_cities=4` cities (`id`, templated `name`, leaf `type`, parent `group`, `city`, `latitude`, `longitude`, `language`, and `0` to `3` `cultural_tags`). `geo_category_correlation` in `[0.0, 1.0]` assigns each place to its type's home city with probability `geo_category_correlation` or uniformly across cities otherwise.

### Users, Visits, and Emotion Model

For `n_users=40` users, visit counts are drawn from `Poisson(mean=8.0)` over a 180-day UTC horizon. Each visit computes logits over `CHIME_DIMENSIONS` (`dreams_app.core.extra.place_emotion_signature`):

$$\text{logits} = c \cdot \text{category\_effect}[\text{type}] + \gamma \cdot \text{group\_effect}[\text{group}] + \alpha \cdot \text{place\_effect}[\text{place\_id}] + \beta \cdot \text{user\_effect}[\text{user\_id}] + \sigma \cdot \mathcal{N}(0, I)$$

followed by softmax so each visit vector sums to `1.0`. All ground-truth effect tables are returned in `SyntheticDataset`.

### Export Shapes

- `export_to_place_visits(dataset)` converts visits to `dreams_app.core.extra.proximity_trajectory.PlaceVisit(place_type, timestamp, chime)` (note that `PlaceVisit` omits `place_id`, `place_group`, and `user_id`).
- `export_chime_by_type(dataset)` groups CHIME dicts by `place_type` for `build_place_signature()`.

### Scope Statement

Synthetic data can only show the method recovers planted effects and says nothing about real recovery journeys. Validation on real human memories remains necessary once ethically sourced and IRB-approved data are available.

## How to Run

Run from the repository root (`DREAMS/`):

```bash
# 1. Reproduce baseline behavior (writes research/results/baseline_output.txt)
python research/proximity_validation/baseline.py

# 2. Run all research proximity unit tests (uses deterministic TF-IDF fallback)
pytest tests/research_proximity/ -v

# 3. Run the 135-run grid sweep (writes research/results/experiments.csv & summary.md)
python research/proximity_validation/run_experiments.py

# 4. Run DREAMS existing test suite cleanly (offline HuggingFace mode)
HF_HUB_OFFLINE=1 pytest --ignore=tests/test_places365_classifier.py -v
```

Full evaluation results and limitations are documented in [RESULTS.md](../RESULTS.md) and [REVIEW.md](../REVIEW.md).

## Limits

- **Synthetic data only:** All places, visits, and CHIME scores in this study come from a parametric softmax model. Recovering planted statistical structure is a sanity check on the estimator and says nothing about real human emotions or recovery journeys.
- **Data needed next:** De-identified or consented participant memories (timestamps, venue/GPS clusters, and validated CHIME annotations) collected under an approved IRB protocol, plus real geocoder/scene labels (`location_extractor.py`, `places365_classifier.py`), would be needed to test whether taxonomy/semantic dimensions hold outside synthetic assumptions and which weights reflect lived recovery experiences.

## References

- Repository: https://github.com/KathiraveluLab/DREAMS
- Maintainer design framing on multi-dimensional proximity and particular place vs. place class: https://github.com/KathiraveluLab/DREAMS/discussions/71
- Maintainer GSoC research framing: https://github.com/KathiraveluLab/DREAMS/discussions/285
- Alaska AI and contributor guidelines: https://github.com/uaanchorage/GSoC/blob/main/Acceptable-and-Ethical-AI-Use-Policy.md
