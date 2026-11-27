# Proximity Validation Experiments

Research code and benchmarks on branch `research/proximity-validation` exploring whether multi-dimensional place proximity predicts emotional similarity in DREAMS, whether dimension weights can be learned from data, and how emotional profiles decompose across physical places versus place categories.

## Synthetic data

Because no annotated user recovery dataset is publicly available without IRB approval, [generate.py](generate.py) provides a deterministic synthetic generator (`generate_dataset(config, seed)`) with planted ground-truth emotional effects.

### Taxonomy Tree

Places are organized in a fixed 3-level taxonomy tree (`depth=0..2`):

- `root` (`depth=0`)
  - `religious` (`depth=1`): `church`, `temple`, `mosque` (`depth=2`)
  - `healthcare` (`depth=1`): `hospital`, `clinic`, `pharmacy` (`depth=2`)
  - `recreation` (`depth=1`): `park`, `garden`, `beach` (`depth=2`)
  - `food` (`depth=1`): `restaurant`, `cafe`, `bar` (`depth=2`)
  - `civic` (`depth=1`): `library`, `school`, `community_center` (`depth=2`)

### Places and Geographic–Categorical Coupling

By default (`GeneratorConfig`), `n_places=60` places are created across `n_cities=4` cities with:

- `id`, templated `name` (e.g., `"St. Mary's Church"`), leaf `type`, parent `group`, `city`
- `latitude`, `longitude` sampled around city centers (clipped to valid ranges)
- `language` (derived from city template) and `0` to `3` `cultural_tags`
- `geo_category_correlation` in `[0.0, 1.0]`: assigns each place to its type's designated home city with probability `geo_category_correlation` or uniformly across all cities otherwise, letting experiments isolate categorical proximity (`0.0`) from co-located city confounding (`1.0`).

### Users, Visits, and Emotion Model

For `n_users=40` users, visit counts are drawn from `Poisson(mean=8.0)` with chronologically ordered UTC timestamps over a configurable time horizon. Each visit produces a 5-dimensional distribution over `CHIME_DIMENSIONS = ['Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment']` (imported from `dreams_app.core.extra.place_emotion_signature`) via:

$$\text{logits} = c \cdot \text{category\_effect}[\text{type}] + \gamma \cdot \text{group\_effect}[\text{group}] + \alpha \cdot \text{place\_effect}[\text{place\_id}] + \beta \cdot \text{user\_effect}[\text{user\_id}] + \sigma \cdot \mathcal{N}(0, I)$$

followed by a softmax across the 5 CHIME dimensions so every visit vector sums to `1.0`. All ground-truth effect tables are drawn once from seeded standard normals and returned alongside `places`, `visits`, and `config` in `SyntheticDataset`.

### Export Shapes for Existing DREAMS Modules

- `export_to_place_visits(dataset)` converts `SyntheticVisit` objects into `dreams_app.core.extra.proximity_trajectory.PlaceVisit(place_type, timestamp, chime)` records (the existing DREAMS visit shape, noting that `PlaceVisit` itself omits `place_id`, `place_group`, and `user_id`).
- `export_chime_by_type(dataset)` groups `visit.chime` dictionaries by `place_type` for `dreams_app.core.extra.place_emotion_signature.build_place_signature()`.

### Scope and Limitation

Synthetic data can only show the method recovers planted effects and says nothing about real recovery journeys. Validation on real human memories remains necessary once ethically sourced and IRB-approved data are available.
