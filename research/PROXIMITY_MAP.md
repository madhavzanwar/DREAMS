# Proximity Code & Design Map

Mapped from `research/proximity-validation` (branched from `upstream/main` at `cf19c15`).

## Maintainer Framing (Discussions & Docs)
- **Discussion `#71` (`pradeeban`):** Proximity in DREAMS *"needs to be formalized"*, is *"multi-dimensional"* (not just geo-coordinates, *"also how places can feel near"*), and must distinguish *"a particular place (e.g., this particular church) vs. a specific class of places (e.g., any churches)"*. Notes real marginalized-participant data is hard to obtain upfront (*"chicken-and-egg problem"*), so initial testing can use synthetic/self-collected data.
- **Discussions `#273`, `#285`, `#294` (`pradeeban`):** Current GSoC ideas list (~2026) will be updated by January for 2027; DREAMS has a research component beyond current source code, and proposal quality/research framing carry the main evaluation weight.

---

## (a) Inventory of Proximity & Similarity Functions

| File | Function | Inputs | Output Range | Assumptions | Hard-coded Constants |
|---|---|---|---|---|---|
| `dreams_app/core/extra/proximity_calculator.py` | `categorical_proximity` (L21–53) | `type1: str`, `type2: str` | `{0.0, 0.5, 1.0}` | Case/whitespace-insensitive exact match or symmetric pair lookup; falsy input returns `0.0` | `CATEGORY_RELATIONSHIPS = {('clinic', 'hospital'): 0.5, ('church', 'temple'): 0.5, ('cafe', 'restaurant'): 0.5}` (L14–18) |
| `dreams_app/core/extra/proximity_calculator.py` | `linguistic_similarity` (L56–78) | `language1: str`, `language2: str` | `{0.0, 1.0}` | Binary exact string equality after `.lower().strip()`; no language-family partial credit | `1.0` match / `0.0` mismatch |
| `dreams_app/core/extra/proximity_calculator.py` | `cultural_similarity` (L81–117) | `tags1: List[str]`, `tags2: List[str]` | `[0.0, 1.0]` | Jaccard index on normalized non-empty tag sets; **both empty returns `1.0`**, one empty returns `0.0` | None |
| `dreams_app/core/extra/proximity_calculator.py` | `composite_proximity` (L120–184) | `place1: Dict`, `place2: Dict`, `weights: Dict[str, float] = None` | `[0.0, 1.0]` (for `[0, 1]` inputs & non-negative weights) | Linear weighted sum over dimensions present in both places (`geo_proximity` on either place counts); renormalized by active weights; `0.0` if total weight is `0` | `default_weights = {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}` (L153) |
| `dreams_app/core/extra/proximity_calculator.py` | `normalize_geographic_distance` (L186–211) | `distance_km: float`, `max_distance_km: float = 500.0` | `(0.0, 1.0]` | Exponential decay $\exp(-d / (\text{max\_distance\_km}/3))$; raises `ValueError` for `distance_km < 0` or `max_distance_km <= 0` | `max_distance_km = 500.0` (L186), scale divisor `3.0` (L210) |
| `dreams_app/core/extra/proximity_calculator.py` | `cluster_places_dbscan` (L214–287) | `places: List[Dict]`, `eps: float = 0.4`, `min_samples: int = 2`, `weights = None` | `List[int]` (`-1` for noise, `>= 0` cluster IDs) | Pairwise distance $D_{ij} = \max(0.0, 1.0 - P_{\text{composite}}(L_i, L_j))$ clustered via DBSCAN | `eps = 0.4`, `min_samples = 2` (L216–217) |
| `dreams_app/location_proximity.py` | `calculate_distance` (L33–60) | `lat1, lon1, lat2, lon2: float` | `[0.0, ~20,015,087]` m | Spherical Earth Haversine formula; valid lat in `[-90, 90]`, lon in `[-180, 180]`; clamps `a = min(1.0, a)` | `R = 6371000.0` m (L48) |
| `dreams_app/location_proximity.py` | `compute_proximity` (L63–78) | `location1: Location`, `location2: Location`, `threshold_meters: float` | `distance >= 0.0`, `is_proximate: bool` | Binary distance threshold `dist <= threshold_meters` | None (threshold passed by caller) |
| `dreams_app/location_proximity.py` | `find_nearby_locations` (L105–133) | `target_location: Location`, `locations: List[Location]`, `radius_meters: float` | `List[Location]` | Excludes target by object identity (`loc is target_location`), silently skips invalid coordinates | None |
| `dreams_app/location_proximity.py` | `cluster_locations` (L136–172) | `locations: List[Location]`, `proximity_threshold: float` | `List[List[Location]]` | BFS connected components where edge exists if `dist <= proximity_threshold` | None |
| `dreams_app/core/extra/place_emotion_signature.py` | `chime_proximity` (L64–88) | `sig_a: PlaceEmotionSignature`, `sig_b: PlaceEmotionSignature`, `min_visits: int = 1` | `[0.0, 1.0]` (for non-negative CHIME scores) | Cosine similarity on mean 5-dim CHIME vectors; returns `0.5` if either `visit_count < min_visits`, `0.0` if either vector has zero norm | `min_visits = 1` (L67), insufficient-data fallback `0.5` (L76) |
| `dreams_app/core/extra/place_emotion_signature.py` | `detect_recovery_anchors` (L91–121) | `signatures: Dict[str, PlaceEmotionSignature]`, `anchor_dimensions = None`, `threshold = 0.6`, `min_visits = 2` | List of `(place_type, score >= threshold, dim)` | Anchor score $= \text{best\_score} \cdot \max(0, 1 - \text{volatility})$; filters `visit_count >= min_visits` | `anchor_dimensions = ['Hope', 'Connectedness']` (L107), `threshold = 0.6`, `min_visits = 2` (L94–95) |
| `dreams_app/core/graph/place_narrative_bridge.py` | `enrich_narrative_edges_with_place_proximity` (L13–58) | `graph: TemporalNarrativeGraph`, `episode_place_signatures: Dict[int, PlaceEmotionSignature]`, `place_weight = 0.3` | `TemporalNarrativeGraph` (weights clamped to `[0.0, 1.0]`) | Blends temporal edge weight with `chime_proximity(sig_a, sig_b)` when both endpoints have signatures | `place_weight = 0.3` (L16) |
| `dreams_app/core/graph/place_narrative_bridge.py` | `compute_place_proximity_matrix` (L61–81) | `signatures: Dict[str, PlaceEmotionSignature]` | `[0.0, 1.0]` symmetric matrix | Sets diagonal `matrix[pt][pt] = 1.0`; delegates off-diagonal to `chime_proximity` | Diagonal `1.0` (L77) |
| `dreams_app/core/graph/place_narrative_bridge.py` | `find_emotionally_proximate_pairs` (L84–106) | `signatures: Dict[str, PlaceEmotionSignature]`, `threshold = 0.7` | Sorted `(pt_a, pt_b, score)` with `score >= threshold` | Uses `chime_proximity` with default `min_visits=1` | `threshold = 0.7` (L86) |
| `dreams_app/core/graph/time_aware_proximity.py` | `temporal_distance` (L96–141) | `a: EmotionTimeline`, `b: EmotionTimeline`, `window: timedelta`, `anchor = "start"`, `anchor_time = None` | `[0.0, inf)` | Counts aligned windows where presence/absence differs; returns `len(non_empty.events)` if one timeline is empty, `0.0` if both empty | `anchor = "start"` (L100) |
| `dreams_app/core/graph/time_aware_proximity.py` | `proximity_matrix` (L144–164) | `timelines: Tuple[EmotionTimeline, ...]`, `window: timedelta` | Square symmetric matrix `>= 0.0`, `0.0` diagonal | Pairwise `temporal_distance(..., anchor="start")` | None |
| `dreams_app/core/graph/emotion_proximity.py` | `compare_timelines_distance` (L279–423) | `timeline_a`, `timeline_b`, `window_duration`, `anchor_time = None`, `use_event_scores = False`, `missing_value = 0.0` | `mean_absolute_difference` in `[0.0, 2.0]` (ordinal) or `None`; `sum_squared_difference >= 0.0` | Maps labels via `EMOTION_LABEL_MAP` (unless `use_event_scores=True`), averages per window, fills missing windows with `missing_value` | `EMOTION_LABEL_MAP = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}` (L61–65), `missing_value = 0.0` (L285) |
| `dreams_app/core/graph/emotion_proximity.py` | `compute_timeline_self_similarity` (L426–506) | `timeline`, `window_duration`, `use_event_scores = False` | `score_range`, `score_variance >= 0.0` | Population variance (`/ N`) across per-window mean scores | `EMOTION_LABEL_MAP` (L61–65) |
| `dreams_app/core/graph/episode_proximity.py` | `compute_temporal_overlap` (L24–49) | `episode_a: Episode`, `episode_b: Episode` | `[0.0, 1.0]` | Overlap duration divided by `min(duration_a, duration_b)` | None |
| `dreams_app/core/graph/episode_proximity.py` | `are_episodes_adjacent` (L71–92) | `episode_a`, `episode_b`, `adjacency_threshold = timedelta(0)` | `bool` | Non-overlapping (`overlap == 0.0`) with `gap <= adjacency_threshold` | `timedelta(0)` (L74) |
| `dreams_app/core/graph/episode_proximity.py` | `classify_episode_proximity` (L95–117) | `episode_a`, `episode_b`, `adjacency_threshold = timedelta(0)` | `ProximityRelation` (`OVERLAPPING`, `ADJACENT`, `DISJOINT`) | Priority order: overlapping first, then gap `<= adjacency_threshold`, else disjoint | `timedelta(0)` (L98) |

Note on `PlaceVisit`: defined as a dataclass in `dreams_app/core/extra/proximity_trajectory.py` (L14–26) with fields `place_type: str`, `timestamp: datetime`, and `chime: Dict[str, float]`.

---

## (b) Exact Formula of `composite_proximity`
In `dreams_app/core/extra/proximity_calculator.py` (L120–184):

1. Effective weights start from `default_weights = {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}` (L153) and update with any caller-supplied `weights` (`{**default_weights, **(weights or {})}`, L154).
2. Dimension scores $s_d$ are computed only for the active subset $S \subseteq \{\text{geo}, \text{cat}, \text{ling}, \text{cult}\}$ (L158–176):
   - $\text{geo} \in S$ iff `place1.get('geo_proximity') is not None` or `place2.get('geo_proximity') is not None`; $s_{\text{geo}}$ is the mean of the non-`None` values (L162–167).
   - $\text{cat} \in S$ iff `bool(place1.get('type')) and bool(place2.get('type'))`; $s_{\text{cat}} = \text{categorical\_proximity}(\text{type}_1, \text{type}_2)$ (L169–170).
   - $\text{ling} \in S$ iff `bool(place1.get('language')) and bool(place2.get('language'))`; $s_{\text{ling}} = \text{linguistic\_similarity}(\text{language}_1, \text{language}_2)$ (L172–173).
   - $\text{cult} \in S$ iff `bool(place1.get('cultural_tags')) and bool(place2.get('cultural_tags'))`; $s_{\text{cult}} = \text{cultural\_similarity}(\text{tags}_1, \text{tags}_2)$ (L175–176). Note that an empty list `[]` is falsy, so missing or empty tag lists are excluded from $S$.
3. Renormalized composite score (L179–183):
   $$P_{\text{composite}}(L_1, L_2) = \begin{cases} 0.0 & \text{if } \sum_{d \in S} w_d = 0 \\ \dfrac{\sum_{d \in S} w_d \, s_d}{\sum_{d \in S} w_d} & \text{otherwise} \end{cases}$$

---

## (c) How `categorical_proximity` Maps Place Types
In `dreams_app/core/extra/proximity_calculator.py` (L14–53):

1. If either `type1` or `type2` is falsy (`""` or `None`), returns `0.0` (L42–43).
2. Normalizes both strings via `.lower().strip()` (L45–46).
3. Returns `1.0` if `type1_lower == type2_lower` (L48–49).
4. Otherwise sorts the pair alphabetically `tuple(sorted([type1_lower, type2_lower]))` and looks it up in `CATEGORY_RELATIONSHIPS` (L52–53), which contains **only three hard-coded pairs**:
   - `('clinic', 'hospital'): 0.5`
   - `('church', 'temple'): 0.5`
   - `('cafe', 'restaurant'): 0.5`
5. Any other pair returns `0.0`. (Notice `location_proximity/RESEARCH.md` L107–111 describes 4 broader groups—`{church, temple, mosque, synagogue}`, `{hospital, clinic, pharmacy}`, `{park, garden, beach}`, `{restaurant, cafe, bar}`—but those additional terms are not in `CATEGORY_RELATIONSHIPS`.)

---

## (d) `PlaceEmotionSignature` Structure & Aggregation Level
In `dreams_app/core/extra/place_emotion_signature.py` (L6–61):

- `CHIME_DIMENSIONS = ['Connectedness', 'Hope', 'Identity', 'Meaning', 'Empowerment']` (L6).
- `PlaceEmotionSignature` dataclass fields (L9–23):
  - `place_type: str` — category label (e.g., `'church'`, `'park'`)
  - `chime_vector: Dict[str, float]` — mean score across all visits for each of the 5 `CHIME_DIMENSIONS` (missing dimensions in a visit default to `0.0`, L45–48)
  - `visit_count: int = 0` — number of visits aggregated
  - `volatility: float = 0.0` — RMS of per-dimension population standard deviations: $\sqrt{\frac{1}{5} \sum_{d \in \text{CHIME}} \frac{1}{n} \sum_{i=1}^n (r_{i,d} - \bar{r}_d)^2}$ (L50–54)
- **Aggregation level:** `build_place_signature(place_type, chime_results)` aggregates strictly by **`place_type`** (category string). Neither `PlaceVisit` (`proximity_trajectory.py` L14–26) nor `PlaceEmotionSignature` stores a physical place ID, name, or coordinates, so all visits to different places of the same type are pooled together.

---

## (e) Concrete Limitations of the Current Approach
1. **Hand-set, unvalidated composite weights:** `default_weights = {'geo': 0.3, 'cat': 0.4, 'ling': 0.15, 'cult': 0.15}` in `dreams_app/core/extra/proximity_calculator.py` (L153) are fixed constants; no optimization or empirical fit checks whether these weights predict emotional similarity.
2. **Sparse, hard-coded category relationships disconnected from geocoder & scene taxonomies:** `CATEGORY_RELATIONSHIPS` in `dreams_app/core/extra/proximity_calculator.py` (L14–18) has only 3 pairs, omitting terms listed in `location_proximity/RESEARCH.md` (L107–111), OSM Nominatim values returned by `reverse_geocode` in `dreams_app/core/extra/location_extractor.py` (L198–203, e.g., `place_of_worship`, `leisure`), and the 6 DREAMS categories in `dreams_app/core/extra/places365_classifier.py` (L28–35).
3. **No place-instance vs. place-type separation ("this particular church" vs. "any church"):** `PlaceVisit` (`dreams_app/core/extra/proximity_trajectory.py` L14–26) and `PlaceEmotionSignature` (`dreams_app/core/extra/place_emotion_signature.py` L9–23) only carry `place_type: str` and no physical place identifier, making it impossible to measure how much emotional variance is specific to a physical location versus shared across its type.
4. **Binary linguistic similarity & missing extraction for linguistic/cultural dimensions:** `linguistic_similarity` in `dreams_app/core/extra/proximity_calculator.py` (L56–78) is exact string equality (`0.0` or `1.0`), `cultural_similarity` (L81–117) returns `1.0` for two empty tag lists when called directly, and `reverse_geocode` in `dreams_app/core/extra/location_extractor.py` (L152–221) forces `accept-language=en` and does not populate `language` or `cultural_tags` on stored posts.
5. **No validation against emotional outcomes & disconnected runtime modules:** Neither `composite_proximity` (`dreams_app/core/extra/proximity_calculator.py` L120), `chime_proximity` (`dreams_app/core/extra/place_emotion_signature.py` L64), nor `build_place_trajectories` (`dreams_app/core/extra/proximity_trajectory.py` L51) is called anywhere in `dreams_app/` outside `dreams_app/core/extra/` and `dreams_app/core/graph/place_narrative_bridge.py`, and `chime_proximity` silently returns `0.5` when either signature has fewer than `min_visits` (`place_emotion_signature.py` L75–76) without flagging low support.
6. **Unvalidated geographic decay scale:** `normalize_geographic_distance` in `dreams_app/core/extra/proximity_calculator.py` (L186, L210) hard-codes `max_distance_km = 500.0` (`scale = 166.7` km), compressing urban/neighborhood distances (e.g., 1 km vs. 10 km yield ~0.994 vs. ~0.942) into nearly identical scores.
7. **Unweighted index-based linear trend and coarse ordinal emotion mapping:** `build_place_trajectories` in `dreams_app/core/extra/proximity_trajectory.py` (L80–95) regresses CHIME scores on visit index `0..n-1` (ignoring actual elapsed time/seasonality), while `emotion_proximity.py` (`EMOTION_LABEL_MAP`, L61–65) reduces emotions to `{1.0, 0.0, -1.0}` and `time_aware_proximity.py` (`temporal_distance`, L96–141) ignores emotion labels altogether.
