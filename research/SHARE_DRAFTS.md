# Drafts to Rewrite and Share with the Maintainer (`#285`)

> **Note:** Draft notes only — rewrite in your own words before posting (`uaanchorage/GSoC/Acceptable-and-Ethical-AI-Use-Policy.md`). Nothing has been posted anywhere.

---

## 1. Plain-Language Summary for Discussion `#285` (Rewrite in Own Words)

To follow up on Idea 2, I built a prototype in my fork (https://github.com/madhavzanwar/DREAMS/tree/research/proximity-validation) testing whether multi-dimensional place proximity predicts emotional similarity better than geography alone, whether weights can be learned instead of fixed (`0.3, 0.4, 0.15, 0.15`), and how much CHIME variance belongs to an individual place versus its type. The code generates synthetic places over a 3-level taxonomy with softmax CHIME visits and measures Haversine, Wu-Palmer taxonomy, char 3-gram TF-IDF name, language, and tag similarities against Jensen-Shannon distance. Across 135 synthetic runs using leave-place-out cross-validation, permutation tests, and nested ANOVA, Ridge (`0.211`) and NNLS (`0.228`) beat fixed repo weights (`0.092`) when types spread across cities, while spatial clustering inflates geo-only Spearman (`-0.030` to `0.062`) without any geographic effect; raising place-specific strength lifts within-type place variance (`0.036` to `0.323`) and drops type-only Spearman (`0.343` to `0.201`). All data is synthetic, checking estimators only and saying nothing about real recovery journeys. `[CHECK: rewrite in your voice]`

---

## 2. Three Specific One-Line Questions for the Maintainer

1. Is validating learned weights alongside separating physical `place_id` from `place_type` signatures the right framing of "formalizing proximity" for Idea 2 `[CHECK: confirm alignment with any January 2027 updates to the ideas list]`?
2. Does any de-identified or simulated sample dataset of location + CHIME visits exist for validation, or should early GSoC benchmarks rely solely on synthetic and self-collected traces pending IRB approval?
3. Should the prototype's weights and place-level signatures feed the app's runtime place-emotion analysis (extending `PlaceVisit`, `build_place_signature`, and `place_narrative_bridge.py`), or remain a standalone analysis module first?

---

## 3. Five Most Important Weaknesses a Skeptical Mentor Would Find

1. **Zero real human data:** All places, visits, and CHIME vectors come from a parametric softmax model (`generate.py`); recovering planted weights and ANOVA shares is a mathematical sanity check and provides no evidence about real recovery journeys (`RESULTS.md` Section 6).
2. **Shared taxonomy and template-name circularity:** `distances.py` scores categories on the same 3-level taxonomy tree used by `generate.py` to generate emotions, and templated names (`"St. Mary's Church"`) contain literal type nouns—giving TF-IDF `semantic_name` (`M2`) Spearman `0.097`–`0.204` purely through lexical category leakage rather than independent semantic meaning (`REVIEW.md` Section 2, `RESULTS.md` Section 5).
3. **Learned multi-feature models still trail univariate `category_tree`:** Under unconfounded category signal (`geo_corr=0.0, cat=1.0, place=0.0`), Ridge (`M4=0.211`) and NNLS (`M5=0.228`) outperform fixed repo weights (`M3=0.092`), yet both underperform single-feature `M1` (`0.316`) because fitting five correlated features on ~`34–44` eligible places adds estimation variance (`experiments.csv`, `summary.md`).
4. **Offline TF-IDF fallback and unplanted `ling`/`cult`/neighborhood effects:** All reported evaluations ran `semantic_backend="tfidf"` rather than `all-MiniLM-L6-v2` `[CHECK: verify whether sentence-transformers embeddings improve M2 on non-templated venue names]`, and the generator did not plant direct effects for `language`, `cultural_tags`, or neighborhood geographic distance (so `max_distance_km=500.0` urban compression remains untested empirically).
5. **Bootstrap independence assumption and finite-sample ANOVA bleed:** `evaluate.py` bootstraps the `M` already-fitted LPO fold correlations instead of re-fitting all `M` folds per draw (producing slightly narrower CIs across overlapping pairs), and unadjusted sum-of-squares assigns `0.077` apparent `Var(Place|Type)` even when `place_strength=0.0` due to small per-place visit counts (~`5.3` visits/place) (`REVIEW.md` Section 2, `experiments.csv` rows `2–6`).
