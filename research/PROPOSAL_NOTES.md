# GSoC 2027 Proposal Notes (Alaska Idea #2 — Outline & Facts Only)

> **Important:** These are structured factual notes and options built from our work on `research/proximity-validation`. **Do NOT copy-paste this verbatim into a final proposal or discussion.** Write the final proposal and all GitHub discussions in your own words per `uaanchorage/GSoC` guidelines (`Acceptable-and-Ethical-AI-Use-Policy.md`, `Rubrics.md`).

---

## Alaska 12-Heading Proposal Outline (`CONTRIBUTOR-GUIDE.md`)

### 1) Project Title
- **Official Idea Title:** Understanding proximity in locations and emotions through digitized memories
- **Working Research-Focused Subtitle Option:** Learning Multi-Dimensional Place Proximity Weights and Separating Place-Instance vs. Category-Level CHIME Signatures in DREAMS

### 2) Abstract / Project Summary (Facts & Prototype Findings to Synthesize in Own Words)
- **Current gap in `KathiraveluLab/DREAMS`:**
  - `composite_proximity` (`dreams_app/core/extra/proximity_calculator.py`) uses hard-coded weights (`geo=0.3, cat=0.4, ling=0.15, cult=0.15`) and a 3-pair lookup (`CATEGORY_RELATIONSHIPS`: `clinic/hospital`, `church/temple`, `cafe/restaurant`).
  - `PlaceEmotionSignature` (`dreams_app/core/extra/place_emotion_signature.py`) and `PlaceVisit` (`dreams_app/core/extra/proximity_trajectory.py`) aggregate only by `place_type`, pooling all physical locations of a type together ("any church" vs. "this particular church").
  - `reverse_geocode` (`dreams_app/core/extra/location_extractor.py`) forces `accept-language=en` and does not extract `language` or `cultural_tags`; proximity/signature modules are not yet wired into main ingestion/analytics endpoints.
- **What this pre-proposal prototype (`research/proximity_validation/`) showed:**
  - Built a deterministic synthetic benchmark (`generate.py`), taxonomy/semantic/composite metrics (`distances.py`), and leave-place-out CV evaluation (`evaluate.py`) with 38 passing tests.
  - Across `135` sweep runs (`5` seeds × `27` parameter combinations), learned models (`M4` Ridge, `M5` NNLS) substantially outperformed the repo's hand-weighted `M3` when geography and category are decoupled (`geo_corr=0.0`), while revealing how spatial clustering (`geo_corr=1.0`) confounds geographic proximity (`M0`) with categorical effects.
  - Demonstrated quantitatively that raising individual place divergence (`place_strength` `0.0 -> 1.0`) raises within-type place variance (`Var(Place|Type)`) and degrades type-only proximity (`M1`), motivating place-instance-aware CHIME modeling.

### 3) Contributor Name
- **Name:** Madhav Zanwar

### 4) Contributor Email and GitHub ID
- **Email:** `madhavzanwar13@gmail.com`
- **GitHub ID:** `madhavzanwar` (fork: `https://github.com/madhavzanwar/DREAMS`)

### 5) Potential Mentor(s)
- **Listed Idea #2 Mentors (`uaanchorage/GSoC`):**
  - Mohamed Abdullah F (`abdullahfakrudeen2020 -at- gmail.com`)
  - David P. Moxley (`dpmoxley -at- alaska.edu`)
- **Project Maintainer / Repository Contact (`KathiraveluLab/DREAMS`):**
  - Pradeeban Kathiravelu (engaged on DREAMS discussions `#71`, `#273`, `#285`, `#294`)

### 6) Personal Background (Brief CV)
- **[FILL IN BY CONTRIBUTOR]:** Degree program, university, graduation year, relevant coursework (ML/statistics, databases, geospatial/NLP).
- **Verified DREAMS Open-Source Record:**
  - Active contributor to `KathiraveluLab/DREAMS` (`madhavzanwar` PRs and issue discussions `#273`, `#285`, `#294`, including work around stored CHIME distributions).
  - Pre-proposal research prototype at `madhavzanwar/DREAMS` branch `research/proximity-validation` (`PROXIMITY_MAP.md`, `baseline.py`, `generate.py`, `distances.py`, `evaluate.py`, `run_experiments.py`, `REVIEW.md`, 38 unit tests).

### 7) Project Goals / Major Contributions (Concrete Scope Options)
- **Goal 1 — Generalize Multi-Dimensional Place Proximity:**
  - Replace the 3-pair `CATEGORY_RELATIONSHIPS` dictionary with taxonomy-aware similarity (e.g., Wu-Palmer over OSM Nominatim / Places365-aligned hierarchies) and semantic name/text similarity (`all-MiniLM-L6-v2` with TF-IDF fallback).
  - Populate missing `language` and `cultural_tags` safely from geocoding/metadata or mark unavailable dimensions cleanly.
- **Goal 2 — Place-Instance vs. Place-Type Emotional Signatures (Hierarchical Modeling):**
  - Extend visit representations (preserving backward compatibility with `PlaceVisit`) to track privacy-preserving `place_id` alongside `place_type` and parent group.
  - Separate place-specific CHIME signatures ("this particular church") from category priors ("any church"), e.g., partial-pooling / empirical-Bayes shrinkage from `place_type` to `place_id` so low-visit places fall back smoothly while well-visited places retain distinct emotional identities.
- **Goal 3 — Data-Driven Weight Learning & Unconfounded Validation:**
  - Fit composite weights (Ridge / NNLS / simplex-constrained weights) under strict leave-place-out (and leave-user-out) cross-validation with permutation tests and variance decomposition.
  - Compare learned weights against `composite_proximity` defaults on both controlled synthetic stress tests and any mentor-approved pilot/anonymized real dataset.
- **Goal 4 — Runtime Integration & Documentation (Subject to Mentor Priority):**
  - Wire `PlaceVisit` adapter and read-only analytics / signature endpoints into `dreams_app`, backed by Flask tests and reproducible CLI scripts.

### 8) Expected AI Use (Truthful Statement Draft Notes)
- **Recommended classification (`CONTRIBUTOR-GUIDE.md`):** **Medium-level AI-assisted coding** (never "No AI use" if any LLM assistance is planned or was used).
- **Truthful summary of actual pre-proposal use:**
  - Used an AI engineering assistant to explore `KathiraveluLab/DREAMS`, audit existing proximity modules, scaffold the modular prototype (`research/proximity_validation/`), write initial unit tests, run independent code-review checks (`REVIEW.md`), and organize these proposal outline notes.
  - Verified all outputs by running `pytest`, `flake8`, `black`, and auditing mathematical logic (catching and fixing corpus-IDF leakage, sentinel p-values, and LPO pooled intercept bias).
- **Planned GSoC policy commitments (`Acceptable-and-Ethical-AI-Use-Policy.md`):**
  - Write the final GSoC proposal, GitHub discussion posts, and research reports personally in own words.
  - Review, understand, and locally test all code with unit/integration tests before opening PRs.
  - Disclose AI assistance accurately in PR descriptions/commits if used for boilerplate, debugging, or test expansion.

### 9) Project Schedule (350 Hours Total, ~7-Day Blocks)

#### 9.1) Community Bonding Period (~3 Weeks Before Coding)
- **CB Block 1:** Confirm scope priorities, privacy constraints on `place_id`, and taxonomy source (OSM vs. Places365 vs. hybrid) with mentors on `KathiraveluLab/DREAMS/discussions`.
- **CB Block 2:** Inspect available pilot / anonymized / self-collected sample schema; verify MongoDB + offline test fixtures end-to-end.
- **CB Block 3:** Finalize PR/architecture design doc (backward compatibility for `PlaceVisit`, adapter schema, and evaluation benchmark specs).

#### 9.2) Development Phase (Weeks 1–8, ~35 Hours/Week = 280 Hours)
- **Week 1 (35h):** Build canonical `Post -> PlaceVisit` adapter supporting optional `place_id`, timezone normalization, and full CHIME score distributions; unit tests + edge cases.
- **Week 2 (35h):** Implement taxonomy-based `categorical_proximity` mapped from `location_extractor.py` (OSM types/categories) and `places365_classifier.py`, keeping backward compatibility.
- **Week 3 (35h):** Implement semantic location similarity (`all-MiniLM-L6-v2` optional + deterministic fallback) and language/cultural tag extraction without leaking raw PII.
- **Week 4 (35h):** Implement hierarchical `PlaceEmotionSignature` (`place_type` prior + `place_id` posterior shrinkage) so particular places diverge from type averages when supported by enough visits.
- **Week 5 (35h) — Midterm Prep:** Implement data-driven weight learner (`NNLS` / constrained Ridge on proximity dimensions) with leave-place-out & leave-user-out CV.
- **Week 6 (35h):** Integrate variance decomposition (`group` / `type|group` / `place|type` / `user` / `residual`) and significance diagnostics (permutation nulls + cluster bootstrap CIs).
- **Week 7 (35h):** Wire read-only analytics API endpoint(s) (e.g., per-user `place_id` vs. `place_type` signatures and recovery anchors) with login validation and Flask client tests.
- **Week 8 (35h):** Extend `place_narrative_bridge.py` and trajectory scoring (`proximity_trajectory.py`) to weight edges/trends using elapsed timestamps and learned proximity weights.

#### 9.3) Project Completion, Testing, and Documentation (Weeks 9–10, ~35 Hours/Week = 70 Hours)
- **Week 9 (35h):** Re-run full benchmarks on synthetic + any mentor-provided/pilot data; stress-test ablation results, missing dimensions, and runtime memory/speed.
- **Week 10 (35h):** Finalize documentation, clean PRs, reproducible scripts, test coverage checks, and written final GSoC report / research paper draft notes in own words.

### 10) Planned GSoC Work Hours
- **Effort:** Large project — **350 hours** (~35 hours/week over 10 weeks).
- **Working Hours & Timezone:** **[FILL IN YOUR LOCAL TIMEZONE & HOURS, expressed also in Alaska Time (`AKDT`, UTC-8)]**, with daily commits/PR updates and **[PREFERRED DAY/TIME]** availability for weekly Zoom meetings.

### 11) Planned Absence / Vacation Days and Other Commitments
- **[FILL IN BY CONTRIBUTOR]:** List any university exams, classes, travel, or external commitments during Community Bonding and the summer coding period (or state "No planned absences; available full-time 35h/week").

### 12) Skill Set
- **Languages & Toolkits:** Python, `numpy`, `scipy`, `scikit-learn`, `pytest`, Flask, MongoDB (`pymongo` / `mongomock`), Git/GitHub CI (`flake8`, `black`).
- **Domain Evidence:**
  - Merged/open contributions to `KathiraveluLab/DREAMS` (`@madhavzanwar`).
  - Research prototype on `madhavzanwar/DREAMS@research/proximity-validation` demonstrating Haversine normalization, Wu-Palmer taxonomy similarity, char-3-gram TF-IDF / `all-MiniLM-L6-v2` embeddings, leave-place-out Ridge/NNLS weight learning, permutation/bootstrap inference, and nested CHIME ANOVA.
- **Code Challenge:** No separate formal challenge listed beyond prior image/data-mining experience (`uaanchorage/GSoC/README.md` Idea #2) — pointer to DREAMS PRs and `research/proximity_validation/` serves as concrete evidence.

---

## Five Concrete Questions to Ask the Mentors

1. **Data availability & format:** Does any non-sensitive, de-identified, or simulated sample dataset of real-world photo locations + timestamps + CHIME annotations already exist in the lab, or should Phase 1 rely solely on synthetic/self-collected location traces until IRB-approved data is cleared?
2. **Dimension priorities & geocoder signals:** Between geographic distance, OSM venue hierarchy (`location_extractor.py`), Places365 scene labels (`places365_classifier.py`), language, and cultural tags, which dimensions do you expect to matter most in practice—and are there specific cultural/community place tags (especially relevant to Alaska communities) that the taxonomy should capture?
3. **Privacy boundary for "this particular church" vs. "any church":** To model physical places (`place_id`) without exposing sensitive user coordinates, is clustering GPS coordinates per user into opaque place clusters (`cluster_places_dbscan` / geohash cell + venue name) acceptable for `PlaceVisit`, or should `place_id` stay ephemeral inside per-user analytics?
4. **Global vs. individualized proximity weights:** In our prototype, individual place divergence (`place_strength`) lowers type-level predictive accuracy sharply. In DREAMS, do you envision learning one population-wide weight vector `{'geo', 'cat', 'ling', 'cult'}`, or user-personalized weights when a user has sufficient visit history?
5. **App integration priority:** Besides the research evaluation, should GSoC deliverables prioritize wiring the place-instance signature and learned weights into `place_narrative_bridge.py` and read-only Flask analytics endpoints (`/api/analytics/...`), or keeping the analytics pipeline cleanly decoupled until validated on pilot data?
