# Proximity Validation Experiment Summary

Total runs: `135` across `27` configurations (fixed defaults: `n_places=60`, `n_cities=4`, `n_users=40`, `mean_visits_per_user=8.0`, `group_strength=0.8`, `user_strength=0.5`, `noise_sd=0.3`, `min_visits=5`, `semantic_backend='tfidf'`).

## Mean Spearman & Variance Decomposition by Configuration

| geo_corr | cat_str | place_str | M0 (Geo) | M1 (CatTree) | M2 (SemName) | M3 (Repo) | M4 (Ridge) | M5 (NNLS) | M4 R2 | Var(Type) | Var(Place\|Type) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 0.013 | 0.638 | 0.139 | 0.142 | 0.440 | 0.516 | 0.431 | 0.514 | 0.077 |
| 0.0 | 0.0 | 0.5 | 0.001 | 0.457 | 0.086 | 0.101 | 0.316 | 0.358 | 0.133 | 0.472 | 0.246 |
| 0.0 | 0.0 | 1.0 | -0.002 | 0.221 | 0.036 | 0.051 | 0.158 | 0.162 | 0.005 | 0.410 | 0.426 |
| 0.0 | 0.5 | 0.0 | -0.020 | 0.496 | 0.149 | 0.115 | 0.315 | 0.378 | 0.277 | 0.605 | 0.063 |
| 0.0 | 0.5 | 0.5 | 0.008 | 0.363 | 0.103 | 0.104 | 0.251 | 0.286 | 0.098 | 0.531 | 0.214 |
| 0.0 | 0.5 | 1.0 | 0.014 | 0.195 | 0.048 | 0.061 | 0.147 | 0.155 | 0.007 | 0.433 | 0.407 |
| 0.0 | 1.0 | 0.0 | -0.030 | 0.316 | 0.163 | 0.092 | 0.212 | 0.225 | 0.192 | 0.751 | 0.039 |
| 0.0 | 1.0 | 0.5 | -0.002 | 0.267 | 0.139 | 0.096 | 0.175 | 0.196 | 0.092 | 0.679 | 0.141 |
| 0.0 | 1.0 | 1.0 | 0.016 | 0.182 | 0.090 | 0.073 | 0.149 | 0.140 | 0.021 | 0.545 | 0.322 |
| 0.5 | 0.0 | 0.0 | -0.013 | 0.635 | 0.132 | 0.132 | 0.444 | 0.526 | 0.392 | 0.566 | 0.074 |
| 0.5 | 0.0 | 0.5 | -0.011 | 0.410 | 0.080 | 0.084 | 0.290 | 0.353 | 0.120 | 0.466 | 0.247 |
| 0.5 | 0.0 | 1.0 | -0.010 | 0.185 | 0.029 | 0.027 | 0.125 | 0.150 | 0.000 | 0.361 | 0.459 |
| 0.5 | 0.5 | 0.0 | -0.019 | 0.513 | 0.162 | 0.119 | 0.351 | 0.407 | 0.280 | 0.650 | 0.059 |
| 0.5 | 0.5 | 0.5 | -0.031 | 0.357 | 0.107 | 0.071 | 0.259 | 0.290 | 0.108 | 0.542 | 0.210 |
| 0.5 | 0.5 | 1.0 | -0.028 | 0.194 | 0.045 | 0.024 | 0.142 | 0.162 | 0.009 | 0.407 | 0.423 |
| 0.5 | 1.0 | 0.0 | -0.009 | 0.339 | 0.162 | 0.116 | 0.222 | 0.235 | 0.199 | 0.757 | 0.040 |
| 0.5 | 1.0 | 0.5 | -0.022 | 0.275 | 0.130 | 0.089 | 0.181 | 0.180 | 0.105 | 0.660 | 0.155 |
| 0.5 | 1.0 | 1.0 | -0.029 | 0.181 | 0.067 | 0.040 | 0.126 | 0.125 | 0.026 | 0.513 | 0.346 |
| 1.0 | 0.0 | 0.0 | 0.066 | 0.656 | 0.182 | 0.158 | 0.487 | 0.621 | 0.559 | 0.610 | 0.060 |
| 1.0 | 0.0 | 0.5 | 0.029 | 0.529 | 0.153 | 0.115 | 0.399 | 0.513 | 0.241 | 0.500 | 0.224 |
| 1.0 | 0.0 | 1.0 | -0.010 | 0.272 | 0.086 | 0.044 | 0.204 | 0.260 | 0.035 | 0.366 | 0.458 |
| 1.0 | 0.5 | 0.0 | 0.053 | 0.555 | 0.213 | 0.129 | 0.439 | 0.477 | 0.368 | 0.703 | 0.045 |
| 1.0 | 0.5 | 0.5 | 0.027 | 0.432 | 0.172 | 0.096 | 0.332 | 0.379 | 0.181 | 0.599 | 0.177 |
| 1.0 | 0.5 | 1.0 | -0.007 | 0.255 | 0.098 | 0.037 | 0.203 | 0.236 | 0.031 | 0.438 | 0.402 |
| 1.0 | 1.0 | 0.0 | 0.062 | 0.374 | 0.212 | 0.122 | 0.306 | 0.300 | 0.244 | 0.809 | 0.029 |
| 1.0 | 1.0 | 0.5 | 0.065 | 0.333 | 0.195 | 0.115 | 0.285 | 0.301 | 0.155 | 0.734 | 0.117 |
| 1.0 | 1.0 | 1.0 | 0.044 | 0.241 | 0.138 | 0.075 | 0.210 | 0.233 | 0.053 | 0.577 | 0.301 |

## Interpretation: Expected Sanity Checks vs. Informative Findings

### Expected Sanity Checks (Consequences of Planted Effects)
1. **Category vs. Geography when unconfounded (`geo_corr=0.0`, `cat_str=1.0`, `place_str=0.0`):** `M1` (category tree) achieves Spearman `0.316` vs. `M0` (geography) `-0.030`, which is expected because `geo_category_correlation=0` removes spatial clustering of types.
2. **Geographic confounding (`geo_corr=1.0`, `cat_str=1.0`, `place_str=0.0`):** `M0` rises to Spearman `0.062` solely because same-type places share cities, showing geographic proximity can masquerade as categorical effect under spatial confounding.
3. **Place variance growth with `place_strength`:** `Var(Place|Type)` rises from `0.036` (`place_str=0.0`) to `0.323` (`place_str=1.0`).

### Informative Findings (Non-Trivial Behavior Beyond Planted Strengths)
1. **Type-level `M1` degrades rapidly when places have distinct identities (`place_strength=1.0`):** At `cat_str=1.0`, raising `place_str` from `0.0` to `1.0` drops `M1` Spearman from `0.343` to `0.201`, quantifying how severely type-level pooling fails when specific places diverge emotionally.
2. **Learned `M4` vs. repo hand-weighted `M3`:** Fixed repo weights (`geo=0.3, cat=0.4, ling=0.15, cult=0.15`) dilute categorical signal with geographic and linguistic noise when `geo_corr=0`, whereas leave-place-out Ridge (`M4`) and NNLS (`M5`) down-weight non-predictive dimensions without overfitting.
3. **Parent-group baseline signal:** Even when `category_strength=0.0`, default `group_strength=0.8` keeps `M1` (taxonomy Wu-Palmer) predictive because places within the same parent group (`religious`, `healthcare`, etc.) still share latent CHIME effects.
