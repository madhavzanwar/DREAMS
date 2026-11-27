"""Pure multi-dimensional place proximity and similarity functions."""

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreams_app.core.extra.proximity_calculator import (  # noqa: E402
    categorical_proximity,
    cultural_similarity,
    linguistic_similarity,
    normalize_geographic_distance,
)
from dreams_app.location_proximity import calculate_distance  # noqa: E402
from research.proximity_validation.generate import (  # noqa: E402
    TAXONOMY_NODES,
    TaxonomyNode,
)

REPO_DEFAULT_WEIGHTS: Dict[str, float] = {
    "geo": 0.3,
    "cat": 0.4,
    "ling": 0.15,
    "cult": 0.15,
}
VALID_SEMANTIC_BACKENDS = ("auto", "tfidf", "sentence_transformers")
VALID_DIMENSIONS = ("geo", "cat", "category_tree", "semantic_name", "ling", "cult")

_ST_MODEL_CACHE: Dict[str, Any] = {}


def _get_field(place: Any, field: str, default: Any = None) -> Any:
    if isinstance(place, Mapping):
        return place.get(field, default)
    return getattr(place, field, default)


def _validate_place_obj(place: Any) -> None:
    if not isinstance(place, Mapping) and not hasattr(place, "__dict__"):
        raise TypeError(f"Expected Mapping or place object, got {type(place).__name__}")


def _validate_coord(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    coord = float(value)
    if not math.isfinite(coord) or not (low <= coord <= high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {coord!r}")
    return coord


def _clamp_unit(score: float) -> float:
    return float(min(1.0, max(0.0, score)))


def geographic_proximity(
    place1: Any,
    place2: Any,
    max_distance_km: float = 500.0,
) -> float:
    """Return geographic proximity in (0, 1] using Haversine and repo exponential decay."""
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    lat1 = _validate_coord(_get_field(place1, "latitude"), "latitude", -90.0, 90.0)
    lon1 = _validate_coord(_get_field(place1, "longitude"), "longitude", -180.0, 180.0)
    lat2 = _validate_coord(_get_field(place2, "latitude"), "latitude", -90.0, 90.0)
    lon2 = _validate_coord(_get_field(place2, "longitude"), "longitude", -180.0, 180.0)
    dist_km = calculate_distance(lat1, lon1, lat2, lon2) / 1000.0
    return _clamp_unit(
        normalize_geographic_distance(dist_km, max_distance_km=max_distance_km)
    )


def _ancestors(node_name: str, taxonomy: Mapping[str, TaxonomyNode]) -> List[str]:
    chain: List[str] = []
    cur: Optional[str] = node_name
    while cur is not None:
        if cur not in taxonomy:
            raise ValueError(f"Broken taxonomy link for node {cur!r}")
        chain.append(cur)
        cur = taxonomy[cur].parent
    return chain


def category_tree_similarity(
    place1: Any,
    place2: Any,
    taxonomy: Mapping[str, TaxonomyNode] = TAXONOMY_NODES,
) -> float:
    """Compute Wu-Palmer similarity over the generator taxonomy tree.

    Compared to ``dreams_app.core.extra.proximity_calculator.categorical_proximity``,
    which only returns ``0.5`` for three hard-coded sibling pairs
    (``clinic/hospital``, ``church/temple``, ``cafe/restaurant``) and ``0.0`` for
    everything else, Wu-Palmer similarity over the 3-level taxonomy tree scores
    identical types at ``1.0``, all within-group siblings (e.g., ``church/mosque``,
    ``park/garden``, ``library/school``) at ``0.5``, and cross-group pairs at ``0.0``.
    """
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    t1_raw = _get_field(place1, "type")
    t2_raw = _get_field(place2, "type")
    if not isinstance(t1_raw, str) or not t1_raw.strip():
        raise ValueError(f"place1 missing valid 'type': {t1_raw!r}")
    if not isinstance(t2_raw, str) or not t2_raw.strip():
        raise ValueError(f"place2 missing valid 'type': {t2_raw!r}")

    t1 = t1_raw.lower().strip()
    t2 = t2_raw.lower().strip()
    if t1 not in taxonomy:
        raise ValueError(f"Unknown place type {t1!r}")
    if t2 not in taxonomy:
        raise ValueError(f"Unknown place type {t2!r}")
    if t1 == t2:
        return 1.0

    anc1 = set(_ancestors(t1, taxonomy))
    lca_name = next((anc for anc in _ancestors(t2, taxonomy) if anc in anc1), None)
    if lca_name is None:
        return 0.0

    depth_sum = taxonomy[t1].depth + taxonomy[t2].depth
    if depth_sum == 0:
        return 1.0
    return _clamp_unit((2.0 * taxonomy[lca_name].depth) / depth_sum)


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _encode_sentence_transformers(names: Sequence[str]) -> np.ndarray:
    model_name = "all-MiniLM-L6-v2"
    if model_name not in _ST_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _ST_MODEL_CACHE[model_name]
    vectors = np.asarray(
        model.encode(list(names), normalize_embeddings=True), dtype=float
    )
    return vectors


def _encode_tfidf_char3(names: Sequence[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        lowercase=True,
        norm="l2",
    )
    return vectorizer.fit_transform(names).toarray()


def _encode_names(names: Sequence[str], backend: str) -> np.ndarray:
    if backend not in VALID_SEMANTIC_BACKENDS:
        raise ValueError(
            f"backend must be one of {VALID_SEMANTIC_BACKENDS}, got {backend!r}"
        )
    if backend == "tfidf":
        return _encode_tfidf_char3(names)
    if backend == "sentence_transformers":
        if not _sentence_transformers_available():
            raise ImportError("sentence-transformers is not installed")
        return _encode_sentence_transformers(names)
    if _sentence_transformers_available():
        try:
            return _encode_sentence_transformers(names)
        except Exception:
            return _encode_tfidf_char3(names)
    return _encode_tfidf_char3(names)


def _validate_name(place: Any, label: str) -> str:
    raw = _get_field(place, "name")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} missing valid 'name': {raw!r}")
    return raw.strip()


def semantic_name_similarity(
    place1: Any,
    place2: Any,
    backend: str = "auto",
) -> float:
    """Return cosine similarity of place-name embeddings in [0, 1]."""
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    name1 = _validate_name(place1, "place1")
    name2 = _validate_name(place2, "place2")
    if backend not in VALID_SEMANTIC_BACKENDS:
        raise ValueError(
            f"backend must be one of {VALID_SEMANTIC_BACKENDS}, got {backend!r}"
        )
    if name1.lower() == name2.lower():
        return 1.0
    embeddings = _encode_names([name1, name2], backend=backend)
    denom = float(np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    if denom == 0.0:
        return 0.0
    return _clamp_unit(float(np.dot(embeddings[0], embeddings[1]) / denom))


def linguistic_proximity(place1: Any, place2: Any) -> float:
    """Return linguistic similarity in {0.0, 1.0} reusing the repo function."""
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    lang1 = _get_field(place1, "language")
    lang2 = _get_field(place2, "language")
    if not isinstance(lang1, str) or not lang1.strip():
        raise ValueError(f"place1 missing valid 'language': {lang1!r}")
    if not isinstance(lang2, str) or not lang2.strip():
        raise ValueError(f"place2 missing valid 'language': {lang2!r}")
    return _clamp_unit(linguistic_similarity(lang1, lang2))


def _extract_tags(place: Any, label: str) -> List[str]:
    tags = _get_field(place, "cultural_tags")
    if (
        tags is None
        or isinstance(tags, (str, bytes))
        or not isinstance(tags, (list, tuple, set))
    ):
        raise ValueError(
            f"{label} 'cultural_tags' must be a list/tuple/set of strings, got {tags!r}"
        )
    cleaned: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError(f"{label} cultural tag must be str, got {tag!r}")
        cleaned.append(tag)
    return cleaned


def cultural_proximity(place1: Any, place2: Any) -> float:
    """Return cultural tag Jaccard similarity in [0, 1] reusing the repo function."""
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    tags1 = _extract_tags(place1, "place1")
    tags2 = _extract_tags(place2, "place2")
    return _clamp_unit(cultural_similarity(tags1, tags2))


def _validate_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    if weights is None:
        return dict(REPO_DEFAULT_WEIGHTS)
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError(
            "weights must be a non-empty mapping of dimension names to weights"
        )
    cleaned: Dict[str, float] = {}
    for dim, val in weights.items():
        if dim not in VALID_DIMENSIONS:
            raise ValueError(
                f"Unknown proximity dimension {dim!r}; expected subset of {VALID_DIMENSIONS}"
            )
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Weight for {dim!r} must be numeric, got {val!r}")
        weight = float(val)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Weight for {dim!r} must be finite and non-negative, got {weight!r}"
            )
        cleaned[dim] = weight
    return cleaned


def _maybe_geo_score(
    place1: Any, place2: Any, max_distance_km: float
) -> Optional[float]:
    geo_vals = [
        _get_field(p, "geo_proximity")
        for p in (place1, place2)
        if _get_field(p, "geo_proximity") is not None
    ]
    if geo_vals:
        parsed = [_validate_coord(v, "geo_proximity", 0.0, 1.0) for v in geo_vals]
        return float(sum(parsed) / len(parsed))
    lat1 = _get_field(place1, "latitude")
    lon1 = _get_field(place1, "longitude")
    lat2 = _get_field(place2, "latitude")
    lon2 = _get_field(place2, "longitude")
    if None in (lat1, lon1, lat2, lon2):
        return None
    return geographic_proximity(place1, place2, max_distance_km=max_distance_km)


def _dimension_score(
    dim: str,
    place1: Any,
    place2: Any,
    semantic_backend: str,
    max_distance_km: float,
) -> Optional[float]:
    if dim == "geo":
        return _maybe_geo_score(place1, place2, max_distance_km)
    if dim == "cat":
        t1, t2 = _get_field(place1, "type"), _get_field(place2, "type")
        return categorical_proximity(t1, t2) if (t1 and t2) else None
    if dim == "category_tree":
        t1, t2 = _get_field(place1, "type"), _get_field(place2, "type")
        return category_tree_similarity(place1, place2) if (t1 and t2) else None
    if dim == "semantic_name":
        n1, n2 = _get_field(place1, "name"), _get_field(place2, "name")
        return (
            semantic_name_similarity(place1, place2, backend=semantic_backend)
            if (
                isinstance(n1, str)
                and n1.strip()
                and isinstance(n2, str)
                and n2.strip()
            )
            else None
        )
    if dim == "ling":
        l1, l2 = _get_field(place1, "language"), _get_field(place2, "language")
        return linguistic_similarity(l1, l2) if (l1 and l2) else None
    if dim == "cult":
        c1, c2 = _get_field(place1, "cultural_tags"), _get_field(
            place2, "cultural_tags"
        )
        return cultural_similarity(list(c1), list(c2)) if (c1 and c2) else None
    raise ValueError(f"Unsupported dimension {dim!r}")


def composite_similarity(
    place1: Any,
    place2: Any,
    weights: Optional[Mapping[str, float]] = None,
    semantic_backend: str = "auto",
    max_distance_km: float = 500.0,
) -> float:
    """Combine any subset of dimensions, renormalizing weights over present dimensions."""
    _validate_place_obj(place1)
    _validate_place_obj(place2)
    weight_map = _validate_weights(weights)

    scores: Dict[str, float] = {}
    for dim in weight_map:
        score = _dimension_score(dim, place1, place2, semantic_backend, max_distance_km)
        if score is not None:
            scores[dim] = score

    total_weight = sum(weight_map[dim] for dim in scores)
    if total_weight == 0.0:
        return 0.0
    combined = (
        sum(weight_map[dim] * score for dim, score in scores.items()) / total_weight
    )
    return _clamp_unit(combined)


def _semantic_name_matrix(places: Sequence[Any], backend: str) -> np.ndarray:
    names = [_validate_name(p, f"places[{idx}]") for idx, p in enumerate(places)]
    embeddings = _encode_names(names, backend=backend)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = embeddings / safe_norms
    mat = normalized @ normalized.T
    mat = np.clip(mat, 0.0, 1.0)
    np.fill_diagonal(mat, 1.0)
    return 0.5 * (mat + mat.T)


def _pairwise_from_fn(
    places: Sequence[Any], fn: Callable[[Any, Any], float]
) -> np.ndarray:
    n = len(places)
    mat = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = fn(places[i], places[j])
            mat[i, j] = val
            mat[j, i] = val
    return mat


def pairwise_matrix(
    places: Sequence[Any],
    dimension: str = "composite",
    weights: Optional[Mapping[str, float]] = None,
    semantic_backend: str = "auto",
    max_distance_km: float = 500.0,
) -> np.ndarray:
    """Compute the full N x N similarity matrix for one dimension or composite weights."""
    if not places:
        raise ValueError("places must be non-empty")

    if dimension == "semantic_name":
        return _semantic_name_matrix(places, backend=semantic_backend)

    fn_map: Dict[str, Callable[[Any, Any], float]] = {
        "geo": lambda a, b: geographic_proximity(a, b, max_distance_km=max_distance_km),
        "cat": lambda a, b: categorical_proximity(
            _get_field(a, "type"), _get_field(b, "type")
        ),
        "category_tree": category_tree_similarity,
        "ling": linguistic_proximity,
        "cult": cultural_proximity,
        "composite": lambda a, b: composite_similarity(
            a,
            b,
            weights=weights,
            semantic_backend=semantic_backend,
            max_distance_km=max_distance_km,
        ),
    }
    if dimension not in fn_map:
        valid = (*VALID_DIMENSIONS, "composite")
        raise ValueError(f"Unknown dimension {dimension!r}; expected one of {valid}")
    return _pairwise_from_fn(places, fn_map[dimension])


def pairwise_dimension_matrices(
    places: Sequence[Any],
    dimensions: Iterable[str] = (
        "geo",
        "category_tree",
        "semantic_name",
        "ling",
        "cult",
    ),
    semantic_backend: str = "auto",
    max_distance_km: float = 500.0,
) -> Dict[str, np.ndarray]:
    """Compute a dictionary mapping each requested dimension name to its N x N matrix."""
    return {
        dim: pairwise_matrix(
            places,
            dimension=dim,
            semantic_backend=semantic_backend,
            max_distance_km=max_distance_km,
        )
        for dim in dimensions
    }
