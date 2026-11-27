"""Deterministic synthetic dataset generator with known CHIME ground truth."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreams_app.core.extra.place_emotion_signature import CHIME_DIMENSIONS  # noqa: E402
from dreams_app.core.extra.proximity_trajectory import PlaceVisit  # noqa: E402


PLACE_TAXONOMY_TREE: Dict[str, Tuple[str, ...]] = {
    "religious": ("church", "temple", "mosque"),
    "healthcare": ("hospital", "clinic", "pharmacy"),
    "recreation": ("park", "garden", "beach"),
    "food": ("restaurant", "cafe", "bar"),
    "civic": ("library", "school", "community_center"),
}


@dataclass(frozen=True)
class TaxonomyNode:
    """Node in the place-type taxonomy tree with parent link and depth."""

    name: str
    parent: Optional[str]
    depth: int
    children: Tuple[str, ...] = ()


def build_taxonomy_nodes() -> Dict[str, TaxonomyNode]:
    """Return the full taxonomy tree mapping node name to TaxonomyNode."""
    nodes: Dict[str, TaxonomyNode] = {
        "root": TaxonomyNode(
            name="root",
            parent=None,
            depth=0,
            children=tuple(PLACE_TAXONOMY_TREE.keys()),
        )
    }
    for group_name, leaf_types in PLACE_TAXONOMY_TREE.items():
        nodes[group_name] = TaxonomyNode(
            name=group_name,
            parent="root",
            depth=1,
            children=leaf_types,
        )
        for leaf in leaf_types:
            nodes[leaf] = TaxonomyNode(
                name=leaf,
                parent=group_name,
                depth=2,
                children=(),
            )
    return nodes


TAXONOMY_NODES: Dict[str, TaxonomyNode] = build_taxonomy_nodes()
TAXONOMY_DEPTHS: Dict[str, int] = {
    name: node.depth for name, node in TAXONOMY_NODES.items()
}
LEAF_TYPES: Tuple[str, ...] = tuple(
    leaf for leaf_types in PLACE_TAXONOMY_TREE.values() for leaf in leaf_types
)
LEAF_TO_GROUP: Dict[str, str] = {
    leaf: group_name
    for group_name, leaf_types in PLACE_TAXONOMY_TREE.items()
    for leaf in leaf_types
}

_CITY_TEMPLATES: Tuple[Dict[str, object], ...] = (
    {
        "city": "Anchorage",
        "latitude": 61.2181,
        "longitude": -149.9003,
        "language": "english",
        "tags": ("alaskan", "northern", "community", "coastal"),
    },
    {
        "city": "Seattle",
        "latitude": 47.6062,
        "longitude": -122.3321,
        "language": "english",
        "tags": ("pacific_northwest", "urban", "coastal", "civic"),
    },
    {
        "city": "Lisbon",
        "latitude": 38.7223,
        "longitude": -9.1393,
        "language": "portuguese",
        "tags": ("european", "historic", "lusophone", "coastal"),
    },
    {
        "city": "Montreal",
        "latitude": 45.5017,
        "longitude": -73.5673,
        "language": "french",
        "tags": ("francophone", "historic", "urban", "multicultural"),
    },
    {
        "city": "Madrid",
        "latitude": 40.4168,
        "longitude": -3.7038,
        "language": "spanish",
        "tags": ("hispanic", "european", "historic", "urban"),
    },
    {
        "city": "Osaka",
        "latitude": 34.6937,
        "longitude": 135.5023,
        "language": "japanese",
        "tags": ("east_asian", "urban", "historic", "coastal"),
    },
)

_GROUP_TAGS: Dict[str, Tuple[str, ...]] = {
    "religious": ("spiritual", "heritage", "quiet"),
    "healthcare": ("care", "wellness", "support"),
    "recreation": ("outdoor", "green", "public"),
    "food": ("hospitality", "social", "local"),
    "civic": ("learning", "public", "neighborhood"),
}

_NAME_PREFIXES: Tuple[str, ...] = (
    "St. Mary's",
    "Northside",
    "Harborview",
    "Cedar Grove",
    "Oakridge",
    "Sunrise",
    "Lakeside",
    "Downtown",
    "Greenfield",
    "Riverside",
    "Valley View",
    "Central",
)

_TYPE_DISPLAY_NOUNS: Dict[str, str] = {
    "church": "Church",
    "temple": "Temple",
    "mosque": "Mosque",
    "hospital": "Hospital",
    "clinic": "Clinic",
    "pharmacy": "Pharmacy",
    "park": "Park",
    "garden": "Garden",
    "beach": "Beach",
    "restaurant": "Restaurant",
    "cafe": "Cafe",
    "bar": "Bar",
    "library": "Library",
    "school": "School",
    "community_center": "Community Center",
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic place and visit emotion generation."""

    n_places: int = 60
    n_cities: int = 4
    n_users: int = 40
    mean_visits_per_user: float = 8.0
    geo_category_correlation: float = 0.0
    category_strength: float = 1.0
    group_strength: float = 0.8
    place_strength: float = 0.8
    user_strength: float = 0.5
    noise_sd: float = 0.3
    city_spread_deg: float = 0.04
    base_timestamp: str = "2026-01-01T00:00:00+00:00"
    horizon_days: int = 180

    def __post_init__(self) -> None:
        if self.n_places < 1:
            raise ValueError("n_places must be at least 1")
        if self.n_cities < 1:
            raise ValueError("n_cities must be at least 1")
        if self.n_users < 1:
            raise ValueError("n_users must be at least 1")
        if self.mean_visits_per_user <= 0:
            raise ValueError("mean_visits_per_user must be positive")
        if not 0.0 <= self.geo_category_correlation <= 1.0:
            raise ValueError("geo_category_correlation must be in [0, 1]")
        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative")
        if self.city_spread_deg < 0.0:
            raise ValueError("city_spread_deg must be non-negative")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")


@dataclass(frozen=True)
class PlaceRecord:
    """Synthetic physical place with geographic, categorical, and cultural attributes."""

    id: str
    name: str
    type: str
    group: str
    city: str
    latitude: float
    longitude: float
    language: str
    cultural_tags: Tuple[str, ...]

    def to_place_dict(self) -> Dict[str, object]:
        """Convert to a dict compatible with composite_proximity."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "group": self.group,
            "city": self.city,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "language": self.language,
            "cultural_tags": list(self.cultural_tags),
        }


@dataclass(frozen=True)
class SyntheticVisit:
    """Single user visit with generated CHIME scores and underlying logits."""

    visit_id: str
    user_id: str
    place_id: str
    place_type: str
    place_group: str
    timestamp: datetime
    chime: Dict[str, float]
    logits: Dict[str, float]


@dataclass(frozen=True)
class GroundTruthEffects:
    """Planted latent effect vectors across CHIME_DIMENSIONS."""

    category_effect: Dict[str, Dict[str, float]]
    group_effect: Dict[str, Dict[str, float]]
    place_effect: Dict[str, Dict[str, float]]
    user_effect: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class SyntheticDataset:
    """Complete synthetic benchmark holding config, places, visits, and ground truth."""

    config: GeneratorConfig
    taxonomy: Mapping[str, TaxonomyNode]
    places: List[PlaceRecord]
    visits: List[SyntheticVisit]
    effects: GroundTruthEffects


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _vec_to_chime_dict(vec: np.ndarray) -> Dict[str, float]:
    return {dim: float(vec[idx]) for idx, dim in enumerate(CHIME_DIMENSIONS)}


def _build_cities(n_cities: int) -> List[Dict[str, object]]:
    languages = ("english", "portuguese", "french", "spanish", "japanese")
    cities: List[Dict[str, object]] = []
    for idx in range(n_cities):
        if idx < len(_CITY_TEMPLATES):
            cities.append(dict(_CITY_TEMPLATES[idx]))
        else:
            cities.append(
                {
                    "city": f"City_{idx + 1}",
                    "latitude": float(((idx * 17) % 100) - 50),
                    "longitude": float(((idx * 43) % 240) - 120),
                    "language": languages[idx % len(languages)],
                    "tags": (f"region_{idx + 1}", "urban", "community"),
                }
            )
    return cities


def _build_places(
    config: GeneratorConfig, rng: np.random.Generator
) -> List[PlaceRecord]:
    cities = _build_cities(config.n_cities)
    type_home_city = {
        leaf: int(city_idx)
        for leaf, city_idx in zip(
            LEAF_TYPES,
            rng.permutation(np.arange(len(LEAF_TYPES)) % config.n_cities),
        )
    }
    raw_types = [LEAF_TYPES[i % len(LEAF_TYPES)] for i in range(config.n_places)]
    assigned_types = [raw_types[idx] for idx in rng.permutation(config.n_places)]

    type_counters: Dict[str, int] = {}
    places: List[PlaceRecord] = []
    for idx, place_type in enumerate(assigned_types):
        group_name = LEAF_TO_GROUP[place_type]
        if rng.random() < config.geo_category_correlation:
            city_idx = type_home_city[place_type]
        else:
            city_idx = int(rng.integers(0, config.n_cities))
        city = cities[city_idx]

        lat = float(
            np.clip(
                city["latitude"] + rng.normal(0.0, config.city_spread_deg), -90.0, 90.0
            )
        )
        lon = float(
            np.clip(
                city["longitude"] + rng.normal(0.0, config.city_spread_deg),
                -180.0,
                180.0,
            )
        )

        occurrence = type_counters.get(place_type, 0)
        type_counters[place_type] = occurrence + 1
        prefix = _NAME_PREFIXES[(idx + occurrence) % len(_NAME_PREFIXES)]
        noun = _TYPE_DISPLAY_NOUNS[place_type]
        suffix = "" if occurrence < len(_NAME_PREFIXES) else f" #{occurrence + 1}"
        name = f"{prefix} {noun}{suffix}"

        tag_pool = list(
            dict.fromkeys((*city["tags"], *_GROUP_TAGS[group_name], place_type))
        )
        n_tags = int(rng.integers(0, min(3, len(tag_pool)) + 1))
        if n_tags == 0:
            tags: Tuple[str, ...] = ()
        else:
            chosen = rng.choice(len(tag_pool), size=n_tags, replace=False)
            tags = tuple(tag_pool[int(k)] for k in chosen)

        places.append(
            PlaceRecord(
                id=f"place_{idx:03d}",
                name=name,
                type=place_type,
                group=group_name,
                city=str(city["city"]),
                latitude=lat,
                longitude=lon,
                language=str(city["language"]),
                cultural_tags=tags,
            )
        )
    return places


def _draw_effects(
    places: List[PlaceRecord],
    user_ids: List[str],
    rng: np.random.Generator,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    n_dims = len(CHIME_DIMENSIONS)
    cat_arr = {leaf: rng.normal(0.0, 1.0, size=n_dims) for leaf in LEAF_TYPES}
    grp_arr = {grp: rng.normal(0.0, 1.0, size=n_dims) for grp in PLACE_TAXONOMY_TREE}
    plc_arr = {p.id: rng.normal(0.0, 1.0, size=n_dims) for p in places}
    usr_arr = {uid: rng.normal(0.0, 1.0, size=n_dims) for uid in user_ids}
    return cat_arr, grp_arr, plc_arr, usr_arr


def generate_dataset(
    config: Optional[GeneratorConfig] = None, seed: int = 42
) -> SyntheticDataset:
    """Generate deterministic places, CHIME visits, and ground-truth effect tables."""
    cfg = config if config is not None else GeneratorConfig()
    rng = np.random.default_rng(seed)

    places = _build_places(cfg, rng)
    user_ids = [f"user_{idx:03d}" for idx in range(cfg.n_users)]
    cat_arr, grp_arr, plc_arr, usr_arr = _draw_effects(places, user_ids, rng)

    base_dt = datetime.fromisoformat(cfg.base_timestamp)
    if base_dt.tzinfo is None:
        base_dt = base_dt.replace(tzinfo=timezone.utc)
    horizon_seconds = max(1, cfg.horizon_days * 86400)
    n_dims = len(CHIME_DIMENSIONS)

    raw_visits: List[Tuple[datetime, str, PlaceRecord, np.ndarray, np.ndarray]] = []
    for uid in user_ids:
        n_visits = int(rng.poisson(cfg.mean_visits_per_user))
        for _ in range(n_visits):
            place = places[int(rng.integers(0, len(places)))]
            offset_sec = int(rng.integers(0, horizon_seconds))
            ts = base_dt + timedelta(seconds=offset_sec)
            noise = (
                rng.normal(0.0, cfg.noise_sd, size=n_dims)
                if cfg.noise_sd > 0
                else np.zeros(n_dims)
            )
            logits = (
                cfg.category_strength * cat_arr[place.type]
                + cfg.group_strength * grp_arr[place.group]
                + cfg.place_strength * plc_arr[place.id]
                + cfg.user_strength * usr_arr[uid]
                + noise
            )
            probs = _softmax(logits)
            raw_visits.append((ts, uid, place, probs, logits))

    raw_visits.sort(key=lambda item: (item[0], item[1], item[2].id))
    visits = [
        SyntheticVisit(
            visit_id=f"visit_{idx:05d}",
            user_id=uid,
            place_id=place.id,
            place_type=place.type,
            place_group=place.group,
            timestamp=ts,
            chime=_vec_to_chime_dict(probs),
            logits=_vec_to_chime_dict(logits),
        )
        for idx, (ts, uid, place, probs, logits) in enumerate(raw_visits)
    ]

    effects = GroundTruthEffects(
        category_effect={k: _vec_to_chime_dict(v) for k, v in cat_arr.items()},
        group_effect={k: _vec_to_chime_dict(v) for k, v in grp_arr.items()},
        place_effect={k: _vec_to_chime_dict(v) for k, v in plc_arr.items()},
        user_effect={k: _vec_to_chime_dict(v) for k, v in usr_arr.items()},
    )

    return SyntheticDataset(
        config=cfg,
        taxonomy=TAXONOMY_NODES,
        places=places,
        visits=visits,
        effects=effects,
    )


def export_to_place_visits(dataset: SyntheticDataset) -> List[PlaceVisit]:
    """Export synthetic visits as dreams_app.core.extra.proximity_trajectory.PlaceVisit objects."""
    return [
        PlaceVisit(
            place_type=visit.place_type,
            timestamp=visit.timestamp,
            chime=dict(visit.chime),
        )
        for visit in dataset.visits
    ]


def export_chime_by_type(
    dataset: SyntheticDataset,
) -> Dict[str, List[Dict[str, float]]]:
    """Group CHIME visit dicts by place_type for build_place_signature()."""
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for visit in dataset.visits:
        grouped.setdefault(visit.place_type, []).append(dict(visit.chime))
    return grouped
