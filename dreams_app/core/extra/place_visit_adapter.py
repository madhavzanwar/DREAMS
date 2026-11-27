"""Pure adapter converting stored MongoDB post dicts into PlaceVisit objects."""

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from dreams_app.core.extra.proximity_trajectory import CHIME_DIMENSIONS, PlaceVisit

_CHIME_LOWER = {dim.lower(): dim for dim in CHIME_DIMENSIONS}


def _safe_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _extract_place_type(post: Mapping[str, Any]) -> Optional[str]:
    loc = post.get("location")
    if not isinstance(loc, Mapping):
        return None
    for key in ("place_category", "place_type", "type"):
        raw = loc.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return None


def _parse_utc_timestamp(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    return (
        dt.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None
        else dt.astimezone(timezone.utc)
    )


def _extract_chime_vector(post: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    corrected = post.get("corrected_label")
    if isinstance(corrected, str) and corrected.strip().lower() == "none":
        return None
    chime = post.get("chime_analysis")
    scores = chime.get("scores") if isinstance(chime, Mapping) else None
    if isinstance(scores, Mapping):
        lowered = {str(k).lower(): v for k, v in scores.items()}
        if any(dim.lower() in lowered for dim in CHIME_DIMENSIONS):
            return {
                dim: (_safe_score(lowered.get(dim.lower())) or 0.0)
                for dim in CHIME_DIMENSIONS
            }
    has_corrected = isinstance(corrected, str) and bool(corrected.strip())
    chime_label = chime.get("label") if isinstance(chime, Mapping) else None
    raw_label = corrected if has_corrected else chime_label
    if not isinstance(raw_label, str):
        return None
    canonical = _CHIME_LOWER.get(raw_label.strip().lower())
    if canonical is None:
        return None
    parsed = _safe_score(chime.get("score")) if isinstance(chime, Mapping) and raw_label == chime_label else None
    score = parsed if parsed is not None else 1.0
    return {dim: (score if dim == canonical else 0.0) for dim in CHIME_DIMENSIONS}


def posts_to_place_visits(posts: Iterable[Mapping[str, Any]]) -> List[PlaceVisit]:
    """Convert stored post documents into chronologically ordered PlaceVisit objects."""
    visits: List[PlaceVisit] = []
    for post in posts:
        if not isinstance(post, Mapping):
            continue
        place_type = _extract_place_type(post)
        timestamp = _parse_utc_timestamp(post.get("timestamp"))
        chime = _extract_chime_vector(post)
        if place_type is not None and timestamp is not None and chime is not None:
            visits.append(
                PlaceVisit(place_type=place_type, timestamp=timestamp, chime=chime)
            )
    visits.sort(key=lambda visit: visit.timestamp)
    return visits
