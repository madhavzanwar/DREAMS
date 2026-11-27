"""Unit tests for the stored-post to PlaceVisit adapter."""

from datetime import datetime, timedelta, timezone

import pytest

from dreams_app.core.extra.place_visit_adapter import posts_to_place_visits
from dreams_app.core.extra.proximity_trajectory import CHIME_DIMENSIONS, build_place_trajectories


def _chime(hope=0.6, connectedness=0.3):
    return {
        "label": "Hope",
        "score": hope,
        "scores": {"Connectedness": connectedness, "Hope": hope, "Identity": 0.05, "Meaning": 0.03, "Empowerment": 0.02},
    }


def _post(ts, loc="park", chime=None, **extra):
    doc = {"timestamp": ts, "chime_analysis": chime if chime is not None else _chime()}
    if loc is not None:
        doc["location"] = {"place_category": loc} if isinstance(loc, str) else loc
    doc.update(extra)
    return doc


def test_full_chime_distribution_and_place_category_preference():
    post = _post(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc), {"lat": 61.2, "place_category": "Park", "place_type": "leisure"}, _chime(0.7, 0.2))
    visits = posts_to_place_visits([post])
    assert len(visits) == 1
    assert visits[0].place_type == "park"
    assert visits[0].chime["Hope"] == pytest.approx(0.7)
    assert set(visits[0].chime) == set(CHIME_DIMENSIONS)


def test_skips_missing_none_and_gps_only_locations():
    ts = datetime(2026, 6, 1)
    posts = [_post(ts, None), _post(ts, {"lat": 61.2, "lon": -149.9}), _post(ts, {"place_type": "amenity"}), {"timestamp": ts}]
    assert [v.place_type for v in posts_to_place_visits(posts)] == ["amenity"]


def test_legacy_posts_top_label_fallback_and_corrections():
    ts = datetime(2026, 6, 1)
    posts = [
        _post(ts, "church", {"label": "Connectedness", "score": 0.85}),
        _post(ts, "church", {"label": "Hope", "score": 0.4}, corrected_label="Empowerment"),
        _post(ts, "church", {"label": "Uncategorized", "score": 0.0}),
        _post(ts, "church", {"label": "Hope", "score": 0.7}, corrected_label="None"),
    ]
    visits = posts_to_place_visits(posts)
    assert len(visits) == 2
    assert visits[0].chime == {"Connectedness": 0.85, "Hope": 0.0, "Identity": 0.0, "Meaning": 0.0, "Empowerment": 0.0}
    assert visits[1].chime == {"Connectedness": 0.0, "Hope": 0.0, "Identity": 0.0, "Meaning": 0.0, "Empowerment": 1.0}


def test_mixed_timezones_and_iso_strings_order_for_trajectories():
    ak_tz = timezone(timedelta(hours=-9))
    posts = [
        _post(datetime(2026, 6, 1, 10, 0, tzinfo=ak_tz), "park", _chime(0.8)),
        _post("2026-06-01T12:00:00Z", "park", _chime(0.4)),
        _post(datetime(2026, 6, 1, 15, 0), "park", _chime(0.6)),
        _post("bad-ts", "park", _chime(0.9)),
    ]
    visits = posts_to_place_visits(posts)
    assert [v.chime["Hope"] for v in visits] == [0.4, 0.6, 0.8]
    assert all(v.timestamp.tzinfo == timezone.utc for v in visits)
    assert build_place_trajectories(visits)["park"].trend["Hope"] == pytest.approx(0.2)
