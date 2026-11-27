"""Integration tests for GET /api/analytics/place-emotion/<user_id>."""

from datetime import datetime, timezone

USER_ID = "5f0000000000000000000001"


def _log_in(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        "_id": USER_ID,
        "username": "alice",
        "email": "alice@example.com",
        "password_hash": "unused",
    }
    with client.session_transaction() as session:
        session["_user_id"] = USER_ID


def _visit_post(day, place, hope, conn):
    return {
        "user_id": "alice",
        "timestamp": datetime(2026, 6, day, 12, 0, tzinfo=timezone.utc),
        "location": {"place_category": place},
        "chime_analysis": {
            "label": "Hope",
            "score": hope,
            "scores": {"Connectedness": conn, "Hope": hope, "Identity": 0.2, "Meaning": 0.3, "Empowerment": 0.2},
        },
    }


def test_place_emotion_requires_login(client):
    resp = client.get("/api/analytics/place-emotion/alice")
    assert resp.status_code == 302
    assert "/auth/login" in resp.headers["Location"]


def test_place_emotion_invalid_user_id_returns_400(client, mock_mongo):
    _log_in(client, mock_mongo)
    resp = client.get(f"/api/analytics/place-emotion/{'a' * 65}")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid user_id format"


def test_place_emotion_missing_user_returns_404(client, mock_mongo):
    _log_in(client, mock_mongo)
    mock_mongo.__getitem__.return_value.find.return_value.sort.return_value = []
    resp = client.get("/api/analytics/place-emotion/nobody")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "No posts found for user", "user_id": "nobody"}


def test_place_emotion_posts_without_locations_returns_empty(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = [{"user_id": "alice", "timestamp": datetime(2026, 6, 1, tzinfo=timezone.utc)}]
    mock_mongo.__getitem__.return_value.find.return_value.sort.return_value = posts
    resp = client.get("/api/analytics/place-emotion/alice")
    assert resp.status_code == 200
    assert resp.get_json() == {"user_id": "alice", "visit_count": 0, "place_signatures": {}, "recovery_anchors": []}


def test_place_emotion_returns_signatures_and_recovery_anchors(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = [
        _visit_post(1, "church", 0.65, 0.7),
        _visit_post(2, "church", 0.75, 0.75),
        _visit_post(3, "church", 0.85, 0.8),
        _visit_post(4, "hospital", 0.25, 0.2),
        _visit_post(5, "hospital", 0.2, 0.2),
    ]
    mock_mongo.__getitem__.return_value.find.return_value.sort.return_value = posts
    resp = client.get("/api/analytics/place-emotion/alice")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["visit_count"] == 5
    assert data["place_signatures"]["church"]["visit_count"] == 3
    assert data["place_signatures"]["church"]["trend_direction"] == "improving"
    assert [a["place_type"] for a in data["recovery_anchors"]] == ["church"]


def test_place_emotion_internal_error_does_not_leak_details(client, mock_mongo):
    _log_in(client, mock_mongo)
    mock_mongo.__getitem__.return_value.find.side_effect = RuntimeError("mongodb://secret:pass@host")
    resp = client.get("/api/analytics/place-emotion/alice")
    assert resp.status_code == 500
    assert resp.get_json() == {"error": "Failed to compute place-emotion analytics"}
    assert "secret" not in resp.get_data(as_text=True)
