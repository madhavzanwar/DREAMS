# dreams_app/app/analytics/routes.py

"""
API routes for graph-based narrative structure and place-emotion analysis.

Endpoints:
    GET /api/analytics/graph-metrics/<user_id>
        Returns quantitative structural metrics for the user's emotional
        narrative graph (centrality, transitions, cycles, etc.).
    GET /api/analytics/place-emotion/<user_id>
        Returns per-place CHIME signatures, trends, and recovery anchors.
"""

import logging
import re

from flask import jsonify, current_app
from flask_login import login_required
from . import bp
from dreams_app.core.extra.place_emotion_signature import (
    build_place_signature,
    detect_recovery_anchors,
    get_dominant_chime_dimension,
)
from dreams_app.core.extra.place_visit_adapter import posts_to_place_visits
from dreams_app.core.extra.proximity_trajectory import (
    build_place_trajectories,
    summarize_trajectories,
)

logger = logging.getLogger(__name__)

# Security: only allow reasonable user_id values (letters, digits, _.-@)
_USER_ID_RE = re.compile(r'^[\w.\-@]{1,64}$')


@bp.route('/graph-metrics/<string:user_id>', methods=['GET'])
@login_required
def graph_metrics(user_id: str):
    """
    Compute and return structural graph metrics for a user's emotional narrative.

    Response 200:
        JSON with ``user_id`` and ``metrics`` (graph_summary, node_metrics,
        pattern_analysis).
    Response 404:
        No posts found for the given user_id.
    Response 500:
        Internal error during analysis (details logged server-side only).
    """
    try:
        if not _USER_ID_RE.match(user_id):
            logger.warning(f"Invalid user_id format: {user_id}")
            return jsonify({'error': 'Invalid user_id format'}), 400

        mongo = current_app.mongo['posts']

        # Fetch posts sorted chronologically
        user_posts = list(
            mongo.find({'user_id': user_id}).sort('timestamp', 1)
        )

        if not user_posts:
            return jsonify({
                'error': 'No posts found for user',
                'user_id': user_id,
            }), 404

        # Delegate heavy graph execution to the global pipeline
        metrics = current_app.dreams_pipeline.generate_narrative_metrics(user_id, user_posts)

        return jsonify({
            'user_id': user_id,
            'metrics': metrics,
        })

    except Exception:
        # Log the full traceback server-side; never expose it to the client.
        logger.exception(
            "Failed to compute graph metrics for user_id=%s", user_id
        )
        return jsonify({'error': 'Failed to compute graph metrics'}), 500


@bp.route('/place-emotion/<string:user_id>', methods=['GET'])
@login_required
def place_emotion(user_id: str):
    """Return per-place CHIME signatures, trajectory summaries, and recovery anchors."""
    if not _USER_ID_RE.match(user_id):
        logger.warning("Invalid user_id format: %s", user_id)
        return jsonify({'error': 'Invalid user_id format'}), 400
    try:
        user_posts = list(current_app.mongo['posts'].find({'user_id': user_id}).sort('timestamp', 1))
        if not user_posts:
            return jsonify({'error': 'No posts found for user', 'user_id': user_id}), 404
        visits = posts_to_place_visits(user_posts)
        trajectories = build_place_trajectories(visits)
        summary = summarize_trajectories(trajectories)
        signatures = {
            pt: build_place_signature(pt, [v.chime for v in traj.visits])
            for pt, traj in trajectories.items()
        }
        anchors = detect_recovery_anchors(signatures)
        return jsonify({
            'user_id': user_id,
            'visit_count': len(visits),
            'place_signatures': {
                pt: {
                    'visit_count': sig.visit_count,
                    'dominant_dimension': get_dominant_chime_dimension(sig)[0],
                    'chime_vector': {k: round(v, 4) for k, v in sig.chime_vector.items()},
                    'volatility': round(sig.volatility, 4),
                    'trend_direction': summary[pt]['trend_direction'],
                    'dominant_trend': summary[pt]['dominant_trend'],
                    'trend_slope': summary[pt]['trend_slope'],
                }
                for pt, sig in signatures.items()
            },
            'recovery_anchors': [
                {'place_type': pt, 'anchor_score': round(score, 4), 'dominant_dimension': dim}
                for pt, score, dim in anchors
            ],
        })
    except Exception:
        logger.exception("Failed to compute place-emotion analytics for user_id=%s", user_id)
        return jsonify({'error': 'Failed to compute place-emotion analytics'}), 500
