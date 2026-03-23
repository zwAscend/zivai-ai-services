from flask import Blueprint, jsonify

bp = Blueprint("health", __name__)

@bp.get("/api/v1/health")
def health():
    return jsonify({
        "status": "ok",
        "contract": "zivai_ai_v1",
        "message": "ZivAI AI services are running",
    }), 200


@bp.get("/api/v1/agents/health-check")
def agents_health_check():
    return jsonify({
        "status": "active",
        "contract": "zivai_ai_v1",
        "message": "ZivAI AI agent services are running",
    }), 200
