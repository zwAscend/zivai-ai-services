from flask import Blueprint, jsonify, request

from asag_engine.db.session import get_session
from asag_engine.dev.grading_seed import bootstrap_grading_scenario, list_grading_targets


bp = Blueprint("dev", __name__)


@bp.post("/api/v1/dev/bootstrap-grading-scenario")
def bootstrap_grading():
    session = get_session()
    try:
        payload = request.get_json(silent=True) or {}
        seeded = bootstrap_grading_scenario(session, payload)
        session.commit()
        result = seeded.__dict__.copy()
        attempt_answer_id = result["attempt_answer_id"]
        assessment_attempt_id = result["assessment_attempt_id"]
        result["grade_question_url"] = f"/api/v1/grade/attempt-answer/{attempt_answer_id}"
        result["grade_assessment_url"] = f"/api/v1/grade/assessment-attempt/{assessment_attempt_id}"
        return jsonify({"status": "ok", "result": result}), 201
    except Exception as exc:
        session.rollback()
        return jsonify({"error": "Bootstrap failed", "details": str(exc)}), 500
    finally:
        session.close()


@bp.get("/api/v1/dev/grading-targets")
def grading_targets():
    session = get_session()
    try:
        limit_raw = request.args.get("limit", "20")
        try:
            limit = int(limit_raw)
        except ValueError:
            limit = 20
        rows = list_grading_targets(session, limit=limit)
        return jsonify({"status": "ok", "count": len(rows), "items": rows}), 200
    except Exception as exc:
        return jsonify({"error": "Failed to list grading targets", "details": str(exc)}), 500
    finally:
        session.close()
