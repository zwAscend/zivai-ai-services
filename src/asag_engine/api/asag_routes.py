from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from asag_engine.asag_model import AsagConfigurationError, AsagInferenceError, score_short_answer
from asag_engine.asag_model.schema import AsagScoreRequest


bp = Blueprint("asag", __name__)


@bp.post("/api/v1/asag/score")
def asag_score_route():
    try:
        payload = AsagScoreRequest.model_validate(request.get_json(silent=True) or {})
        result = score_short_answer(
            question=payload.question,
            reference_answer=payload.reference_answer,
            student_answer=payload.student_answer,
            max_score=float(payload.max_score),
            expected_points=payload.expected_points,
        )
        return jsonify({"status": "ok", "result": result.model_dump()}), 200
    except ValidationError as exc:
        return jsonify({"error": "Invalid request body", "details": exc.json()}), 400
    except AsagConfigurationError as exc:
        return jsonify({"error": "ASAG configuration error", "details": str(exc)}), 503
    except AsagInferenceError as exc:
        return jsonify({"error": "ASAG inference error", "details": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": "Internal ASAG error", "details": str(exc)}), 500
