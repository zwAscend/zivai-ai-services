from uuid import UUID

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from asag_engine.db.session import get_session
from asag_engine.dkt import (
    DktConfigurationError,
    DktNotFoundError,
    DktUpdateRequest,
    DktValidationError,
    get_student_mastery,
    update_student_mastery,
)


bp = Blueprint("dkt", __name__)


def _error_response(error: str, details: str, status: int):
    return jsonify({"error": error, "details": details}), status


@bp.post("/api/v1/dkt/update")
def dkt_update():
    session = get_session()
    try:
        payload = DktUpdateRequest.model_validate(request.get_json(silent=True) or {})
        result = update_student_mastery(session, payload)
        if payload.persist:
            session.commit()
        return jsonify({"status": "ok", "result": result.model_dump(mode="json")}), 200
    except ValidationError as exc:
        session.rollback()
        return _error_response("Invalid request body", exc.json(), 400)
    except DktValidationError as exc:
        session.rollback()
        return _error_response("Invalid DKT request", str(exc), 400)
    except DktNotFoundError as exc:
        session.rollback()
        return _error_response("Not found", str(exc), 404)
    except DktConfigurationError as exc:
        session.rollback()
        return _error_response("DKT configuration error", str(exc), 503)
    except Exception as exc:
        session.rollback()
        return _error_response("Internal DKT update error", str(exc), 500)
    finally:
        session.close()


@bp.get("/api/v1/dkt/mastery/<student_id>")
def dkt_mastery(student_id: str):
    session = get_session()
    try:
        subject_code = request.args.get("subject_code") or None
        subject_id_raw = request.args.get("subject_id") or None
        subject_id = UUID(subject_id_raw) if subject_id_raw else None
        include_mastery_vector = (request.args.get("full") or "false").strip().lower() in {"1", "true", "yes", "on"}
        refresh = (request.args.get("refresh") or "false").strip().lower() in {"1", "true", "yes", "on"}
        result = get_student_mastery(
            session,
            student_id=student_id,
            subject_id=subject_id,
            subject_code=subject_code,
            include_mastery_vector=include_mastery_vector,
            refresh=refresh,
        )
        return jsonify({"status": "ok", "result": result.model_dump(mode="json")}), 200
    except ValueError as exc:
        return _error_response("Invalid request", str(exc), 400)
    except DktValidationError as exc:
        return _error_response("Invalid DKT request", str(exc), 400)
    except DktNotFoundError as exc:
        return _error_response("Not found", str(exc), 404)
    except DktConfigurationError as exc:
        return _error_response("DKT configuration error", str(exc), 503)
    except Exception as exc:
        return _error_response("Internal DKT mastery error", str(exc), 500)
    finally:
        session.close()
