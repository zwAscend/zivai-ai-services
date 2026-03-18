from uuid import UUID

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from asag_engine.db.session import get_session
from asag_engine.grading.grader import GradeGenerationError
from asag_engine.grading.llm_client import build_llm_client
from asag_engine.grading.schema import AssessmentGradePayload, GradeRequestOptions, QuestionGradePayload
from asag_engine.grading.service import (
    GradingNotFoundError,
    grade_assessment_attempt,
    grade_attempt_answer,
    grade_payload_assessment,
    grade_payload_question,
)


bp = Blueprint("grading", __name__)
_llm = build_llm_client()


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _parse_options() -> GradeRequestOptions:
    payload = request.get_json(silent=True) or {}
    options_payload = payload.get("options")
    if isinstance(options_payload, dict):
        payload = options_payload
    return GradeRequestOptions(
        dry_run=_coerce_bool(payload.get("dry_run", False)),
        force=_coerce_bool(payload.get("force", False)),
        allow_holistic_fallback=_coerce_bool(payload.get("allow_holistic_fallback", True), default=True),
    )



def _error_response(error: str, details: str, status: int, extra: dict | None = None):
    body = {"error": error, "details": details}
    if extra:
        body.update(extra)
    return jsonify(body), status


@bp.post("/api/v1/grade/question")
def grade_question_payload_route():
    try:
        payload = QuestionGradePayload.model_validate(request.get_json(silent=True) or {})
        result = grade_payload_question(payload, llm_client=_llm)
        return jsonify({"status": "ok", "write_back": False, "result": result.model_dump()}), 200
    except ValidationError as exc:
        return _error_response("Invalid request body", exc.json(), 400)
    except GradeGenerationError as exc:
        return _error_response(
            "Model returned invalid grading output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except ValueError as exc:
        return _error_response("Invalid grading request", str(exc), 400)
    except Exception as exc:
        return _error_response("Internal grading error", str(exc), 500)


@bp.post("/api/v1/grade/assessment")
def grade_assessment_payload_route():
    try:
        payload = AssessmentGradePayload.model_validate(request.get_json(silent=True) or {})
        result = grade_payload_assessment(payload, llm_client=_llm)
        return jsonify({"status": "ok", "write_back": False, "result": result.model_dump()}), 200
    except ValidationError as exc:
        return _error_response("Invalid request body", exc.json(), 400)
    except GradeGenerationError as exc:
        return _error_response(
            "Model returned invalid grading output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except ValueError as exc:
        return _error_response("Invalid grading request", str(exc), 400)
    except Exception as exc:
        return _error_response("Internal grading error", str(exc), 500)


@bp.post("/api/v1/grade/attempt-answer/<uuid:attempt_answer_id>")
def grade_one_question(attempt_answer_id: UUID):
    session = get_session()
    try:
        options = _parse_options()
        result = grade_attempt_answer(session, attempt_answer_id, options=options, llm_client=_llm)
        if not options.dry_run:
            session.commit()
        return jsonify({"status": "ok", "write_back": not options.dry_run, "result": result.model_dump()}), 200
    except GradingNotFoundError as exc:
        session.rollback()
        return _error_response("Not found", str(exc), 404)
    except GradeGenerationError as exc:
        session.rollback()
        return _error_response(
            "Model returned invalid grading output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except Exception as exc:
        session.rollback()
        return _error_response("Internal grading error", str(exc), 500)
    finally:
        session.close()


@bp.post("/api/v1/grade/assessment-attempt/<uuid:assessment_attempt_id>")
def grade_whole_assessment(assessment_attempt_id: UUID):
    session = get_session()
    try:
        options = _parse_options()
        result = grade_assessment_attempt(session, assessment_attempt_id, options=options, llm_client=_llm)
        if not options.dry_run:
            session.commit()
        return jsonify({"status": "ok", "write_back": not options.dry_run, "result": result.model_dump()}), 200
    except GradingNotFoundError as exc:
        session.rollback()
        return _error_response("Not found", str(exc), 404)
    except GradeGenerationError as exc:
        session.rollback()
        return _error_response(
            "Model returned invalid grading output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except Exception as exc:
        session.rollback()
        return _error_response("Internal grading error", str(exc), 500)
    finally:
        session.close()
