from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from asag_engine.agents.schema import AssessmentGenerationRequest, PlanGenerationRequest
from asag_engine.agents.service import (
    AssessmentGenerationError,
    generate_teacher_assessment_questions,
    generate_teacher_development_plan,
)
from asag_engine.agents.student_assessment import assess_student_file, assess_student_text
from asag_engine.grading.llm_client import build_llm_client
from asag_engine.ocr.service import OcrConfigurationError, OcrProcessingError, OcrValidationError


bp = Blueprint("agents", __name__)
_llm = build_llm_client()


def _error_response(error: str, details: str, status: int, extra: dict | None = None):
    body = {"error": error, "details": details}
    if extra:
        body.update(extra)
    return jsonify(body), status


@bp.post("/api/v1/agents/teacher/assessment-generation")
def teacher_assessment_generation():
    try:
        payload = AssessmentGenerationRequest.model_validate(request.get_json(silent=True) or {})
        questions = generate_teacher_assessment_questions(payload, llm_client=_llm)
        return jsonify([question.model_dump() for question in questions]), 200
    except ValidationError as exc:
        return _error_response("Invalid request body", exc.json(), 400)
    except AssessmentGenerationError as exc:
        return _error_response(
            "Model returned invalid assessment generation output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except Exception as exc:
        return _error_response("Internal assessment generation error", str(exc), 500)


@bp.post("/api/v1/agents/teacher/plan-generation")
def teacher_plan_generation():
    try:
        payload = PlanGenerationRequest.model_validate(request.get_json(silent=True) or {})
        plan = generate_teacher_development_plan(payload, llm_client=_llm)
        return jsonify(plan.model_dump()), 200
    except ValidationError as exc:
        return _error_response("Invalid request body", exc.json(), 400)
    except AssessmentGenerationError as exc:
        return _error_response(
            "Model returned invalid development plan output",
            str(exc),
            422,
            extra={
                "first_pass_preview": " ".join((exc.first_raw or "").split())[:320],
                "retry_preview": " ".join((exc.retry_raw or "").split())[:320],
            },
        )
    except Exception as exc:
        return _error_response("Internal development plan generation error", str(exc), 500)


@bp.post("/api/v1/agents/student/assessment")
def student_assessment():
    try:
        module = (request.form.get("module") or request.args.get("module") or "").strip()
        if not module:
            return _error_response("Invalid request body", "module is required", 400)

        uploaded = request.files.get("file")
        text = (request.form.get("text") or "").strip()

        if uploaded and uploaded.filename:
            result = assess_student_file(module=module, file_storage=uploaded, llm_client=_llm)
            return jsonify(result.model_dump()), 200

        if text:
            result = assess_student_text(module=module, text=text, llm_client=_llm, ocr_type="text")
            return jsonify(result.model_dump()), 200

        return _error_response("Invalid request body", "Provide either text or file.", 400)
    except OcrValidationError as exc:
        return _error_response("Invalid OCR request", str(exc), 400)
    except OcrConfigurationError as exc:
        return _error_response("OCR configuration error", str(exc), 503)
    except OcrProcessingError as exc:
        return _error_response("OCR processing error", str(exc), 502)
    except Exception as exc:
        return _error_response("Internal student assessment error", str(exc), 500)
