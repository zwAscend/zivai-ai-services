import json

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from asag_engine.ocr.schema import OcrGeneralRequest
from asag_engine.ocr.service import (
    OcrConfigurationError,
    OcrProcessingError,
    OcrValidationError,
    extract_general_ocr,
)


bp = Blueprint("ocr", __name__)


@bp.post("/api/v1/agents/ocr/general")
def ocr_general():
    try:
        payload, legacy_response = _parse_request_payload()
        files = _collect_files()
        result = extract_general_ocr(files, payload)
        if legacy_response:
            return jsonify([
                {
                    "documentName": item.referenceDocument.documentName,
                    "markdown": item.referenceDocument.markdown,
                }
                for item in result.documents
            ]), 200
        return jsonify(result.model_dump()), 200
    except ValidationError as exc:
        return _error_response("Invalid OCR request", exc.json(), 400)
    except OcrValidationError as exc:
        return _error_response("Invalid OCR request", str(exc), 400)
    except OcrConfigurationError as exc:
        return _error_response("OCR configuration error", str(exc), 503)
    except OcrProcessingError as exc:
        return _error_response("OCR processing error", str(exc), 502)
    except Exception as exc:
        return _error_response("Internal OCR error", str(exc), 500)


def _parse_request_payload() -> tuple[OcrGeneralRequest, bool]:
    if request.is_json:
        return OcrGeneralRequest.model_validate(request.get_json(silent=True) or {}), False

    form_payload = request.form.to_dict(flat=True)
    request_blob = form_payload.pop("request", None)
    response_mode = (
        request.args.get("response_mode")
        or form_payload.pop("response_mode", None)
        or form_payload.pop("responseMode", None)
    )
    if request_blob:
        try:
            base_payload = json.loads(request_blob)
        except json.JSONDecodeError as exc:
            raise OcrValidationError("Form field 'request' must contain valid JSON.") from exc
    else:
        base_payload = {}
    base_payload.update(form_payload)
    legacy_response = _wants_legacy_response(response_mode=response_mode, has_request_blob=bool(request_blob), form_payload=form_payload)
    return OcrGeneralRequest.model_validate(base_payload), legacy_response


def _collect_files() -> list:
    collected = []
    collected.extend([item for item in request.files.getlist("files") if item and item.filename])
    single = request.files.get("file")
    if single and single.filename:
        collected.append(single)
    return collected


def _error_response(error: str, details: str, status: int):
    return jsonify({"error": error, "details": details}), status


def _wants_legacy_response(*, response_mode: str | None, has_request_blob: bool, form_payload: dict) -> bool:
    if response_mode:
        normalized = response_mode.strip().lower()
        if normalized == "legacy":
            return True
        if normalized == "full":
            return False
    return not has_request_blob and not form_payload
