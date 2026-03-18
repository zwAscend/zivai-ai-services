import time

from flask import Blueprint, jsonify, request

from asag_engine.grading.llm_client import build_llm_client

bp = Blueprint("llm", __name__)
_llm = build_llm_client()

def _default_system_text() -> str:
    return "You are a helpful assistant. Return only concise plain text."

def _default_user_text() -> str:
    return "Reply with exactly: LLM_OK"

@bp.post("/api/v1/llm/test")
def llm_test():
    payload = request.get_json(silent=True) or {}
    system_text = (payload.get("system_text") or _default_system_text()).strip()
    user_text = (payload.get("user_text") or _default_user_text()).strip()

    if not system_text or not user_text:
        return jsonify({"error": "system_text and user_text must not be empty"}), 400

    started = time.perf_counter()
    output = _llm.generate(system_text, user_text)
    elapsed = time.perf_counter() - started

    return jsonify({
        "status": "ok",
        "system_text": system_text,
        "user_text": user_text,
        "output": output,
        "output_length": len(output),
        "seconds": round(elapsed, 2),
    }), 200
