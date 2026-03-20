from __future__ import annotations

import json
from typing import Any

from .schema import TeacherPerformanceInsightRequest, TeacherPerformanceInsightResponse

_MAX_HISTORY = 8
_MAX_CELLS = 18


def _collapse_whitespace(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _trim_text(text: str, limit: int) -> str:
    compact = _collapse_whitespace(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _preview(text: str | None, limit: int = 240) -> str:
    return _collapse_whitespace(text)[:limit]


def _parse_json_candidate(raw_text: str) -> Any:
    decoder = json.JSONDecoder()
    best_candidate: Any | None = None
    best_span = -1
    for index, char in enumerate(raw_text):
        if char not in "[{":
            continue
        try:
            candidate, end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if end > best_span:
            best_candidate = candidate
            best_span = end
    if best_candidate is None:
        raise ValueError("No JSON found in model response")
    return best_candidate


def _build_prompt(request: TeacherPerformanceInsightRequest) -> tuple[str, str]:
    subject_name = _collapse_whitespace(request.subjectName or "Classroom performance")
    system_text = (
        f"You are an AI classroom performance analyst helping a teacher understand learner trends in {subject_name}.\n"
        "Return JSON ONLY. No markdown fences. No prose outside JSON.\n\n"
        "Rules:\n"
        "- Use a concise teacher-facing tone.\n"
        "- Ground every observation in the provided classroom data only.\n"
        "- Default to ZIMSEC O Level high-school expectations unless the input explicitly requests another level.\n"
        "- Do not introduce tertiary or university-level assumptions.\n"
        "- Highlight actionable misconceptions, learners to check first, and one practical next move.\n"
        "- Do not claim exact misconceptions unless the weak areas support that claim.\n"
        "- If the data is thin, say so clearly and give the safest next step.\n\n"
        "Return exactly this schema:\n"
        "{\n"
        '  "reply": "...",\n'
        '  "suggestedNextAction": "...",\n'
        '  "focusStudents": ["student-id"],\n'
        '  "focusTopics": ["topic-name"]\n'
        "}"
    )
    user_payload = {
        "teacherId": request.teacherId,
        "subjectName": subject_name,
        "currentView": request.currentView,
        "filterLabel": _collapse_whitespace(request.filterLabel or ""),
        "latestMessage": _trim_text(_collapse_whitespace(request.latestMessage), 600),
        "summary": request.summary.model_dump() if request.summary else {},
        "misconceptions": [item.model_dump() for item in request.misconceptions[:4]],
        "heatmapCells": [item.model_dump() for item in request.heatmapCells[:_MAX_CELLS]],
        "messages": [
            {"role": message.role, "text": _trim_text(_collapse_whitespace(message.text), 240)}
            for message in request.messages[-_MAX_HISTORY:]
            if _collapse_whitespace(message.text)
        ],
    }
    return system_text, json.dumps(user_payload, ensure_ascii=False)


def _fallback_response(request: TeacherPerformanceInsightRequest) -> TeacherPerformanceInsightResponse:
    summary = request.summary
    misconceptions = request.misconceptions[:3]
    cells = sorted(
        request.heatmapCells,
        key=lambda cell: (
            101.0 if cell.score is None else float(cell.score),
            _collapse_whitespace(cell.lastName).lower(),
            _collapse_whitespace(cell.firstName).lower(),
        ),
    )
    focus_students = [cell.studentId for cell in cells[:3]]
    focus_topics = [item.title for item in misconceptions if _collapse_whitespace(item.title)]

    if summary and summary.supportCount > 0:
        reply = (
            f"The most urgent issue is that {summary.supportCount} learners are below the secure range in "
            f"{_collapse_whitespace(summary.filterLabel or request.filterLabel or request.subjectName or 'this view')}. "
            "Start by checking the lowest-scoring learners, then reteach the weakest concept with one short model and one immediate check-for-understanding."
        )
    elif misconceptions:
        reply = (
            f"The clearest classroom friction point is {misconceptions[0].title}. "
            "Revisit the key idea, show one worked example, and then ask learners to justify the reasoning step that usually breaks down."
        )
    else:
        reply = (
            "There is not enough scored evidence yet to make a strong diagnosis. Use the heatmap to find missing evidence first, then ask one focused diagnostic question before planning reteach."
        )

    return TeacherPerformanceInsightResponse(
        reply=reply,
        suggestedNextAction="Open the lowest-intensity learners first and compare them with the weakest area shown in the heatmap.",
        focusStudents=focus_students,
        focusTopics=focus_topics,
    )


def generate_teacher_performance_insight(request: TeacherPerformanceInsightRequest, *, llm_client) -> TeacherPerformanceInsightResponse:
    system_text, user_text = _build_prompt(request)
    try:
        raw = llm_client.generate(system_text, user_text, max_new_tokens=320)
        candidate = _parse_json_candidate(raw)
        if not isinstance(candidate, dict):
            raise ValueError("Teacher performance response is not an object")
        reply = _collapse_whitespace(str(candidate.get("reply") or ""))
        if not reply:
            raise ValueError("Teacher performance reply is empty")
        suggested_next_action = _collapse_whitespace(str(candidate.get("suggestedNextAction") or "")) or None
        focus_students = [
            _collapse_whitespace(str(value))
            for value in candidate.get("focusStudents") or []
            if _collapse_whitespace(str(value))
        ]
        focus_topics = [
            _collapse_whitespace(str(value))
            for value in candidate.get("focusTopics") or []
            if _collapse_whitespace(str(value))
        ]
        return TeacherPerformanceInsightResponse(
            reply=reply,
            suggestedNextAction=suggested_next_action,
            focusStudents=focus_students,
            focusTopics=focus_topics,
        )
    except Exception as exc:
        print(f"[teacher-performance] using fallback: {exc}; preview={_preview(locals().get('raw'))}")
        return _fallback_response(request)
