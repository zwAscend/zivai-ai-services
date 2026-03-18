from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from asag_engine.agents.schema import (
    StudentAssessmentCriterion,
    StudentAssessmentData,
    StudentAssessmentQuestionDetail,
    StudentAssessmentResponse,
)
from asag_engine.ocr.schema import OcrGeneralRequest, OcrGeneralResponse
from asag_engine.ocr.service import extract_general_ocr


STUDENT_ASSESSMENT_TOTAL_MARKS = float(os.getenv("STUDENT_ASSESSMENT_TOTAL_MARKS", "10"))
STUDENT_ASSESSMENT_MAX_NEW_TOKENS = int(os.getenv("STUDENT_ASSESSMENT_MAX_NEW_TOKENS", "320"))
STUDENT_ASSESSMENT_MAX_TEXT_CHARS = int(os.getenv("STUDENT_ASSESSMENT_MAX_TEXT_CHARS", "5000"))
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "this", "that", "these", "those", "about",
}


class StudentAssessmentError(RuntimeError):
    def __init__(self, message: str, first_raw: str | None = None, retry_raw: str | None = None):
        super().__init__(message)
        self.first_raw = first_raw
        self.retry_raw = retry_raw


class _StudentAssessmentModel(BaseModel):
    is_correct_module: bool
    confidence_assessment_score: float = Field(..., ge=0, le=1)
    total_possible_marks: float = Field(..., ge=0)
    marks_achieved: float = Field(..., ge=0)
    overall_feedback: str = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    criteria: list[StudentAssessmentCriterion] = Field(default_factory=list)
    detected_module: str | None = None
    mark_consistency_check: str = "consistent"
    marking_scheme_used: bool = False


_EXPECTED_KEYS = {
    "is_correct_module",
    "confidence_assessment_score",
    "total_possible_marks",
    "marks_achieved",
    "overall_feedback",
    "strengths",
    "improvements",
    "criteria",
    "detected_module",
    "mark_consistency_check",
    "marking_scheme_used",
}


def assess_student_text(
    *,
    module: str,
    text: str,
    llm_client,
    filename: str | None = None,
    content_type: str | None = None,
    ocr_type: str | None = None,
    pages: int | None = None,
) -> StudentAssessmentResponse:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return _empty_text_response(module, filename, content_type, ocr_type, pages)

    truncated_text = normalized_text[:STUDENT_ASSESSMENT_MAX_TEXT_CHARS]
    system_text, user_text = _build_student_assessment_prompt(module, truncated_text)
    raw = llm_client.generate(system_text, user_text, max_new_tokens=STUDENT_ASSESSMENT_MAX_NEW_TOKENS)

    try:
        parsed = _parse_model_response(raw)
    except Exception as first_exc:
        print(f"[student-assessment] first pass invalid output: {first_exc}; preview={_preview(raw)}")
        retry_system = system_text + "\n\nReturn ONLY valid JSON matching the schema. No extra words."
        raw_retry = llm_client.generate(retry_system, user_text, max_new_tokens=STUDENT_ASSESSMENT_MAX_NEW_TOKENS)
        try:
            parsed = _parse_model_response(raw_retry)
            raw = raw_retry
        except Exception as retry_exc:
            print(f"[student-assessment] retry invalid output: {retry_exc}; preview={_preview(raw_retry)}")
            parsed = _heuristic_assessment(module, truncated_text)
            raw = None

    return _build_student_response(
        module=module,
        markdown=normalized_text,
        filename=filename,
        content_type=content_type,
        ocr_type=ocr_type,
        pages=pages,
        parsed=parsed,
    )


def assess_student_file(*, module: str, file_storage, llm_client) -> StudentAssessmentResponse:
    ocr_result = extract_general_ocr(
        [file_storage],
        OcrGeneralRequest(
            source="student_assessment",
            module=module,
            preferDigitalExtraction=True,
            forceOcr=False,
            returnMarkdownResult=True,
        ),
    )
    markdown = ocr_result.combinedText.strip()
    first_document = ocr_result.documents[0] if ocr_result.documents else None
    return assess_student_text(
        module=module,
        text=markdown,
        llm_client=llm_client,
        filename=getattr(file_storage, "filename", None),
        content_type=getattr(file_storage, "mimetype", None),
        ocr_type=(first_document.mode if first_document else None),
        pages=len(ocr_result.documents) or None,
    )


def _build_student_assessment_prompt(module: str, text: str) -> tuple[str, str]:
    system_text = """You are an AI assessment assistant for Computer Science student practice.
Assess whether the student's response is relevant to the declared module and estimate the quality of understanding.
Return JSON ONLY. No markdown. No prose outside JSON.

Rules:
- Use the declared module as the reference topic.
- If the answer is unrelated to the module, set is_correct_module to false and award very low marks.
- total_possible_marks must be 10.
- marks_achieved must be between 0 and 10.
- strengths should list concise positives.
- improvements should list concise next steps.
- criteria should cover module relevance, concept understanding, and clarity/completeness.
- marking_scheme_used must be false.

Return exactly this schema:
{
  "is_correct_module": true,
  "confidence_assessment_score": 0.0,
  "total_possible_marks": 10,
  "marks_achieved": 0,
  "overall_feedback": "...",
  "strengths": ["..."],
  "improvements": ["..."],
  "criteria": [
    {"criterion": "Module relevance", "score": 0, "feedback": "..."},
    {"criterion": "Concept understanding", "score": 0, "feedback": "..."},
    {"criterion": "Clarity and completeness", "score": 0, "feedback": "..."}
  ],
  "detected_module": "...",
  "mark_consistency_check": "consistent",
  "marking_scheme_used": false
}
"""
    user_payload = {
        "module": module,
        "student_submission": text,
        "total_possible_marks": int(STUDENT_ASSESSMENT_TOTAL_MARKS),
    }
    return system_text, json.dumps(user_payload, ensure_ascii=False)


def _parse_model_response(raw_text: str) -> _StudentAssessmentModel:
    obj = json.loads(_extract_json_object(raw_text))
    model = _StudentAssessmentModel(**obj)
    total = max(1.0, float(STUDENT_ASSESSMENT_TOTAL_MARKS))
    model.total_possible_marks = total
    model.marks_achieved = max(0.0, min(float(model.marks_achieved), total))
    model.confidence_assessment_score = max(0.0, min(float(model.confidence_assessment_score), 1.0))
    cleaned_criteria: list[StudentAssessmentCriterion] = []
    for criterion in model.criteria:
        cleaned_criteria.append(
            StudentAssessmentCriterion(
                criterion=criterion.criterion.strip(),
                score=max(0.0, min(float(criterion.score), total)),
                feedback=criterion.feedback.strip() or "Assessment generated.",
            )
        )
    model.criteria = cleaned_criteria[:3]
    return model


def _build_student_response(
    *,
    module: str,
    markdown: str,
    filename: str | None,
    content_type: str | None,
    ocr_type: str | None,
    pages: int | None,
    parsed: _StudentAssessmentModel,
) -> StudentAssessmentResponse:
    total = max(1.0, float(parsed.total_possible_marks or STUDENT_ASSESSMENT_TOTAL_MARKS))
    achieved = max(0.0, min(float(parsed.marks_achieved), total))
    percentage = round((achieved / total) * 100, 2) if total > 0 else 0.0
    strengths = [item.strip() for item in parsed.strengths if isinstance(item, str) and item.strip()][:3]
    improvements = [item.strip() for item in parsed.improvements if isinstance(item, str) and item.strip()][:3]
    if not improvements and achieved < total:
        improvements = ["Add more module-specific detail and explain the key idea more clearly."]

    detail = StudentAssessmentQuestionDetail(
        max_marks=round(total, 2),
        awarded_marks=round(achieved, 2),
        feedback=(parsed.overall_feedback or "Assessment generated.").strip(),
        improvement=improvements[0] if improvements else "Keep practising with module-focused responses.",
    )
    assessment = StudentAssessmentData(
        is_correct_module=bool(parsed.is_correct_module),
        confidence_assessment_score=round(float(parsed.confidence_assessment_score), 4),
        total_possible_marks=round(total, 2),
        marks_achieved=round(achieved, 2),
        marks_percentage=percentage,
        overall_feedback=(parsed.overall_feedback or "Assessment generated.").strip(),
        strengths=strengths,
        improvements=improvements,
        criteria=parsed.criteria or _default_criteria(module, achieved, total),
        assessment_details={"response": detail},
        detected_module=(parsed.detected_module or module).strip() if parsed.is_correct_module else parsed.detected_module,
        mark_consistency_check=(parsed.mark_consistency_check or "consistent").strip() or "consistent",
        marking_scheme_used=bool(parsed.marking_scheme_used),
    )
    return StudentAssessmentResponse(
        module=module,
        filename=filename,
        content_type=content_type,
        ocr_type=ocr_type,
        markdown=markdown,
        pages=pages,
        assessment=assessment,
    )


def _heuristic_assessment(module: str, text: str) -> _StudentAssessmentModel:
    total = max(1.0, float(STUDENT_ASSESSMENT_TOTAL_MARKS))
    module_tokens = _meaningful_tokens(module)
    text_tokens = _meaningful_tokens(text)
    token_counts = Counter(text_tokens)
    overlap = sum(1 for token in module_tokens if token in token_counts)
    relevance = overlap / max(1, len(module_tokens)) if module_tokens else 0.5
    word_count = max(1, len(text.split()))
    completeness = min(1.0, word_count / 70.0)
    clarity = 0.7 if any(punct in text for punct in ".;:\n") else 0.45
    weighted = (0.45 * relevance) + (0.35 * completeness) + (0.20 * clarity)
    achieved = round(min(total, max(0.0, weighted * total)), 2)
    is_correct_module = relevance >= 0.34 or overlap >= 2

    strengths: list[str] = []
    if relevance >= 0.5:
        strengths.append("Your response stays on the declared module.")
    if completeness >= 0.55:
        strengths.append("You provided enough detail to show partial understanding.")

    improvements: list[str] = []
    if relevance < 0.5:
        improvements.append(f"Focus more directly on the module '{module}'.")
    if completeness < 0.6:
        improvements.append("Add more concrete explanation and supporting detail.")
    if clarity < 0.6:
        improvements.append("Structure the response in clearer, complete sentences.")
    if not improvements and achieved < total:
        improvements.append("Add one or two more precise module-specific ideas to strengthen the answer.")

    feedback = (
        f"This response shows {'some' if achieved > 0 else 'limited'} understanding of {module}. "
        f"Keep the explanation closely tied to the core ideas of the module."
    )
    criteria = _default_criteria(module, achieved, total, relevance=relevance, completeness=completeness, clarity=clarity)
    return _StudentAssessmentModel(
        is_correct_module=is_correct_module,
        confidence_assessment_score=0.35,
        total_possible_marks=total,
        marks_achieved=achieved,
        overall_feedback=feedback,
        strengths=strengths,
        improvements=improvements,
        criteria=criteria,
        detected_module=module if is_correct_module else None,
        mark_consistency_check="heuristic-fallback",
        marking_scheme_used=False,
    )


def _default_criteria(
    module: str,
    achieved: float,
    total: float,
    *,
    relevance: float | None = None,
    completeness: float | None = None,
    clarity: float | None = None,
) -> list[StudentAssessmentCriterion]:
    relevance_score = round((relevance if relevance is not None else achieved / max(total, 1.0)) * total, 2)
    completeness_score = round((completeness if completeness is not None else achieved / max(total, 1.0)) * total, 2)
    clarity_score = round((clarity if clarity is not None else min(1.0, achieved / max(total, 1.0) + 0.1)) * total, 2)
    return [
        StudentAssessmentCriterion(
            criterion="Module relevance",
            score=max(0.0, min(relevance_score, total)),
            feedback=f"The response is assessed against the declared module '{module}'.",
        ),
        StudentAssessmentCriterion(
            criterion="Concept understanding",
            score=max(0.0, min(completeness_score, total)),
            feedback="This reflects how well the answer demonstrates the main idea.",
        ),
        StudentAssessmentCriterion(
            criterion="Clarity and completeness",
            score=max(0.0, min(clarity_score, total)),
            feedback="This reflects how clearly and fully the response is expressed.",
        ),
    ]


def _empty_text_response(
    module: str,
    filename: str | None,
    content_type: str | None,
    ocr_type: str | None,
    pages: int | None,
) -> StudentAssessmentResponse:
    total = max(1.0, float(STUDENT_ASSESSMENT_TOTAL_MARKS))
    assessment = StudentAssessmentData(
        is_correct_module=False,
        confidence_assessment_score=1.0,
        total_possible_marks=total,
        marks_achieved=0.0,
        marks_percentage=0.0,
        overall_feedback="No readable response was detected. Submit text or a clearer image to receive AI feedback.",
        strengths=[],
        improvements=["Provide a readable response that directly addresses the module."],
        criteria=_default_criteria(module, 0.0, total, relevance=0.0, completeness=0.0, clarity=0.0),
        assessment_details={
            "response": StudentAssessmentQuestionDetail(
                max_marks=total,
                awarded_marks=0.0,
                feedback="No readable response was detected.",
                improvement="Submit text or upload a clearer image.",
            )
        },
        detected_module=None,
        mark_consistency_check="consistent",
        marking_scheme_used=False,
    )
    return StudentAssessmentResponse(
        module=module,
        filename=filename,
        content_type=content_type,
        ocr_type=ocr_type,
        markdown="",
        pages=pages,
        assessment=assessment,
    )


def _extract_json_object(raw_text: str) -> str:
    decoder = json.JSONDecoder()
    candidates: list[tuple[dict[str, Any], str]] = []
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append((obj, raw_text[index:index + end]))
    if not candidates:
        raise ValueError("Model output did not contain a JSON object")
    for obj, raw_obj in reversed(candidates):
        if _EXPECTED_KEYS.issubset(obj.keys()):
            return raw_obj
    return candidates[-1][1]


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _preview(raw_text: str | None, limit: int = 240) -> str:
    if not raw_text:
        return ""
    return " ".join(raw_text.split())[:limit]
