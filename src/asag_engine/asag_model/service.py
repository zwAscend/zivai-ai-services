from __future__ import annotations

import os
from typing import Iterable

from .model import AsagConfigurationError, AsagInferenceError, AsagScore, build_asag_scorer
from .schema import AsagScoreResponse


ASAG_ENABLED = os.getenv("ASAG_ENABLED", "true").strip().lower() == "true"
ASAG_SHORT_ANSWER_MAX_MARKS = float(os.getenv("ASAG_SHORT_ANSWER_MAX_MARKS", "5.0"))


def score_short_answer(
    question: str,
    reference_answer: str,
    student_answer: str,
    max_score: float,
    expected_points: Iterable[str] | None = None,
) -> AsagScoreResponse:
    if not ASAG_ENABLED:
        raise AsagConfigurationError("ASAG scoring is disabled. Set ASAG_ENABLED=true to enable it.")
    scorer = build_asag_scorer()
    score = scorer.score(question=question, reference_answer=reference_answer, student_answer=student_answer, max_score=max_score)
    return _build_response(score, reference_answer=reference_answer, expected_points=list(expected_points or []))


def should_use_asag(question_type_code: str | None, expected_answer: str | None, rubric_items_count: int, max_score: float) -> bool:
    if not ASAG_ENABLED:
        return False
    question_type = (question_type_code or "").strip().lower()
    if any(token in question_type for token in ("mcq", "multiple_choice", "true_false", "boolean")):
        return False
    if any(token in question_type for token in ("essay", "code", "long_answer", "programming")):
        return False
    if rubric_items_count > 0:
        return False
    if not (expected_answer or "").strip():
        return False
    return max_score <= ASAG_SHORT_ANSWER_MAX_MARKS


def _build_response(score: AsagScore, reference_answer: str, expected_points: list[str]) -> AsagScoreResponse:
    missing_points = []
    if score.normalized_score < 0.999:
        missing_points = [point.strip() for point in expected_points if isinstance(point, str) and point.strip()][:3]
        if not missing_points:
            missing_points = ["Align your answer more closely with the expected answer."]

    feedback_text = _feedback_text(score, reference_answer, missing_points)
    strengths = _strengths(score)
    next_steps = _next_steps(score, missing_points)
    return AsagScoreResponse(
        raw_score_0_5=score.raw_score_0_5,
        normalized_score=score.normalized_score,
        scaled_score=score.scaled_score,
        band=score.band,
        confidence=score.confidence,
        backend=score.backend,
        artifact_path=score.artifact_path,
        max_score=score.max_score,
        feedback_text=feedback_text,
        feedback_summary=feedback_text,
        strengths=strengths,
        missing_points=missing_points,
        next_steps=next_steps,
    )


def _feedback_text(score: AsagScore, reference_answer: str, missing_points: list[str]) -> str:
    if score.band == "strong":
        return "The answer aligns closely with the expected answer and covers the key idea clearly."
    if score.band == "good":
        return "The answer is mostly correct but still misses some detail from the expected answer."
    if score.band == "partial":
        return "The answer shows partial understanding but does not cover the full expected answer."
    if score.band == "weak":
        return "The answer has limited overlap with the expected answer and needs more of the required idea."
    return "The answer is far from the expected answer and needs a more direct, accurate response."


def _strengths(score: AsagScore) -> list[str]:
    if score.normalized_score >= 0.85:
        return ["Your answer closely matches the expected answer."]
    if score.normalized_score >= 0.5:
        return ["Your answer includes some relevant ideas from the expected answer."]
    return []


def _next_steps(score: AsagScore, missing_points: list[str]) -> list[str]:
    if missing_points:
        return [f"Review: {point}" for point in missing_points[:3]]
    if score.normalized_score < 1.0:
        return ["Compare your wording with the expected answer and tighten the key idea."]
    return ["Keep using the same structure on similar short-answer questions."]
