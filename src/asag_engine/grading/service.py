from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from asag_engine.asag_model import AsagConfigurationError, AsagInferenceError
from asag_engine.asag_model.service import score_short_answer, should_use_asag
from asag_engine.db.shared_models import (
    AiInferenceRun,
    AiModel,
    AiModelVersion,
    LmsAssessmentAssignment,
    LmsAssessmentAttempt,
    LmsAssessmentEnrollment,
    LmsAssessmentQuestion,
    LmsAssessmentResult,
    LmsAttemptAnswer,
    LmsMarkingScheme,
    LmsMarkingSchemeItem,
)

from .grader import grade_holistically, grade_with_rubric
from .llm_client import LLMClient, build_llm_client
from .schema import (
    AssessmentGradePayload,
    AssessmentGradeResult,
    GradeRequestOptions,
    MarkingGuidePayload,
    QuestionGradePayload,
    QuestionGradeResult,
    QuestionRubricOutcome,
)


REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("AI_REVIEW_CONFIDENCE_THRESHOLD", "0.6"))
MAX_STUDENT_ANSWER_CHARS = int(os.getenv("GRADING_MAX_ANSWER_CHARS", "4000"))


class GradingNotFoundError(RuntimeError):
    pass


@dataclass
class RubricDescriptor:
    rubric_index: int
    description: str
    max_marks: float
    rubric_item_id: str | None = None
    rubric_code: str | None = None


@dataclass
class GradingContext:
    answer: LmsAttemptAnswer
    student_answer: str | None
    max_score: float
    school_id: UUID | None
    student_id: UUID | None

    @property
    def attempt(self) -> LmsAssessmentAttempt:
        return self.answer.assessment_attempt

    @property
    def assessment_question(self):
        return self.answer.assessment_question

    @property
    def question(self):
        return self.answer.assessment_question.question

    @property
    def assignment(self):
        return self.answer.assessment_attempt.assessment_enrollment.assessment_assignment

    @property
    def assessment(self):
        return self.assignment.assessment


@dataclass
class DirectGradingContext:
    question_text: str
    max_score: float
    student_answer: str | None
    question_type_code: str | None = None
    expected_answer: str | None = None
    expected_points: list[str] = field(default_factory=list)
    rubric_items: list[RubricDescriptor] = field(default_factory=list)
    attempt_answer_id: str | None = None
    assessment_attempt_id: str | None = None
    assessment_question_id: str | None = None
    question_id: str | None = None
    school_id: str | None = None
    student_id: str | None = None


def grade_attempt_answer(
    session: Session,
    attempt_answer_id: UUID,
    options: GradeRequestOptions | None = None,
    llm_client: LLMClient | None = None,
) -> QuestionGradeResult:
    options = options or GradeRequestOptions()
    llm_client = llm_client or build_llm_client()
    context = _load_attempt_answer_context(session, attempt_answer_id)
    result = _grade_question_context(session, context, options, llm_client)
    if not options.dry_run:
        _refresh_attempt_rollup(session, context.attempt.id)
    return result



def grade_assessment_attempt(
    session: Session,
    assessment_attempt_id: UUID,
    options: GradeRequestOptions | None = None,
    llm_client: LLMClient | None = None,
) -> AssessmentGradeResult:
    options = options or GradeRequestOptions()
    llm_client = llm_client or build_llm_client()
    attempt = _load_assessment_attempt(session, assessment_attempt_id)
    answers = sorted(attempt.answers, key=lambda item: item.assessment_question.sequence_index)
    if not answers:
        raise GradingNotFoundError(f"Assessment attempt {assessment_attempt_id} has no answers")

    results: list[QuestionGradeResult] = []
    for answer in answers:
        context = _build_context(answer)
        results.append(_grade_question_context(session, context, options, llm_client))

    if not options.dry_run:
        _refresh_attempt_rollup(session, attempt.id)
        session.refresh(attempt)

    total_score = sum(_effective_score(answer) for answer in attempt.answers) if not options.dry_run else sum(r.score_awarded for r in results)
    max_score = _attempt_max_score(attempt) if not options.dry_run else sum(r.max_score for r in results)
    ai_confidence = _average_confidence(results)
    requires_review = any(r.requires_review for r in results)
    trace_id = next((r.trace_id for r in results if r.trace_id), None)
    grading_status_code = attempt.grading_status_code if not options.dry_run else _derive_attempt_status(results)

    return AssessmentGradeResult(
        assessment_attempt_id=str(attempt.id),
        grading_status_code=grading_status_code,
        total_score=round(total_score, 2),
        max_score=round(max_score, 2),
        ai_confidence=ai_confidence,
        requires_review=requires_review,
        trace_id=trace_id,
        question_results=results,
    )


def grade_payload_question(
    payload: QuestionGradePayload,
    llm_client: LLMClient | None = None,
) -> QuestionGradeResult:
    llm_client = llm_client or build_llm_client()
    context = _build_direct_context(payload)
    return _grade_direct_context(context, payload.options, llm_client)


def grade_payload_assessment(
    payload: AssessmentGradePayload,
    llm_client: LLMClient | None = None,
) -> AssessmentGradeResult:
    llm_client = llm_client or build_llm_client()
    results = [
        _grade_direct_context(
            _build_direct_context(
                QuestionGradePayload(
                    request_context=_merge_request_context(payload.request_context.model_dump(), question_payload.request_context.model_dump()),
                    question=question_payload.question,
                    student_answer=question_payload.student_answer,
                    marking_guide=question_payload.marking_guide,
                    options=payload.options,
                )
            ),
            payload.options,
            llm_client,
        )
        for question_payload in payload.questions
    ]
    return AssessmentGradeResult(
        assessment_attempt_id=payload.request_context.assessment_attempt_id,
        grading_status_code=_derive_attempt_status(results),
        total_score=round(sum(result.score_awarded for result in results), 2),
        max_score=round(sum(result.max_score for result in results), 2),
        ai_confidence=_average_confidence(results),
        requires_review=any(result.requires_review for result in results),
        trace_id=next((result.trace_id for result in results if result.trace_id), None),
        question_results=results,
    )


def _merge_request_context(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value not in (None, ""):
            merged[key] = value
    return merged


def _build_direct_context(payload: QuestionGradePayload) -> DirectGradingContext:
    guide = payload.marking_guide or MarkingGuidePayload()
    expected_points = [point.strip() for point in guide.expected_points if isinstance(point, str) and point.strip()]
    if not expected_points:
        expected_points = [item.description for item in guide.rubric_items]
    return DirectGradingContext(
        question_text=payload.question.text.strip(),
        max_score=float(payload.question.max_marks),
        student_answer=(payload.student_answer.text or "").strip() or None,
        question_type_code=(payload.question.question_type or "").strip() or None,
        expected_answer=(guide.expected_answer or "").strip() or None,
        expected_points=expected_points,
        rubric_items=[
            RubricDescriptor(
                rubric_index=item.index,
                description=item.description.strip(),
                max_marks=float(item.marks),
                rubric_item_id=item.rubric_item_id,
                rubric_code=item.rubric_code,
            )
            for item in guide.rubric_items
        ],
        attempt_answer_id=payload.request_context.attempt_answer_id,
        assessment_attempt_id=payload.request_context.assessment_attempt_id,
        assessment_question_id=payload.request_context.assessment_question_id,
        question_id=payload.request_context.question_id,
        school_id=payload.request_context.school_id,
        student_id=payload.request_context.student_id,
    )


def _grade_direct_context(
    context: DirectGradingContext,
    options: GradeRequestOptions,
    llm_client: LLMClient,
) -> QuestionGradeResult:
    if not context.student_answer:
        return _no_answer_result_direct(context)

    objective_score = _evaluate_objective_score_values(
        context.question_type_code,
        context.expected_answer,
        context.student_answer,
        context.max_score,
    )
    if objective_score is not None:
        return _objective_result_direct(context, objective_score)

    if context.rubric_items:
        parsed, _, _, _, _ = grade_with_rubric(
            context.question_text,
            context.max_score,
            [
                {
                    "rubric_index": item.rubric_index,
                    "description": item.description,
                    "max_marks": item.max_marks,
                }
                for item in context.rubric_items
            ],
            _truncate_student_answer(context.student_answer),
            llm_client,
        )
        return _rubric_result_direct(context, context.rubric_items, parsed)

    if should_use_asag(context.question_type_code, context.expected_answer, len(context.rubric_items), context.max_score):
        try:
            asag_result = score_short_answer(
                question=context.question_text,
                reference_answer=context.expected_answer or "",
                student_answer=_truncate_student_answer(context.student_answer),
                max_score=context.max_score,
                expected_points=context.expected_points,
            )
            return _asag_result_direct(context, asag_result)
        except (AsagConfigurationError, AsagInferenceError) as exc:
            print(f"[asag] direct grading fallback to holistic: {exc}")

    if not options.allow_holistic_fallback:
        raise ValueError("No marking guide is available for this question and holistic fallback is disabled.")

    parsed, _, _, _, _ = grade_holistically(
        context.question_text,
        context.max_score,
        _truncate_student_answer(context.student_answer),
        llm_client,
        expected_answer=context.expected_answer,
        expected_points=context.expected_points,
    )
    return _holistic_result_direct(context, parsed)



def _load_attempt_answer_context(session: Session, attempt_answer_id: UUID) -> GradingContext:
    stmt = (
        select(LmsAttemptAnswer)
        .where(LmsAttemptAnswer.id == attempt_answer_id, LmsAttemptAnswer.deleted_at.is_(None))
        .options(
            selectinload(LmsAttemptAnswer.assessment_question).selectinload(LmsAssessmentQuestion.question),
            selectinload(LmsAttemptAnswer.assessment_question).selectinload(LmsAssessmentQuestion.rubric_scheme),
            selectinload(LmsAttemptAnswer.assessment_attempt)
            .selectinload(LmsAssessmentAttempt.assessment_enrollment)
            .selectinload(LmsAssessmentEnrollment.assessment_assignment)
            .selectinload(LmsAssessmentAssignment.assessment),
        )
    )
    answer = session.execute(stmt).scalar_one_or_none()
    if answer is None:
        raise GradingNotFoundError(f"Attempt answer not found: {attempt_answer_id}")
    return _build_context(answer)



def _load_assessment_attempt(session: Session, assessment_attempt_id: UUID) -> LmsAssessmentAttempt:
    stmt = (
        select(LmsAssessmentAttempt)
        .where(LmsAssessmentAttempt.id == assessment_attempt_id, LmsAssessmentAttempt.deleted_at.is_(None))
        .options(
            selectinload(LmsAssessmentAttempt.answers)
            .selectinload(LmsAttemptAnswer.assessment_question)
            .selectinload(LmsAssessmentQuestion.question),
            selectinload(LmsAssessmentAttempt.answers)
            .selectinload(LmsAttemptAnswer.assessment_question)
            .selectinload(LmsAssessmentQuestion.rubric_scheme),
            selectinload(LmsAssessmentAttempt.assessment_enrollment)
            .selectinload(LmsAssessmentEnrollment.assessment_assignment)
            .selectinload(LmsAssessmentAssignment.assessment),
        )
    )
    attempt = session.execute(stmt).scalar_one_or_none()
    if attempt is None:
        raise GradingNotFoundError(f"Assessment attempt not found: {assessment_attempt_id}")
    return attempt



def _build_context(answer: LmsAttemptAnswer) -> GradingContext:
    attempt = answer.assessment_attempt
    enrollment = attempt.assessment_enrollment
    assignment = enrollment.assessment_assignment
    school_id = assignment.assessment.school_id if assignment and assignment.assessment else None
    student_id = enrollment.student_id if enrollment else None
    max_score = float(answer.max_score or answer.assessment_question.points or answer.assessment_question.question.max_mark or 0.0)
    return GradingContext(
        answer=answer,
        student_answer=_resolve_student_answer(answer),
        max_score=max_score,
        school_id=school_id,
        student_id=student_id,
    )



def _grade_question_context(
    session: Session,
    context: GradingContext,
    options: GradeRequestOptions,
    llm_client: LLMClient,
) -> QuestionGradeResult:
    answer = context.answer
    if answer.human_score is not None:
        return _existing_result(context, source="human")
    if not options.force and (answer.ai_score is not None or answer.graded_at is not None):
        return _existing_result(context, source="ai")
    if not context.student_answer:
        result = _no_answer_result(context)
        if not options.dry_run:
            _apply_question_result(session, context, result)
        return result

    objective_score = _evaluate_objective_score(context)
    if objective_score is not None:
        result = _objective_result(context, objective_score)
        if not options.dry_run:
            _apply_question_result(session, context, result)
        return result

    rubric_items, scheme = _resolve_rubric_items(session, context)
    if rubric_items:
        parsed, raw, elapsed, system_text, user_text = grade_with_rubric(
            context.question.stem,
            context.max_score,
            [
                {
                    "rubric_index": item.rubric_index,
                    "description": item.description,
                    "max_marks": item.max_marks,
                }
                for item in rubric_items
            ],
            _truncate_student_answer(context.student_answer),
            llm_client,
        )
        result = _rubric_result(context, rubric_items, parsed)
        if not options.dry_run:
            trace_id = _persist_inference_run(
                session,
                context,
                mode="rubric",
                system_text=system_text,
                user_text=user_text,
                raw_text=raw,
                parsed_payload=parsed.model_dump(),
                latency_ms=int(elapsed * 1000),
                rubric_items=rubric_items,
                scheme=scheme,
            )
            result.trace_id = trace_id
            _apply_question_result(session, context, result)
        return result

    expected_answer = _extract_correct_answer(context.question.rubric_json)
    expected_points = _extract_expected_points_from_rubric_json(context.question.rubric_json)
    if should_use_asag(context.question.question_type_code, expected_answer, len(rubric_items), context.max_score):
        try:
            asag_result = score_short_answer(
                question=context.question.stem,
                reference_answer=expected_answer or "",
                student_answer=_truncate_student_answer(context.student_answer),
                max_score=context.max_score,
                expected_points=expected_points,
            )
            result = _asag_result(context, asag_result)
            if not options.dry_run:
                trace_id = _persist_asag_inference_run(
                    session,
                    context,
                    expected_answer=expected_answer or "",
                    expected_points=expected_points,
                    asag_result=asag_result.model_dump(),
                )
                result.trace_id = trace_id
                _apply_question_result(session, context, result)
            return result
        except (AsagConfigurationError, AsagInferenceError) as exc:
            print(f"[asag] shared grading fallback to holistic: {exc}")

    parsed, raw, elapsed, system_text, user_text = grade_holistically(
        context.question.stem,
        context.max_score,
        _truncate_student_answer(context.student_answer),
        llm_client,
        expected_answer=expected_answer,
        expected_points=expected_points,
    )
    result = _holistic_result(context, parsed)
    if not options.dry_run:
        trace_id = _persist_inference_run(
            session,
            context,
            mode="holistic",
            system_text=system_text,
            user_text=user_text,
            raw_text=raw,
            parsed_payload=parsed.model_dump(),
            latency_ms=int(elapsed * 1000),
            rubric_items=[],
            scheme=scheme,
        )
        result.trace_id = trace_id
        _apply_question_result(session, context, result)
    return result



def _existing_result(context: GradingContext, source: str) -> QuestionGradeResult:
    answer = context.answer
    score = _effective_score(answer)
    feedback_text = answer.feedback_text or f"Existing {source} grading found."
    return QuestionGradeResult(
        mode="existing",
        attempt_answer_id=str(answer.id),
        assessment_attempt_id=str(answer.assessment_attempt_id),
        assessment_question_id=str(answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=round(score, 2),
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, []),
        strengths=_derive_generic_strengths(score, context.max_score),
        missing_points=[],
        next_steps=_derive_next_steps([], score, context.max_score),
        confidence=float(answer.ai_confidence) if answer.ai_confidence is not None else None,
        requires_review=bool(answer.requires_review),
        trace_id=answer.answer_trace_id,
        rubric_items=[],
    )



def _no_answer_result(context: GradingContext) -> QuestionGradeResult:
    feedback_text = "No answer was submitted. Review the question and provide the key ideas next time."
    missing_points = ["No answer submitted."]
    return QuestionGradeResult(
        mode="no_answer",
        attempt_answer_id=str(context.answer.id),
        assessment_attempt_id=str(context.answer.assessment_attempt_id),
        assessment_question_id=str(context.answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=0.0,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=[],
        missing_points=missing_points,
        next_steps=["State the main idea even if you are unsure, then refine it."],
        confidence=1.0,
        requires_review=False,
        trace_id=None,
        rubric_items=[],
    )



def _objective_result(context: GradingContext, score: float) -> QuestionGradeResult:
    correct_answer = _extract_correct_answer(context.question.rubric_json)
    if score > 0:
        feedback = "Correct answer."
        missing_points: list[str] = []
        strengths = ["You matched the expected answer."]
    else:
        feedback = "Incorrect answer."
        if correct_answer:
            feedback += f" Correct answer: {correct_answer}."
        missing_points = ["Match the expected answer exactly."]
        strengths = []
    return QuestionGradeResult(
        mode="objective",
        attempt_answer_id=str(context.answer.id),
        assessment_attempt_id=str(context.answer.assessment_attempt_id),
        assessment_question_id=str(context.answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=round(score, 2),
        max_score=round(context.max_score, 2),
        feedback_text=feedback,
        feedback_summary=_feedback_summary(feedback, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score, context.max_score),
        confidence=1.0,
        requires_review=False,
        trace_id=None,
        rubric_items=[],
    )



def _rubric_result(context: GradingContext, rubric_items: list[RubricDescriptor], parsed) -> QuestionGradeResult:
    decisions = {int(item.rubric_index): item for item in parsed.items}
    outcomes: list[QuestionRubricOutcome] = []
    total = 0.0
    for descriptor in rubric_items:
        decision = decisions.get(descriptor.rubric_index)
        awarded = 0.0 if decision is None else max(0.0, min(float(decision.awarded), descriptor.max_marks))
        reason = decision.reason.strip() if decision is not None else "Not demonstrated."
        total += awarded
        outcomes.append(
            QuestionRubricOutcome(
                rubric_index=descriptor.rubric_index,
                rubric_item_id=descriptor.rubric_item_id,
                description=descriptor.description,
                max_marks=round(descriptor.max_marks, 2),
                awarded=round(awarded, 2),
                reason=reason,
                rubric_code=descriptor.rubric_code,
            )
        )

    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    if not missing_points:
        missing_points = [item.description for item in rubric_items if decisions.get(item.rubric_index) is None or float(decisions[item.rubric_index].awarded) <= 0]
    feedback_text = parsed.feedback_text.strip() or _build_feedback_from_missing_points(missing_points)
    confidence = max(0.0, min(float(parsed.confidence), 1.0))
    requires_review = confidence < REVIEW_CONFIDENCE_THRESHOLD
    score_awarded = round(min(total, context.max_score), 2)
    strengths = _derive_strengths_from_rubric(outcomes)
    if not strengths:
        strengths = _derive_generic_strengths(score_awarded, context.max_score)

    return QuestionGradeResult(
        mode="rubric",
        attempt_answer_id=str(context.answer.id),
        assessment_attempt_id=str(context.answer.assessment_attempt_id),
        assessment_question_id=str(context.answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score_awarded, context.max_score),
        confidence=confidence,
        requires_review=requires_review,
        trace_id=None,
        rubric_items=outcomes,
    )



def _holistic_result(context: GradingContext, parsed) -> QuestionGradeResult:
    confidence = max(0.0, min(float(parsed.confidence), 1.0))
    feedback_text = parsed.feedback_text.strip() or parsed.reason.strip()
    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    score_awarded = round(min(float(parsed.score_awarded), context.max_score), 2)
    if not missing_points and score_awarded < context.max_score:
        missing_points = ["Key ideas from the expected answer were missing."]
    strengths = _derive_generic_strengths(score_awarded, context.max_score, fallback=parsed.reason.strip())
    return QuestionGradeResult(
        mode="holistic",
        attempt_answer_id=str(context.answer.id),
        assessment_attempt_id=str(context.answer.assessment_attempt_id),
        assessment_question_id=str(context.answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score_awarded, context.max_score),
        confidence=confidence,
        requires_review=True,
        trace_id=None,
        rubric_items=[],
    )


def _asag_result(context: GradingContext, parsed) -> QuestionGradeResult:
    score_awarded = round(min(float(parsed.scaled_score), context.max_score), 2)
    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    return QuestionGradeResult(
        mode="asag",
        attempt_answer_id=str(context.answer.id),
        assessment_attempt_id=str(context.answer.assessment_attempt_id),
        assessment_question_id=str(context.answer.assessment_question_id),
        question_id=str(context.question.id),
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=parsed.feedback_text,
        feedback_summary=parsed.feedback_summary,
        strengths=list(parsed.strengths),
        missing_points=missing_points,
        next_steps=list(parsed.next_steps),
        confidence=float(parsed.confidence),
        requires_review=float(parsed.confidence) < REVIEW_CONFIDENCE_THRESHOLD,
        trace_id=None,
        rubric_items=[],
    )


def _no_answer_result_direct(context: DirectGradingContext) -> QuestionGradeResult:
    feedback_text = "No answer was submitted. Review the question and provide the key ideas next time."
    missing_points = ["No answer submitted."]
    return QuestionGradeResult(
        mode="no_answer",
        attempt_answer_id=context.attempt_answer_id,
        assessment_attempt_id=context.assessment_attempt_id,
        assessment_question_id=context.assessment_question_id,
        question_id=context.question_id,
        score_awarded=0.0,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=[],
        missing_points=missing_points,
        next_steps=["State the main idea even if you are unsure, then refine it."],
        confidence=1.0,
        requires_review=False,
        trace_id=None,
        rubric_items=[],
    )


def _objective_result_direct(context: DirectGradingContext, score: float) -> QuestionGradeResult:
    if score > 0:
        feedback = "Correct answer."
        missing_points: list[str] = []
        strengths = ["You matched the expected answer."]
    else:
        feedback = "Incorrect answer."
        if context.expected_answer:
            feedback += f" Correct answer: {context.expected_answer}."
        missing_points = ["Match the expected answer exactly."]
        strengths = []
    return QuestionGradeResult(
        mode="objective",
        attempt_answer_id=context.attempt_answer_id,
        assessment_attempt_id=context.assessment_attempt_id,
        assessment_question_id=context.assessment_question_id,
        question_id=context.question_id,
        score_awarded=round(score, 2),
        max_score=round(context.max_score, 2),
        feedback_text=feedback,
        feedback_summary=_feedback_summary(feedback, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score, context.max_score),
        confidence=1.0,
        requires_review=False,
        trace_id=None,
        rubric_items=[],
    )


def _rubric_result_direct(context: DirectGradingContext, rubric_items: list[RubricDescriptor], parsed) -> QuestionGradeResult:
    decisions = {int(item.rubric_index): item for item in parsed.items}
    outcomes: list[QuestionRubricOutcome] = []
    total = 0.0
    for descriptor in rubric_items:
        decision = decisions.get(descriptor.rubric_index)
        awarded = 0.0 if decision is None else max(0.0, min(float(decision.awarded), descriptor.max_marks))
        reason = decision.reason.strip() if decision is not None else "Not demonstrated."
        total += awarded
        outcomes.append(
            QuestionRubricOutcome(
                rubric_index=descriptor.rubric_index,
                rubric_item_id=descriptor.rubric_item_id,
                description=descriptor.description,
                max_marks=round(descriptor.max_marks, 2),
                awarded=round(awarded, 2),
                reason=reason,
                rubric_code=descriptor.rubric_code,
            )
        )

    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    if not missing_points:
        missing_points = [item.description for item in rubric_items if decisions.get(item.rubric_index) is None or float(decisions[item.rubric_index].awarded) <= 0]
    feedback_text = parsed.feedback_text.strip() or _build_feedback_from_missing_points(missing_points)
    confidence = max(0.0, min(float(parsed.confidence), 1.0))
    score_awarded = round(min(total, context.max_score), 2)
    strengths = _derive_strengths_from_rubric(outcomes)
    if not strengths:
        strengths = _derive_generic_strengths(score_awarded, context.max_score)
    return QuestionGradeResult(
        mode="rubric",
        attempt_answer_id=context.attempt_answer_id,
        assessment_attempt_id=context.assessment_attempt_id,
        assessment_question_id=context.assessment_question_id,
        question_id=context.question_id,
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score_awarded, context.max_score),
        confidence=confidence,
        requires_review=confidence < REVIEW_CONFIDENCE_THRESHOLD,
        trace_id=None,
        rubric_items=outcomes,
    )


def _holistic_result_direct(context: DirectGradingContext, parsed) -> QuestionGradeResult:
    confidence = max(0.0, min(float(parsed.confidence), 1.0))
    feedback_text = parsed.feedback_text.strip() or parsed.reason.strip()
    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    score_awarded = round(min(float(parsed.score_awarded), context.max_score), 2)
    if not missing_points and score_awarded < context.max_score:
        missing_points = ["Key ideas from the expected answer were missing."]
    strengths = _derive_generic_strengths(score_awarded, context.max_score, fallback=parsed.reason.strip())
    return QuestionGradeResult(
        mode="holistic",
        attempt_answer_id=context.attempt_answer_id,
        assessment_attempt_id=context.assessment_attempt_id,
        assessment_question_id=context.assessment_question_id,
        question_id=context.question_id,
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=feedback_text,
        feedback_summary=_feedback_summary(feedback_text, missing_points),
        strengths=strengths,
        missing_points=missing_points,
        next_steps=_derive_next_steps(missing_points, score_awarded, context.max_score),
        confidence=confidence,
        requires_review=True,
        trace_id=None,
        rubric_items=[],
    )


def _asag_result_direct(context: DirectGradingContext, parsed) -> QuestionGradeResult:
    score_awarded = round(min(float(parsed.scaled_score), context.max_score), 2)
    missing_points = [point.strip() for point in parsed.missing_points if isinstance(point, str) and point.strip()]
    return QuestionGradeResult(
        mode="asag",
        attempt_answer_id=context.attempt_answer_id,
        assessment_attempt_id=context.assessment_attempt_id,
        assessment_question_id=context.assessment_question_id,
        question_id=context.question_id,
        score_awarded=score_awarded,
        max_score=round(context.max_score, 2),
        feedback_text=parsed.feedback_text,
        feedback_summary=parsed.feedback_summary,
        strengths=list(parsed.strengths),
        missing_points=missing_points,
        next_steps=list(parsed.next_steps),
        confidence=float(parsed.confidence),
        requires_review=float(parsed.confidence) < REVIEW_CONFIDENCE_THRESHOLD,
        trace_id=None,
        rubric_items=[],
    )



def _apply_question_result(session: Session, context: GradingContext, result: QuestionGradeResult) -> None:
    answer = context.answer
    answer.ai_score = round(float(result.score_awarded), 2)
    answer.ai_confidence = float(result.confidence) if result.confidence is not None else None
    answer.feedback_text = result.feedback_text
    answer.requires_review = bool(result.requires_review)
    answer.graded_at = _utcnow()
    answer.answer_trace_id = result.trace_id
    session.flush()



def _refresh_attempt_rollup(session: Session, assessment_attempt_id: UUID) -> None:
    attempt = _load_assessment_attempt(session, assessment_attempt_id)
    answers = attempt.answers
    total_score = sum(_effective_score(answer) for answer in answers)
    max_score = _attempt_max_score(attempt)
    ai_confidence_values = [float(answer.ai_confidence) for answer in answers if answer.ai_confidence is not None]
    requires_review = any(bool(answer.requires_review) for answer in answers)
    all_scored = all(_has_any_score(answer) for answer in answers)

    attempt.total_score = round(total_score, 2)
    attempt.max_score = round(max_score, 2)
    attempt.ai_confidence = round(sum(ai_confidence_values) / len(ai_confidence_values), 4) if ai_confidence_values else None
    attempt.grading_status_code = "auto_graded" if all_scored and not requires_review else "pending"
    attempt.attempt_trace_id = next((answer.answer_trace_id for answer in answers if answer.answer_trace_id), attempt.attempt_trace_id)
    session.flush()

    _upsert_assessment_result(session, attempt)



def _upsert_assessment_result(session: Session, attempt: LmsAssessmentAttempt) -> None:
    enrollment = attempt.assessment_enrollment
    assignment = enrollment.assessment_assignment
    student_id = enrollment.student_id
    stmt = select(LmsAssessmentResult).where(
        LmsAssessmentResult.assessment_assignment_id == assignment.id,
        LmsAssessmentResult.student_id == student_id,
        LmsAssessmentResult.deleted_at.is_(None),
    )
    result = session.execute(stmt).scalar_one_or_none()
    if result is None:
        result = LmsAssessmentResult(
            assessment_assignment_id=assignment.id,
            student_id=student_id,
            status="draft",
        )
        session.add(result)

    answer_rows = attempt.answers
    all_scored = all(_has_any_score(answer) for answer in answer_rows)
    requires_review = any(bool(answer.requires_review) for answer in answer_rows)
    total_score = round(sum(_effective_score(answer) for answer in answer_rows), 2)

    result.finalized_attempt_id = attempt.id
    result.expected_mark = round(_attempt_max_score(attempt), 2)
    result.submitted_at = attempt.submitted_at
    result.feedback = _build_attempt_feedback(answer_rows)
    if all_scored and not requires_review:
        result.actual_mark = total_score
        result.graded_at = _utcnow()
        result.status = "published"
    else:
        result.actual_mark = None
        result.graded_at = None
        result.status = "draft"
    session.flush()



def _build_attempt_feedback(answer_rows: list[LmsAttemptAnswer]) -> str:
    lines: list[str] = []
    ordered_answers = sorted(answer_rows, key=lambda item: item.assessment_question.sequence_index)
    for answer in ordered_answers:
        if not answer.feedback_text:
            continue
        label = f"Q{answer.assessment_question.sequence_index}"
        lines.append(f"{label}: {answer.feedback_text}")
    return "\n".join(lines)



def _persist_inference_run(
    session: Session,
    context: GradingContext,
    mode: str,
    system_text: str,
    user_text: str,
    raw_text: str,
    parsed_payload: dict[str, Any],
    latency_ms: int,
    rubric_items: list[RubricDescriptor],
    scheme: LmsMarkingScheme | None,
) -> str:
    model_version = _ensure_llm_model_version(session)
    trace_id = _new_trace_id()
    run = AiInferenceRun(
        trace_id=trace_id,
        model_version_id=model_version.id,
        school_id=context.school_id,
        student_id=context.student_id,
        assessment_attempt_id=context.answer.assessment_attempt_id,
        attempt_answer_id=context.answer.id,
        prompt_text=system_text,
        context_json={
            "mode": mode,
            "question_id": str(context.question.id),
            "assessment_question_id": str(context.answer.assessment_question_id),
            "question_type_code": context.question.question_type_code,
            "max_score": context.max_score,
            "rubric": [
                {
                    "rubric_index": item.rubric_index,
                    "rubric_item_id": item.rubric_item_id,
                    "description": item.description,
                    "max_marks": item.max_marks,
                    "rubric_code": item.rubric_code,
                }
                for item in rubric_items
            ],
        },
        rubric_scheme_id=scheme.id if scheme else None,
        rubric_scheme_version=scheme.version if scheme else None,
        request_json={
            "system_text": system_text,
            "user_text": user_text,
        },
        response_json={
            "raw_text": raw_text,
            "parsed": parsed_payload,
        },
        latency_ms=latency_ms,
        created_at=_utcnow(),
    )
    session.add(run)
    session.flush()
    return trace_id


def _persist_asag_inference_run(
    session: Session,
    context: GradingContext,
    expected_answer: str,
    expected_points: list[str],
    asag_result: dict[str, Any],
) -> str:
    model_version = _ensure_asag_model_version(session)
    trace_id = _new_trace_id(prefix="asag")
    run = AiInferenceRun(
        trace_id=trace_id,
        model_version_id=model_version.id,
        school_id=context.school_id,
        student_id=context.student_id,
        assessment_attempt_id=context.answer.assessment_attempt_id,
        attempt_answer_id=context.answer.id,
        prompt_text=None,
        context_json={
            "mode": "asag",
            "question_id": str(context.question.id),
            "assessment_question_id": str(context.answer.assessment_question_id),
            "question_type_code": context.question.question_type_code,
            "max_score": context.max_score,
            "expected_points": expected_points,
        },
        rubric_scheme_id=context.assessment_question.rubric_scheme_id,
        rubric_scheme_version=context.assessment_question.rubric_scheme_version,
        request_json={
            "question": context.question.stem,
            "reference_answer": expected_answer,
            "student_answer": _truncate_student_answer(context.student_answer),
        },
        response_json=asag_result,
        latency_ms=None,
        created_at=_utcnow(),
    )
    session.add(run)
    session.flush()
    return trace_id


def _ensure_llm_model_version(session: Session) -> AiModelVersion:
    model_name = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    version_name = os.getenv("MODEL_VERSION", model_name)
    stmt = (
        select(AiModelVersion)
        .join(AiModel, AiModelVersion.model_id == AiModel.id)
        .where(
            AiModel.name == model_name,
            AiModel.model_type == "llm",
            AiModel.deleted_at.is_(None),
            AiModelVersion.version == version_name,
            AiModelVersion.deleted_at.is_(None),
        )
    )
    model_version = session.execute(stmt).scalar_one_or_none()
    if model_version is not None:
        return model_version

    model_stmt = select(AiModel).where(
        AiModel.name == model_name,
        AiModel.model_type == "llm",
        AiModel.deleted_at.is_(None),
    )
    model = session.execute(model_stmt).scalar_one_or_none()
    if model is None:
        model = AiModel(
            name=model_name,
            model_type="llm",
            description="MindSpore/MindNLP grading model",
            is_active=True,
        )
        session.add(model)
        session.flush()

    model_version = AiModelVersion(
        model_id=model.id,
        version=version_name,
        artifact_uri=f"huggingface://{model_name}",
        metrics=None,
        config={
            "provider": os.getenv("LLM_PROVIDER", "mindnlp"),
            "ms_mode": os.getenv("MS_MODE", "GRAPH_MODE"),
            "device_target": os.getenv("MS_DEVICE_TARGET", "CPU"),
            "max_new_tokens": os.getenv("MAX_NEW_TOKENS", "128"),
        },
        is_active=True,
    )
    session.add(model_version)
    session.flush()
    return model_version


def _ensure_asag_model_version(session: Session) -> AiModelVersion:
    model_name = os.getenv("ASAG_MODEL_NAME", "asag_mohler")
    version_name = os.getenv("ASAG_MODEL_VERSION", "asag_mohler_best")
    artifact_uri = os.getenv("ASAG_CKPT_PATH") or os.getenv("ASAG_MINDIR_PATH") or os.getenv("ASAG_MODEL_DIR", "models/asag")

    stmt = (
        select(AiModelVersion)
        .join(AiModel, AiModelVersion.model_id == AiModel.id)
        .where(
            AiModel.name == model_name,
            AiModel.model_type == "asag",
            AiModel.deleted_at.is_(None),
            AiModelVersion.version == version_name,
            AiModelVersion.deleted_at.is_(None),
        )
    )
    model_version = session.execute(stmt).scalar_one_or_none()
    if model_version is not None:
        return model_version

    model_stmt = select(AiModel).where(
        AiModel.name == model_name,
        AiModel.model_type == "asag",
        AiModel.deleted_at.is_(None),
    )
    model = session.execute(model_stmt).scalar_one_or_none()
    if model is None:
        model = AiModel(
            name=model_name,
            model_type="asag",
            description="MindSpore short-answer regression model",
            is_active=True,
        )
        session.add(model)
        session.flush()

    model_version = AiModelVersion(
        model_id=model.id,
        version=version_name,
        artifact_uri=artifact_uri,
        metrics=None,
        config={
            "provider": "mindspore",
            "artifact_dir": os.getenv("ASAG_MODEL_DIR", "models/asag"),
            "max_length": os.getenv("ASAG_MAX_LENGTH", "256"),
            "raw_score_max": os.getenv("ASAG_RAW_SCORE_MAX", "5.0"),
        },
        is_active=True,
    )
    session.add(model_version)
    session.flush()
    return model_version



def _resolve_rubric_items(session: Session, context: GradingContext) -> tuple[list[RubricDescriptor], LmsMarkingScheme | None]:
    scheme = context.assessment_question.rubric_scheme
    if scheme is None:
        scheme = _load_latest_active_scheme_for_question(session, context.question.id)
    if scheme is not None:
        return _load_scheme_items(session, scheme), scheme
    return _extract_rubric_items_from_rubric_json(context.question.rubric_json), None



def _load_latest_active_scheme_for_question(session: Session, question_id: UUID) -> LmsMarkingScheme | None:
    stmt = (
        select(LmsMarkingScheme)
        .where(
            LmsMarkingScheme.question_id == question_id,
            LmsMarkingScheme.is_active.is_(True),
            LmsMarkingScheme.deleted_at.is_(None),
        )
        .order_by(LmsMarkingScheme.version.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()



def _load_scheme_items(session: Session, scheme: LmsMarkingScheme) -> list[RubricDescriptor]:
    stmt = (
        select(LmsMarkingSchemeItem)
        .where(
            LmsMarkingSchemeItem.marking_scheme_id == scheme.id,
            LmsMarkingSchemeItem.deleted_at.is_(None),
        )
        .order_by(LmsMarkingSchemeItem.step_index.asc())
    )
    items = session.execute(stmt).scalars().all()
    return [
        RubricDescriptor(
            rubric_index=item.step_index,
            description=item.description,
            max_marks=float(item.mark_value),
            rubric_item_id=str(item.id),
            rubric_code=item.rubric_code,
        )
        for item in items
    ]



def _extract_rubric_items_from_rubric_json(rubric_json: Any) -> list[RubricDescriptor]:
    if rubric_json is None:
        return []

    raw_items: list[Any] = []
    if isinstance(rubric_json, list):
        raw_items = rubric_json
    elif isinstance(rubric_json, dict):
        for key in ("markingPoints", "marking_points", "criteria", "steps", "keyPoints", "key_points"):
            node = rubric_json.get(key)
            if isinstance(node, list):
                raw_items.extend(node)
            elif node is not None:
                raw_items.append(node)
    else:
        return []

    items: list[RubricDescriptor] = []
    for index, node in enumerate(raw_items, start=1):
        descriptor = _rubric_descriptor_from_node(index, node)
        if descriptor is not None:
            items.append(descriptor)
    return items



def _rubric_descriptor_from_node(index: int, node: Any) -> RubricDescriptor | None:
    if node is None:
        return None
    if isinstance(node, str):
        text = node.strip()
        if not text:
            return None
        return RubricDescriptor(rubric_index=index, description=text, max_marks=1.0)
    if not isinstance(node, dict):
        text = str(node).strip()
        if not text:
            return None
        return RubricDescriptor(rubric_index=index, description=text, max_marks=1.0)

    description = _first_non_blank(
        node.get("description"),
        node.get("point"),
        node.get("criterion"),
        node.get("text"),
        node.get("expected"),
        node.get("answer"),
    )
    if description is None:
        return None
    mark_value = _first_numeric(node.get("mark"), node.get("marks"), node.get("score"), node.get("value"), 1.0)
    rubric_code = _as_text(node.get("rubricCode")) or _as_text(node.get("rubric_code"))
    return RubricDescriptor(
        rubric_index=index,
        description=description,
        max_marks=float(mark_value),
        rubric_item_id=None,
        rubric_code=rubric_code,
    )



def _extract_expected_points_from_rubric_json(rubric_json: Any) -> list[str]:
    if rubric_json is None:
        return []
    points = [item.description for item in _extract_rubric_items_from_rubric_json(rubric_json)]
    if points:
        return points
    if isinstance(rubric_json, dict):
        for key in ("expectedAnswer", "modelAnswer", "correctAnswer", "correct_answer", "answer"):
            value = _as_text(rubric_json.get(key))
            if value:
                return [value]
    if isinstance(rubric_json, str) and rubric_json.strip():
        return [rubric_json.strip()]
    return []



def _extract_correct_answer(rubric_json: Any) -> str | None:
    if not isinstance(rubric_json, dict):
        return None
    for key in ("correctAnswer", "correct_answer", "answer"):
        value = _as_text(rubric_json.get(key))
        if value:
            return value
    return None



def _evaluate_objective_score(context: GradingContext) -> float | None:
    return _evaluate_objective_score_values(
        context.question.question_type_code,
        _extract_correct_answer(context.question.rubric_json),
        context.student_answer,
        context.max_score,
    )


def _evaluate_objective_score_values(
    question_type_code: str | None,
    correct_answer: str | None,
    student_answer: str | None,
    max_score: float,
) -> float | None:
    question_type = (question_type_code or "").lower()
    objective = any(token in question_type for token in ("mcq", "multiple_choice", "true_false", "truefalse", "boolean"))
    if not objective:
        return None
    if not correct_answer or not student_answer:
        return None
    if _normalize_answer(correct_answer) == _normalize_answer(student_answer):
        return round(max_score, 2)
    return 0.0



def _resolve_student_answer(answer: LmsAttemptAnswer) -> str | None:
    for value in (answer.text_content, answer.student_answer_text, answer.ocr_text):
        if value and value.strip():
            return value.strip()

    blob = answer.student_answer_blob
    if blob is None:
        return None
    if isinstance(blob, bool):
        return str(blob).lower()
    if isinstance(blob, str):
        return blob.strip() or None
    if isinstance(blob, list) and blob:
        return _as_text(blob[0])
    if isinstance(blob, dict):
        for key in ("answer", "selected", "selectedAnswer", "selectedOption", "value"):
            value = _as_text(blob.get(key))
            if value:
                return value
    text = _as_text(blob)
    return text or None



def _attempt_max_score(attempt: LmsAssessmentAttempt) -> float:
    values = [float(answer.max_score or 0.0) for answer in attempt.answers]
    return round(sum(values), 2)



def _effective_score(answer: LmsAttemptAnswer) -> float:
    if answer.human_score is not None:
        return float(answer.human_score)
    if answer.ai_score is not None:
        return float(answer.ai_score)
    return 0.0



def _has_any_score(answer: LmsAttemptAnswer) -> bool:
    return answer.human_score is not None or answer.ai_score is not None



def _average_confidence(results: list[QuestionGradeResult]) -> float | None:
    values = [float(result.confidence) for result in results if result.confidence is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)



def _derive_attempt_status(results: list[QuestionGradeResult]) -> str:
    all_scored = all(result.mode != "no_answer" or result.score_awarded == 0 for result in results)
    if all_scored and not any(result.requires_review for result in results):
        return "auto_graded"
    return "pending"


def _feedback_summary(feedback_text: str, missing_points: list[str]) -> str:
    text = (feedback_text or "").strip()
    if text:
        return text
    return _build_feedback_from_missing_points(missing_points)


def _derive_strengths_from_rubric(outcomes: list[QuestionRubricOutcome]) -> list[str]:
    strengths = [item.description for item in outcomes if float(item.awarded) > 0]
    return strengths[:3]


def _derive_generic_strengths(score_awarded: float, max_score: float, fallback: str | None = None) -> list[str]:
    if score_awarded >= max_score > 0:
        return [fallback or "You covered the key idea accurately."]
    if score_awarded > 0:
        return [fallback or "You showed some relevant understanding in your answer."]
    return []


def _derive_next_steps(missing_points: list[str], score_awarded: float, max_score: float) -> list[str]:
    if missing_points:
        return [f"Review: {point}" for point in missing_points[:3]]
    if score_awarded < max_score:
        return ["Revise the key idea and try a similar practice question."]
    return ["Keep using the same method on similar questions."]


def _build_feedback_from_missing_points(missing_points: list[str]) -> str:
    if not missing_points:
        return "Good attempt."
    lead = "; ".join(missing_points[:2])
    return f"Review these missing ideas: {lead}."



def _truncate_student_answer(student_answer: str | None) -> str:
    if not student_answer:
        return ""
    return student_answer[:MAX_STUDENT_ANSWER_CHARS]



def _normalize_answer(value: str) -> str:
    return value.strip().lower()



def _first_non_blank(*values: Any) -> str | None:
    for value in values:
        text = _as_text(value)
        if text:
            return text
    return None



def _first_numeric(*values: Any) -> float | None:
    for value in values:
        number = _to_float(value)
        if number is not None:
            return number
    return None



def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None



def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    return None



def _new_trace_id(prefix: str = "asag") -> str:
    return f"{prefix}-{uuid4().hex}"



def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
