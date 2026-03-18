from typing import Literal

from pydantic import BaseModel, Field, confloat, conint


class GradeRequestOptions(BaseModel):
    dry_run: bool = False
    force: bool = False
    allow_holistic_fallback: bool = True


class GradingRequestContext(BaseModel):
    assessment_attempt_id: str | None = None
    attempt_answer_id: str | None = None
    assessment_question_id: str | None = None
    question_id: str | None = None
    student_id: str | None = None
    school_id: str | None = None
    submitted_at: str | None = None


class QuestionPayload(BaseModel):
    text: str = Field(..., min_length=1)
    subject: str | None = None
    topic: str | None = None
    question_type: str | None = None
    max_marks: confloat(ge=0)


class StudentAnswerPayload(BaseModel):
    text: str | None = None


class MarkingGuideRubricItem(BaseModel):
    index: conint(ge=1)
    description: str = Field(..., min_length=1)
    marks: confloat(ge=0)
    keywords: list[str] = Field(default_factory=list)
    rubric_item_id: str | None = None
    rubric_code: str | None = None


class MarkingGuidePayload(BaseModel):
    rubric_items: list[MarkingGuideRubricItem] = Field(default_factory=list)
    expected_answer: str | None = None
    expected_points: list[str] = Field(default_factory=list)


class QuestionGradePayload(BaseModel):
    request_context: GradingRequestContext = Field(default_factory=GradingRequestContext)
    question: QuestionPayload
    student_answer: StudentAnswerPayload
    marking_guide: MarkingGuidePayload | None = None
    options: GradeRequestOptions = Field(default_factory=GradeRequestOptions)


class AssessmentQuestionPayload(BaseModel):
    request_context: GradingRequestContext = Field(default_factory=GradingRequestContext)
    question: QuestionPayload
    student_answer: StudentAnswerPayload
    marking_guide: MarkingGuidePayload | None = None


class AssessmentGradePayload(BaseModel):
    request_context: GradingRequestContext = Field(default_factory=GradingRequestContext)
    questions: list[AssessmentQuestionPayload] = Field(..., min_length=1)
    options: GradeRequestOptions = Field(default_factory=GradeRequestOptions)


class RubricDecision(BaseModel):
    rubric_index: conint(ge=1)
    awarded: confloat(ge=0)
    reason: str = Field(..., min_length=1)


class RubricLLMResult(BaseModel):
    items: list[RubricDecision]
    missing_points: list[str] = Field(default_factory=list)
    feedback_text: str = Field(..., min_length=1)
    confidence: confloat(ge=0, le=1)


class HolisticLLMResult(BaseModel):
    score_awarded: confloat(ge=0)
    feedback_text: str = Field(..., min_length=1)
    missing_points: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1)
    reason: str = Field(..., min_length=1)


class QuestionRubricOutcome(BaseModel):
    rubric_index: conint(ge=1)
    rubric_item_id: str | None = None
    description: str
    max_marks: confloat(ge=0)
    awarded: confloat(ge=0)
    reason: str
    rubric_code: str | None = None


class QuestionGradeResult(BaseModel):
    mode: Literal["existing", "objective", "rubric", "holistic", "asag", "no_answer"]
    attempt_answer_id: str | None = None
    assessment_attempt_id: str | None = None
    assessment_question_id: str | None = None
    question_id: str | None = None
    score_awarded: confloat(ge=0)
    max_score: confloat(ge=0)
    feedback_text: str
    feedback_summary: str
    strengths: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) | None = None
    requires_review: bool
    trace_id: str | None = None
    rubric_items: list[QuestionRubricOutcome] = Field(default_factory=list)


class AssessmentGradeResult(BaseModel):
    assessment_attempt_id: str | None = None
    grading_status_code: str
    total_score: confloat(ge=0)
    max_score: confloat(ge=0)
    ai_confidence: confloat(ge=0, le=1) | None = None
    requires_review: bool
    trace_id: str | None = None
    question_results: list[QuestionGradeResult]
