from typing import Literal

from pydantic import BaseModel, Field, confloat, conint


class GradeRequestOptions(BaseModel):
    dry_run: bool = False
    force: bool = False


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
    mode: Literal["existing", "objective", "rubric", "holistic", "no_answer"]
    attempt_answer_id: str
    assessment_attempt_id: str
    assessment_question_id: str
    question_id: str
    score_awarded: confloat(ge=0)
    max_score: confloat(ge=0)
    feedback_text: str
    missing_points: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) | None = None
    requires_review: bool
    trace_id: str | None = None
    rubric_items: list[QuestionRubricOutcome] = Field(default_factory=list)


class AssessmentGradeResult(BaseModel):
    assessment_attempt_id: str
    grading_status_code: str
    total_score: confloat(ge=0)
    max_score: confloat(ge=0)
    ai_confidence: confloat(ge=0, le=1) | None = None
    requires_review: bool
    trace_id: str | None = None
    question_results: list[QuestionGradeResult]
