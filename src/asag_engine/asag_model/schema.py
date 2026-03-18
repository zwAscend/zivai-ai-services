from pydantic import BaseModel, Field, confloat


class AsagScoreRequest(BaseModel):
    question: str = Field(..., min_length=1)
    reference_answer: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    max_score: confloat(gt=0) = 5.0
    expected_points: list[str] = Field(default_factory=list)


class AsagScoreResponse(BaseModel):
    raw_score_0_5: confloat(ge=0)
    normalized_score: confloat(ge=0, le=1)
    scaled_score: confloat(ge=0)
    band: str
    confidence: confloat(ge=0, le=1)
    backend: str
    artifact_path: str
    max_score: confloat(gt=0)
    feedback_text: str
    feedback_summary: str
    strengths: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
