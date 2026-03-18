from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DktEvent(BaseModel):
    skill_code: str
    is_correct: int
    score: float | None = None
    max_score: float | None = None
    event_time: datetime
    assessment_attempt_id: UUID | None = None
    attempt_answer_id: UUID | None = None
    attempt_id: str | None = None
    question_id: str | None = None

    @field_validator("skill_code")
    @classmethod
    def _skill_code_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("skill_code is required")
        return value

    @field_validator("is_correct")
    @classmethod
    def _coerce_binary(cls, value: int) -> int:
        normalized = int(value)
        if normalized not in (0, 1):
            raise ValueError("is_correct must be 0 or 1")
        return normalized


class DktUpdateRequest(BaseModel):
    student_id: str
    subject_code: str | None = None
    subject_id: UUID | None = None
    school_id: UUID | None = None
    events: list[DktEvent]
    persist: bool = True
    include_mastery_vector: bool = False

    @field_validator("student_id")
    @classmethod
    def _student_id_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("student_id is required")
        return value


class DktWeakSkill(BaseModel):
    skill_code: str
    mastery_prob: float
    skill_name: str | None = None
    skill_id: UUID | None = None


class DktUpdateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    student_id: UUID
    subject_id: UUID
    subject_code: str
    average_mastery: float
    risk_level: str
    weak_skills: list[DktWeakSkill]
    mastery_vector: dict[str, float] | None = None
    snapshot_id: UUID | None = None
    trace_id: str
    persisted: bool
    events_applied: int
    ignored_skill_codes: list[str] = Field(default_factory=list)
    snapshot_time: datetime
    model_name: str
    model_version: str


class DktMasteryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    student_id: UUID
    subject_id: UUID
    subject_code: str
    average_mastery: float
    risk_level: str
    weak_skills: list[DktWeakSkill]
    mastery_vector: dict[str, float] | None = None
    snapshot_id: UUID | None = None
    snapshot_time: datetime
    source: str
    trace_id: str | None = None
