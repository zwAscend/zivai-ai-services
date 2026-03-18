from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class LmsSchool(Base):
    __tablename__ = "schools"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))


class LmsUser(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(100))


class LmsSchoolUser(Base):
    __tablename__ = "school_users"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    school_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.schools.id"))
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    is_active: Mapped[bool] = mapped_column(Boolean)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    school: Mapped[LmsSchool] = relationship()
    user: Mapped[LmsUser] = relationship()


class LmsSubject(Base):
    __tablename__ = "subjects"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    code: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(200))


class LmsTopic(Base):
    __tablename__ = "topics"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))


class LmsSkill(Base):
    __tablename__ = "skills"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.subjects.id"))
    topic_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.topics.id"))
    code: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    sequence_index: Mapped[int | None] = mapped_column(Integer)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subject: Mapped[LmsSubject] = relationship()
    topic: Mapped[LmsTopic | None] = relationship()


class LmsQuestion(Base):
    __tablename__ = "questions"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.subjects.id"))
    topic_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.topics.id"))
    author_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    code: Mapped[str | None] = mapped_column(String(100))
    stem: Mapped[str] = mapped_column(Text)
    question_type_code: Mapped[str] = mapped_column(String(50))
    max_mark: Mapped[float] = mapped_column(Float)
    difficulty: Mapped[int | None] = mapped_column(Integer)
    exam_style_code: Mapped[str | None] = mapped_column(String(50))
    source_year: Mapped[int | None] = mapped_column(Integer)
    rubric_json: Mapped[dict | list | str | None] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subject: Mapped[LmsSubject] = relationship()
    topic: Mapped[LmsTopic | None] = relationship()


class LmsMarkingScheme(Base):
    __tablename__ = "marking_schemes"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    question_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.questions.id"))
    version: Mapped[int] = mapped_column(Integer)
    total_mark: Mapped[float] = mapped_column(Float)
    scheme_source: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    question: Mapped[LmsQuestion] = relationship()


class LmsMarkingSchemeItem(Base):
    __tablename__ = "marking_scheme_items"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    marking_scheme_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.marking_schemes.id"))
    step_index: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text)
    mark_value: Mapped[float] = mapped_column(Float)
    rubric_code: Mapped[str | None] = mapped_column(String(100))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    marking_scheme: Mapped[LmsMarkingScheme] = relationship()


class LmsAssessment(Base):
    __tablename__ = "assessments"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    school_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.schools.id"))
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.subjects.id"))
    name: Mapped[str] = mapped_column(String(255))
    assessment_type: Mapped[str] = mapped_column(String(16))
    max_score: Mapped[float] = mapped_column(Float)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    school: Mapped[LmsSchool] = relationship()
    subject: Mapped[LmsSubject] = relationship()


class LmsAssessmentQuestion(Base):
    __tablename__ = "assessment_questions"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    assessment_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessments.id"))
    question_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.questions.id"))
    sequence_index: Mapped[int] = mapped_column(Integer)
    points: Mapped[float] = mapped_column(Float)
    rubric_scheme_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.marking_schemes.id"))
    rubric_scheme_version: Mapped[int | None] = mapped_column(Integer)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment: Mapped[LmsAssessment] = relationship()
    question: Mapped[LmsQuestion] = relationship()
    rubric_scheme: Mapped[LmsMarkingScheme | None] = relationship()


class LmsAssessmentAssignment(Base):
    __tablename__ = "assessment_assignments"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    assessment_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessments.id"))
    title: Mapped[str | None] = mapped_column(String(200))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment: Mapped[LmsAssessment] = relationship()


class LmsAssessmentEnrollment(Base):
    __tablename__ = "assessment_enrollments"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    assessment_assignment_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_assignments.id"))
    student_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    status_code: Mapped[str] = mapped_column(String(50))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment_assignment: Mapped[LmsAssessmentAssignment] = relationship()
    student: Mapped[LmsUser] = relationship()


class LmsAssessmentAttempt(Base):
    __tablename__ = "assessment_attempts"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    assessment_enrollment_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_enrollments.id"))
    attempt_number: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    final_score: Mapped[float | None] = mapped_column(Float)
    final_grade: Mapped[str | None] = mapped_column(String(8))
    grading_status_code: Mapped[str] = mapped_column(String(50))
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    attempt_trace_id: Mapped[str | None] = mapped_column(String(64))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment_enrollment: Mapped[LmsAssessmentEnrollment] = relationship()
    answers: Mapped[list[LmsAttemptAnswer]] = relationship(back_populates="assessment_attempt")


class LmsAttemptAnswer(Base):
    __tablename__ = "attempt_answers"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    assessment_attempt_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_attempts.id"))
    assessment_question_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_questions.id"))
    student_answer_text: Mapped[str | None] = mapped_column(Text)
    student_answer_blob: Mapped[dict | list | str | None] = mapped_column(JSONB)
    text_content: Mapped[str | None] = mapped_column(Text)
    external_assessment_data: Mapped[dict | list | str | None] = mapped_column(JSONB)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    ai_score: Mapped[float | None] = mapped_column(Float)
    human_score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float)
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    requires_review: Mapped[bool] = mapped_column(Boolean)
    feedback_text: Mapped[str | None] = mapped_column(Text)
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    answer_trace_id: Mapped[str | None] = mapped_column(String(64))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment_attempt: Mapped[LmsAssessmentAttempt] = relationship(back_populates="answers")
    assessment_question: Mapped[LmsAssessmentQuestion] = relationship()


class LmsInteractionEvent(Base):
    __tablename__ = "interaction_events"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    school_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.schools.id"))
    student_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    subject_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.subjects.id"))
    skill_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.skills.id"))
    assessment_attempt_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_attempts.id"))
    attempt_answer_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.attempt_answers.id"))
    is_correct: Mapped[int] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trace_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    school: Mapped[LmsSchool] = relationship()
    student: Mapped[LmsUser] = relationship()
    subject: Mapped[LmsSubject | None] = relationship()
    skill: Mapped[LmsSkill] = relationship()


class LmsMasterySnapshot(Base):
    __tablename__ = "mastery_snapshots"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    student_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.subjects.id"))
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(50))
    average_mastery: Mapped[float | None] = mapped_column(Float)
    risk_level_code: Mapped[str | None] = mapped_column(String(20))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    student: Mapped[LmsUser] = relationship()
    subject: Mapped[LmsSubject] = relationship()


class LmsMasterySnapshotSkill(Base):
    __tablename__ = "mastery_snapshot_skills"
    __table_args__ = {"schema": "lms"}

    mastery_snapshot_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("lms.mastery_snapshots.id"),
        primary_key=True,
    )
    skill_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("lms.skills.id"),
        primary_key=True,
    )
    mastery_prob: Mapped[float] = mapped_column(Float)

    mastery_snapshot: Mapped[LmsMasterySnapshot] = relationship()
    skill: Mapped[LmsSkill] = relationship()


class LmsStudentAttribute(Base):
    __tablename__ = "student_attributes"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    student_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    skill_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.skills.id"))
    current_score: Mapped[float] = mapped_column(Float)
    potential_score: Mapped[float] = mapped_column(Float)
    last_assessed: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    student: Mapped[LmsUser] = relationship()
    skill: Mapped[LmsSkill] = relationship()


class LmsAssessmentResult(Base):
    __tablename__ = "assessment_results"
    __table_args__ = {"schema": "lms"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    assessment_assignment_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_assignments.id"))
    student_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    finalized_attempt_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_attempts.id"))
    expected_mark: Mapped[float | None] = mapped_column(Float)
    actual_mark: Mapped[float | None] = mapped_column(Float)
    grade: Mapped[str | None] = mapped_column(String(8))
    feedback: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assessment_assignment: Mapped[LmsAssessmentAssignment] = relationship()
    student: Mapped[LmsUser] = relationship()
    finalized_attempt: Mapped[LmsAssessmentAttempt | None] = relationship()


class AiModel(Base):
    __tablename__ = "ai_models"
    __table_args__ = {"schema": "ai"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    name: Mapped[str] = mapped_column(String(255))
    model_type: Mapped[str] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AiModelVersion(Base):
    __tablename__ = "ai_model_versions"
    __table_args__ = {"schema": "ai"}

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("util.gen_uuid_v4()"),
    )
    model_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("ai.ai_models.id"))
    version: Mapped[str] = mapped_column(String(64))
    artifact_uri: Mapped[str | None] = mapped_column(String(1024))
    metrics: Mapped[dict | list | str | None] = mapped_column(JSONB)
    config: Mapped[dict | list | str | None] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model: Mapped[AiModel] = relationship()


class AiInferenceRun(Base):
    __tablename__ = "ai_inference_runs"
    __table_args__ = {"schema": "ai"}

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("ai.ai_model_versions.id"))
    school_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.schools.id"))
    student_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.users.id"))
    assessment_attempt_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.assessment_attempts.id"))
    attempt_answer_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.attempt_answers.id"))
    prompt_text: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[dict | list | str | None] = mapped_column(JSONB)
    rubric_scheme_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("lms.marking_schemes.id"))
    rubric_scheme_version: Mapped[int | None] = mapped_column(Integer)
    request_json: Mapped[dict | list | str] = mapped_column(JSONB)
    response_json: Mapped[dict | list | str] = mapped_column(JSONB)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    model_version: Mapped[AiModelVersion] = relationship()
