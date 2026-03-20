from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReferenceDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    documentName: str = Field(..., min_length=1)
    markdown: str = ""


class AssessmentGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    context: str = Field(default="Generate a well-structured assessment.")
    subjectName: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    questionTypes: Literal["multiple_choice", "structured", "mixed"] = "multiple_choice"
    numberOfQuestions: int = Field(default=5, ge=1, le=20)
    attributes: dict[str, Any] = Field(default_factory=dict)
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("context", mode="before")
    @classmethod
    def _normalize_context(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        return text or "Generate a well-structured assessment."

    @field_validator("attributes", mode="before")
    @classmethod
    def _normalize_attributes(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return {"subject": str(value)}

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class TeacherResourceGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subjectName: str | None = None
    topicTitle: str = Field(..., min_length=1)
    unitTitle: str | None = None
    gradeLevel: str | None = None
    contentType: Literal["resource", "practice"] = "resource"
    title: str | None = None
    objective: str | None = None
    teacherPrompt: str | None = None
    existingContent: str | None = None
    variant: bool = False
    relatedRecords: list[str] = Field(default_factory=list)
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)


class TeacherPracticeExistingQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = None
    questionType: Literal["multiple-choice", "short-answer"] | None = None
    type: Literal["multiple-choice", "short-answer"] | None = None
    marks: int = Field(default=1, ge=1, le=20)
    options: list[str] = Field(default_factory=list)
    correctAnswers: list[str] = Field(default_factory=list)
    correctAnswer: str | None = None
    markingGuide: str | None = None


class TeacherPracticeGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subjectName: str | None = None
    topicTitle: str = Field(..., min_length=1)
    unitTitle: str | None = None
    gradeLevel: str | None = None
    title: str | None = None
    objective: str | None = None
    teacherPrompt: str | None = None
    description: str | None = None
    practiceType: Literal["quiz", "assignment", "test", "project", "exam"] = "quiz"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    questionTypeMode: Literal["multiple_choice", "structured", "mixed"] = "mixed"
    numberOfQuestions: int = Field(default=5, ge=1, le=20)
    variant: bool = False
    relatedRecords: list[str] = Field(default_factory=list)
    existingQuestions: list[TeacherPracticeExistingQuestion] = Field(default_factory=list)
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)


class GeneratedRubricItem(BaseModel):
    index: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    marks: int = Field(..., ge=1)
    keywords: list[str] = Field(default_factory=list)


class GeneratedMarkingGuide(BaseModel):
    mode: Literal["objective", "rubric", "holistic"]
    expectedAnswer: str | None = None
    rubricItems: list[GeneratedRubricItem] = Field(default_factory=list)


class GeneratedAssessmentQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    type: Literal["multiple_choice", "true_false", "short_answer", "essay"]
    options: list[str] = Field(default_factory=list)
    correctAnswer: str | None = None
    correctAnswers: list[str] = Field(default_factory=list)
    explanation: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = Field(default_factory=list)
    points: int = Field(default=1, ge=1)
    maxMarks: int = Field(default=1, ge=1)
    markingGuide: GeneratedMarkingGuide | None = None
    rubricJson: dict[str, Any] = Field(default_factory=dict)
    referenceFallbackUsed: bool = False
    sourceDocumentsUsed: list[str] = Field(default_factory=list)


class GeneratedTeacherResource(BaseModel):
    title: str = Field(..., min_length=1)
    contentHtml: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    teacherMessage: str = Field(..., min_length=1)
    sourceDocumentsUsed: list[str] = Field(default_factory=list)
    referenceFallbackUsed: bool = False


class GeneratedTeacherPracticeQuestion(BaseModel):
    id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    type: Literal["multiple-choice", "short-answer"]
    marks: int = Field(default=1, ge=1, le=20)
    options: list[str] = Field(default_factory=list)
    correctAnswers: list[str] = Field(default_factory=list)
    correctAnswer: str = ""
    markingGuide: str = ""


class GeneratedTeacherPractice(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    practiceType: Literal["quiz", "assignment", "test", "project", "exam"] = "quiz"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    numberOfQuestions: int = Field(..., ge=1, le=20)
    questions: list[GeneratedTeacherPracticeQuestion] = Field(default_factory=list)
    summary: str = Field(..., min_length=1)
    teacherMessage: str = Field(..., min_length=1)
    sourceDocumentsUsed: list[str] = Field(default_factory=list)
    referenceFallbackUsed: bool = False


class LegacyPlanAttributeDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    currentScore: str
    potentialScore: str
    targetScore: str
    gap: str
    weight: str = "1"


class CanonicalPlanRequestContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    teacherId: str | None = None
    studentId: str | None = None
    subjectId: str | None = None
    schoolId: str | None = None


class CanonicalPlanStudentProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    firstName: str = Field(..., min_length=1)
    lastName: str = Field(..., min_length=1)
    email: str | None = None
    overallScore: float | None = None
    performance: str | None = None
    engagement: str | None = None
    strength: str | None = None


class CanonicalPlanSubject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: str = Field(..., min_length=1)


class CanonicalSelectedSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attributeId: str | None = None
    name: str = Field(..., min_length=1)
    currentScore: float | None = None
    potentialScore: float | None = None
    targetScore: float | None = None
    gap: float | None = None
    weight: float | None = None


class CanonicalCriticalSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attributeId: str | None = None
    name: str = Field(..., min_length=1)
    priority: int | None = None
    gap: float | None = None
    reason: str | None = None


class CanonicalSkillSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selectedSkills: list[CanonicalSelectedSkill] = Field(default_factory=list)
    criticalSkills: list[CanonicalCriticalSkill] = Field(default_factory=list)


class CanonicalPlanPreferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    context: str | None = None
    stepCount: int | None = None
    stepApproach: str | None = None
    objective: str | None = None
    guidance: str | None = None


class PlanGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    firstName: str | None = None
    lastName: str | None = None
    subjectName: str | None = None
    subjectID: str | None = None
    currentOverallScore: str | None = None
    potentialOverallScore: str | None = None
    targetScore: str | None = None
    overallPerformance: str | None = None
    overallEngagement: str | None = None
    attributeDetails: list[LegacyPlanAttributeDetail] = Field(default_factory=list)
    context: str | None = None
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)

    requestContext: CanonicalPlanRequestContext | None = None
    studentProfile: CanonicalPlanStudentProfile | None = None
    subject: CanonicalPlanSubject | None = None
    skillSnapshot: CanonicalSkillSnapshot | None = None
    planPreferences: CanonicalPlanPreferences | None = None


class TeacherPerformanceInsightSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    totalStudents: int = Field(default=0, ge=0)
    studentsWithData: int = Field(default=0, ge=0)
    supportCount: int = Field(default=0, ge=0)
    onTrackCount: int = Field(default=0, ge=0)
    averageScore: float = Field(default=0.0, ge=0, le=100)
    filterLabel: str | None = None
    strongestArea: str | None = None
    weakestArea: str | None = None


class TeacherPerformanceInsightMisconception(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    riskLevel: str = "watch"
    learnerCount: int = Field(default=0, ge=0)
    averageScore: float = Field(default=0.0, ge=0, le=100)


class TeacherPerformanceInsightHeatmapCell(BaseModel):
    model_config = ConfigDict(extra="ignore")

    studentId: str = Field(..., min_length=1)
    firstName: str = Field(..., min_length=1)
    lastName: str = Field(..., min_length=1)
    score: float | None = Field(default=None, ge=0, le=100)
    status: str | None = None
    performance: str | None = None
    engagement: str | None = None
    strength: str | None = None
    focusArea: str | None = None
    note: str | None = None


class TeacherPerformanceInsightMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"] = "user"
    text: str = Field(..., min_length=1)


class TeacherPerformanceInsightRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    teacherId: str | None = None
    subjectId: str | None = None
    subjectName: str | None = None
    currentView: Literal["subject", "topic", "assessment"] = "subject"
    filterLabel: str | None = None
    latestMessage: str = Field(..., min_length=1)
    summary: TeacherPerformanceInsightSummary | None = None
    misconceptions: list[TeacherPerformanceInsightMisconception] = Field(default_factory=list)
    heatmapCells: list[TeacherPerformanceInsightHeatmapCell] = Field(default_factory=list)
    messages: list[TeacherPerformanceInsightMessage] = Field(default_factory=list)


class TeacherPerformanceInsightResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reply: str = Field(..., min_length=1)
    suggestedNextAction: str | None = None
    focusStudents: list[str] = Field(default_factory=list)
    focusTopics: list[str] = Field(default_factory=list)


class GeneratedPlanSubskill(BaseModel):
    name: str = Field(..., min_length=1)
    score: int = Field(..., ge=0, le=100)
    color: Literal["yellow", "cyan", "blue", "green", "red"] = "blue"


class GeneratedPlanSkill(BaseModel):
    name: str = Field(..., min_length=1)
    score: int = Field(..., ge=0, le=100)
    subskills: list[GeneratedPlanSubskill] = Field(default_factory=list)


class GeneratedPlanStep(BaseModel):
    title: str = Field(..., min_length=1)
    type: Literal["video", "document", "assessment", "assignment", "quiz", "discussion"]
    content: str = ""
    link: str = ""
    additionalResources: list[str] = Field(default_factory=list)
    order: int = Field(..., ge=1)


class GeneratedDevelopmentPlan(BaseModel):
    id: str | None = None
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    progress: int = Field(default=0, ge=0, le=100)
    potentialOverall: int = Field(..., ge=0, le=100)
    eta: int = Field(..., ge=1)
    performance: str
    skills: list[GeneratedPlanSkill] = Field(default_factory=list)
    steps: list[GeneratedPlanStep] = Field(default_factory=list)
    subjectId: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None
    referenceFallbackUsed: bool = False
    criticalSkillsUsed: list[str] = Field(default_factory=list)


class StudentAssessmentCriterion(BaseModel):
    criterion: str = Field(..., min_length=1)
    score: float = Field(..., ge=0)
    feedback: str = Field(..., min_length=1)


class StudentAssessmentQuestionDetail(BaseModel):
    max_marks: float = Field(..., ge=0)
    awarded_marks: float = Field(..., ge=0)
    feedback: str = Field(..., min_length=1)
    improvement: str = ""


class StudentAssessmentData(BaseModel):
    is_correct_module: bool
    confidence_assessment_score: float = Field(..., ge=0, le=1)
    total_possible_marks: float = Field(..., ge=0)
    marks_achieved: float = Field(..., ge=0)
    marks_percentage: float = Field(..., ge=0, le=100)
    overall_feedback: str = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    criteria: list[StudentAssessmentCriterion] = Field(default_factory=list)
    assessment_details: dict[str, StudentAssessmentQuestionDetail] = Field(default_factory=dict)
    detected_module: str | None = None
    mark_consistency_check: str = Field(default="consistent", min_length=1)
    marking_scheme_used: bool = False


class StudentAssessmentResponse(BaseModel):
    module: str = Field(..., min_length=1)
    filename: str | None = None
    content_type: str | None = None
    ocr_type: str | None = None
    markdown: str = ""
    pages: int | None = None
    assessment: StudentAssessmentData
    file_id: str | None = None
    file_url: str | None = None
    view_url: str | None = None


class StudentTutorMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["student", "assistant"] = "student"
    text: str = Field(..., min_length=1)


class StudentMasteryTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topicId: str | None = None
    title: str = Field(..., min_length=1)
    masteryPercent: float = Field(default=0.0, ge=0, le=100)
    questionCount: int = Field(default=0, ge=0)
    priority: int | None = Field(default=None, ge=1)


class StudentTutorRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    studentId: str | None = None
    subjectId: str | None = None
    subjectName: str | None = None
    unitTitle: str | None = None
    topicTitle: str | None = None
    planTitle: str | None = None
    planStepTitle: str | None = None
    coachMode: Literal["socratic", "hint"] = "socratic"
    latestMessage: str = Field(..., min_length=1)
    taskGoal: str | None = None
    reasoningCanvas: str | None = None
    messages: list[StudentTutorMessage] = Field(default_factory=list)
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)
    masteryTopics: list[StudentMasteryTopic] = Field(default_factory=list)

    @field_validator("latestMessage", mode="before")
    @classmethod
    def _normalize_latest_message(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("latestMessage is required")
        return text


class StudentTutorResponse(BaseModel):
    reply: str = Field(..., min_length=1)
    suggestedNextAction: str | None = None
    followUpQuestion: str | None = None


class StudentChallengeQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=1)
    type: Literal["input", "single", "multiple"] = "input"
    prompt: str = Field(..., min_length=1)
    helpText: str | None = None
    options: list[str] = Field(default_factory=list)
    acceptedAnswers: list[str] = Field(default_factory=list)
    correctOptionIndexes: list[int] = Field(default_factory=list)


class StudentChallengeGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    studentId: str | None = None
    subjectId: str | None = None
    subjectName: str = Field(..., min_length=1)
    mode: Literal["topic_challenge", "subject_challenge"] = "topic_challenge"
    unitTitle: str | None = None
    topicTitle: str | None = None
    questionCount: int = Field(default=10, ge=1, le=40)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    objective: str | None = None
    referenceDocuments: list[ReferenceDocument] = Field(default_factory=list)
    masteryTopics: list[StudentMasteryTopic] = Field(default_factory=list)


class StudentChallengeGenerationResponse(BaseModel):
    challengeId: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    coachMessage: str = Field(..., min_length=1)
    focusTopics: list[str] = Field(default_factory=list)
    questions: list[StudentChallengeQuestion] = Field(default_factory=list)
