from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .schema import (
    AssessmentGenerationRequest,
    StudentChallengeGenerationRequest,
    StudentChallengeGenerationResponse,
    StudentChallengeQuestion,
    StudentMasteryTopic,
    StudentTutorRequest,
    StudentTutorResponse,
)
from .service import AssessmentGenerationError, generate_teacher_assessment_questions


_MAX_TUTOR_MESSAGE_LENGTH = 900
_MAX_TUTOR_HISTORY = 8
_MAX_REFERENCE_DOCS = 3
_MAX_REFERENCE_CHARS = 1400


def _collapse_whitespace(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _trim_text(text: str, limit: int) -> str:
    compact = _collapse_whitespace(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _collapse_whitespace(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


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


def _prepare_reference_documents(reference_documents: list[Any]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for document in reference_documents[:_MAX_REFERENCE_DOCS]:
        name = _collapse_whitespace(getattr(document, "documentName", None) or getattr(document, "name", None) or "Reference")
        markdown = _trim_text(_collapse_whitespace(getattr(document, "markdown", None) or ""), _MAX_REFERENCE_CHARS)
        if not markdown:
            continue
        prepared.append({"documentName": name or "Reference", "markdown": markdown})
    return prepared


def _sorted_mastery_topics(topics: list[StudentMasteryTopic]) -> list[StudentMasteryTopic]:
    return sorted(
        topics,
        key=lambda topic: (
            float(topic.priority if topic.priority is not None else 999),
            float(topic.masteryPercent),
            -int(topic.questionCount or 0),
            _collapse_whitespace(topic.title).lower(),
        ),
    )


def _fallback_student_tutor_response(request: StudentTutorRequest) -> StudentTutorResponse:
    focus = (
        _collapse_whitespace(request.planStepTitle)
        or _collapse_whitespace(request.topicTitle)
        or _collapse_whitespace(request.unitTitle)
        or _collapse_whitespace(request.subjectName)
        or "this topic"
    )
    weakest_topics = [topic.title for topic in _sorted_mastery_topics(request.masteryTopics)[:2] if _collapse_whitespace(topic.title)]
    weakness_note = (
        f" Pay extra attention to {', '.join(weakest_topics)} while you think it through."
        if weakest_topics
        else ""
    )
    if request.coachMode == "hint":
        return StudentTutorResponse(
            reply=(
                f"Start with the core idea behind {focus}. Break it into one definition and one example from your work.{weakness_note}"
            ),
            suggestedNextAction="Write one short explanation in your own words, then test it on a simple example.",
            followUpQuestion=f"What is the first principle or definition you should state before you continue with {focus}?",
        )
    return StudentTutorResponse(
        reply=(
            f"Before you answer, pause and explain the reasoning path for {focus} step by step.{weakness_note}"
        ),
        suggestedNextAction="Identify the key concept, the evidence you have, and the step that connects them.",
        followUpQuestion=f"Which part of {focus} are you most confident about, and which part still feels uncertain?",
    )


def _build_student_tutor_prompt(request: StudentTutorRequest) -> tuple[str, str]:
    subject_name = _collapse_whitespace(request.subjectName or "the current subject")
    system_text = (
        "You are a {subject_name} tutor helping a student think more clearly and independently.\n"
        "Return JSON ONLY. No markdown fences. No prose outside JSON.\n\n"
        "Rules:\n"
        "- Speak directly to the student using \"you\" and \"your\".\n"
        "- Default to ZIMSEC O Level high-school expectations unless the input explicitly requests another level.\n"
        "- Do not use tertiary or university-level wording, examples, or assumptions.\n"
        "- Use a coaching tone, not teacher administration language.\n"
        "- Do not say \"the student\".\n"
        "- Do not give away the full solution unless the student is clearly asking for a final check after showing work.\n"
        "- Prefer one concise explanation, one actionable next step, and one follow-up question.\n"
        "- When coachMode is socratic, guide with probing questions and reasoning prompts.\n"
        "- When coachMode is hint, give one small hint and then ask the student to continue.\n"
        "- Use the provided subject, topic, plan step, mastery topics, and reference material when they are useful.\n"
        "- Keep the response concise and focused.\n\n"
        "Return exactly this schema:\n"
        "{\n"
        "  \"reply\": \"...\",\n"
        "  \"suggestedNextAction\": \"...\",\n"
        "  \"followUpQuestion\": \"...\"\n"
        "}"
    ).replace("{subject_name}", subject_name)

    user_obj = {
        "studentId": request.studentId,
        "subjectId": request.subjectId,
        "subjectName": subject_name,
        "curriculumLevel": "ZIMSEC O Level",
        "unitTitle": _collapse_whitespace(request.unitTitle or ""),
        "topicTitle": _collapse_whitespace(request.topicTitle or ""),
        "planTitle": _collapse_whitespace(request.planTitle or ""),
        "planStepTitle": _collapse_whitespace(request.planStepTitle or ""),
        "coachMode": request.coachMode,
        "latestMessage": _trim_text(_collapse_whitespace(request.latestMessage), _MAX_TUTOR_MESSAGE_LENGTH),
        "taskGoal": _trim_text(_collapse_whitespace(request.taskGoal or ""), 220),
        "reasoningCanvas": _trim_text(_collapse_whitespace(request.reasoningCanvas or ""), 800),
        "messages": [
            {"role": message.role, "text": _trim_text(_collapse_whitespace(message.text), 280)}
            for message in request.messages[-_MAX_TUTOR_HISTORY:]
            if _collapse_whitespace(message.text)
        ],
        "referenceDocuments": _prepare_reference_documents(request.referenceDocuments),
        "masteryTopics": [
            {
                "title": _collapse_whitespace(topic.title),
                "masteryPercent": round(float(topic.masteryPercent), 1),
                "questionCount": int(topic.questionCount or 0),
                "priority": topic.priority,
            }
            for topic in _sorted_mastery_topics(request.masteryTopics)[:5]
            if _collapse_whitespace(topic.title)
        ],
    }
    return system_text, json.dumps(user_obj, ensure_ascii=False)


def generate_student_tutor_response(request: StudentTutorRequest, *, llm_client) -> StudentTutorResponse:
    system_text, user_text = _build_student_tutor_prompt(request)
    try:
        raw = llm_client.generate(system_text, user_text, max_new_tokens=320)
        candidate = _parse_json_candidate(raw)
        if not isinstance(candidate, dict):
            raise ValueError("Student tutor response is not an object")
        reply = _collapse_whitespace(str(candidate.get("reply") or candidate.get("overall_feedback") or ""))
        if not reply:
            raise ValueError("Student tutor reply is empty")
        suggested_next_action = _collapse_whitespace(str(candidate.get("suggestedNextAction") or "")) or None
        follow_up_question = _collapse_whitespace(str(candidate.get("followUpQuestion") or "")) or None
        return StudentTutorResponse(
            reply=reply,
            suggestedNextAction=suggested_next_action,
            followUpQuestion=follow_up_question,
        )
    except Exception as exc:
        print(f"[student-tutor] using fallback: {exc}; preview={_preview(locals().get('raw'))}")
        return _fallback_student_tutor_response(request)


def _fallback_challenge_questions(request: StudentChallengeGenerationRequest, focus_topics: list[str]) -> list[StudentChallengeQuestion]:
    target_topics = focus_topics or [_collapse_whitespace(request.topicTitle or request.unitTitle or request.subjectName)]
    questions: list[StudentChallengeQuestion] = []
    for index in range(request.questionCount):
        topic = target_topics[index % len(target_topics)] if target_topics else _collapse_whitespace(request.subjectName)
        correct_option = f"It correctly applies {topic} in context"
        options = [
            correct_option,
            f"It ignores the main idea of {topic}",
            f"It treats {topic} as unrelated terminology",
            f"It guesses without checking the reasoning",
        ]
        questions.append(
            StudentChallengeQuestion(
                id=f"challenge-question-{index + 1}",
                type="single",
                prompt=f"Which response best demonstrates sound reasoning about {topic}?",
                options=options,
                correctOptionIndexes=[0],
                acceptedAnswers=[],
                helpText=f"Focus on the option that shows correct understanding of {topic}.",
            )
        )
    return questions


def _map_generated_question(raw_question: Any, index: int) -> StudentChallengeQuestion | None:
    prompt = _collapse_whitespace(str(getattr(raw_question, "text", None) or ""))
    if not prompt:
        return None
    raw_type = _collapse_whitespace(str(getattr(raw_question, "type", "multiple_choice"))).lower()
    options = [_collapse_whitespace(str(option)) for option in (getattr(raw_question, "options", None) or []) if _collapse_whitespace(str(option))]
    if raw_type == "true_false" and not options:
        options = ["True", "False"]
    correct_answer = _collapse_whitespace(str(getattr(raw_question, "correctAnswer", None) or ""))
    correct_indexes = [idx for idx, option in enumerate(options) if option == correct_answer]
    if not correct_indexes and options:
        correct_indexes = [0]
    return StudentChallengeQuestion(
        id=f"challenge-question-{index}",
        type="single",
        prompt=prompt,
        helpText=_collapse_whitespace(str(getattr(raw_question, "explanation", None) or "")) or None,
        options=options,
        acceptedAnswers=[],
        correctOptionIndexes=correct_indexes,
    )


def generate_student_challenge(
    request: StudentChallengeGenerationRequest,
    *,
    llm_client,
) -> StudentChallengeGenerationResponse:
    sorted_topics = _sorted_mastery_topics(request.masteryTopics)
    focus_topics = _unique_strings([topic.title for topic in sorted_topics[:3]])
    if not focus_topics:
        fallback_topic = _collapse_whitespace(request.topicTitle or request.unitTitle or request.subjectName)
        focus_topics = [fallback_topic] if fallback_topic else ["core subject skills"]

    objective = _collapse_whitespace(request.objective or "") or (
        f"Strengthen understanding in {', '.join(focus_topics)} through a targeted challenge."
    )
    context_parts = [
        f"Generate a student-facing {request.mode.replace('_', ' ')} for {request.subjectName}.",
        "Keep every question within ZIMSEC O Level high-school level unless the input explicitly requests another level.",
        "Do not generate tertiary, university, or advanced specialist content.",
        f"Target {request.questionCount} multiple-choice questions.",
        f"Focus areas: {', '.join(focus_topics)}.",
        f"Objective: {objective}",
        "Questions should build reasoning and conceptual understanding, not rote recall.",
        "Make distractors plausible and based on common misconceptions.",
    ]
    if request.unitTitle:
        context_parts.append(f"Unit: {request.unitTitle}.")
    if request.topicTitle:
        context_parts.append(f"Selected topic: {request.topicTitle}.")
    if sorted_topics:
        topic_snapshot = "; ".join(
            f"{topic.title} ({round(float(topic.masteryPercent), 1)}% mastery, {int(topic.questionCount or 0)} questions)"
            for topic in sorted_topics[:5]
        )
        context_parts.append(f"Mastery snapshot: {topic_snapshot}.")

    assessment_request = AssessmentGenerationRequest(
        context="\n".join(part for part in context_parts if _collapse_whitespace(part)),
        difficulty=request.difficulty,
        questionTypes="multiple_choice",
        numberOfQuestions=request.questionCount,
        attributes={
            request.subjectName: objective,
            (request.topicTitle or request.unitTitle or focus_topics[0]): ", ".join(focus_topics),
        },
        referenceDocuments=request.referenceDocuments,
        tags=_unique_strings([request.subjectName, *focus_topics]),
    )

    try:
        generated = generate_teacher_assessment_questions(assessment_request, llm_client=llm_client)
        questions = [
            mapped
            for index, item in enumerate(generated, start=1)
            if (mapped := _map_generated_question(item, index)) is not None
        ]
    except (AssessmentGenerationError, Exception) as exc:
        print(f"[student-challenge] using fallback questions: {exc}")
        questions = []

    if not questions:
        questions = _fallback_challenge_questions(request, focus_topics)

    challenge_title_target = _collapse_whitespace(request.topicTitle or request.unitTitle or request.subjectName)
    title_prefix = "Topic challenge" if request.mode == "topic_challenge" else "Subject challenge"
    return StudentChallengeGenerationResponse(
        challengeId=str(uuid.uuid4()),
        title=f"{title_prefix}: {challenge_title_target}",
        summary=f"Generated {len(questions)} personalized challenge questions focused on {', '.join(focus_topics[:3])}.",
        coachMessage=(
            f"Work through these questions slowly. Your main focus is {', '.join(focus_topics[:2])}. Explain why each answer is right before you move on."
        ),
        focusTopics=focus_topics,
        questions=questions,
    )
