from __future__ import annotations

import json
import os
import re
import html
import math
from typing import Any, Iterable

from .schema import (
    AssessmentGenerationRequest,
    GeneratedDevelopmentPlan,
    GeneratedAssessmentQuestion,
    GeneratedMarkingGuide,
    GeneratedPlanSkill,
    GeneratedPlanStep,
    GeneratedPlanSubskill,
    GeneratedTeacherPractice,
    GeneratedTeacherPracticeQuestion,
    GeneratedTeacherResource,
    GeneratedRubricItem,
    PlanGenerationRequest,
    ReferenceDocument,
    TeacherPracticeGenerationRequest,
    TeacherResourceGenerationRequest,
)


MAX_REFERENCE_DOCUMENTS = 3
MAX_REFERENCE_TOTAL_CHARS = 4000
MAX_REFERENCE_PER_DOC_CHARS = 1600
MAX_CONTEXT_CHARS = 1800
_QUESTION_COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "another": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


class AssessmentGenerationError(Exception):
    def __init__(self, message: str, *, first_raw: str | None = None, retry_raw: str | None = None):
        super().__init__(message)
        self.first_raw = first_raw
        self.retry_raw = retry_raw


class ReferencePreparationResult:
    def __init__(self, usable_documents: list[dict[str, str]], fallback_used: bool):
        self.usable_documents = usable_documents
        self.fallback_used = fallback_used


def generate_teacher_assessment_questions(
    request: AssessmentGenerationRequest,
    *,
    llm_client,
) -> list[GeneratedAssessmentQuestion]:
    prepared_docs = _prepare_reference_documents(request.referenceDocuments)
    system_text, user_text = _build_assessment_generation_prompt(request, prepared_docs)
    max_new_tokens = _assessment_generation_max_tokens(request.numberOfQuestions)

    raw = llm_client.generate(system_text, user_text, max_new_tokens=max_new_tokens)
    try:
        raw_questions = _parse_questions_payload(raw)
    except Exception as exc:
        print(f"[assessment-generation] first pass invalid output: {exc}; preview={_preview(raw)}")
        retry_system = (
            system_text
            + "\n\nReturn ONLY valid JSON with this exact top-level schema: {\"questions\": [...]}"
        )
        raw_retry = llm_client.generate(retry_system, user_text, max_new_tokens=max_new_tokens)
        try:
            raw_questions = _parse_questions_payload(raw_retry)
            raw = raw_retry
        except Exception as retry_exc:
            print(
                f"[assessment-generation] retry invalid output: {retry_exc}; preview={_preview(raw_retry)}"
            )
            raise AssessmentGenerationError(
                "Model returned invalid assessment generation JSON after retry.",
                first_raw=raw,
                retry_raw=raw_retry,
            ) from retry_exc

    normalized = _normalize_generated_questions(
        raw_questions,
        request=request,
        prepared_docs=prepared_docs,
    )
    if not normalized:
        raise AssessmentGenerationError(
            "Model returned no usable generated questions.",
            first_raw=raw,
        )
    return normalized


def _assessment_generation_max_tokens(number_of_questions: int) -> int:
    configured = os.getenv("ASSESSMENT_GENERATION_MAX_NEW_TOKENS")
    if configured:
        try:
            return max(128, int(configured))
        except ValueError:
            pass
    return min(768, max(256, number_of_questions * 110))


def _build_assessment_generation_prompt(
    request: AssessmentGenerationRequest,
    prepared_docs: ReferencePreparationResult,
) -> tuple[str, str]:
    subject_name = _resolve_assessment_subject_name(request)
    system_text = (
        f"You are an assessment generation assistant for teachers of {subject_name}.\n"
        + """Generate classroom-ready assessment questions and their marking guides.
Return JSON ONLY. No markdown. No prose outside JSON.

Rules:
- Produce exactly the requested number of questions.
- Default to ZIMSEC O Level high-school expectations unless the input explicitly requests another level.
- Do not generate tertiary, university, or advanced specialist content.
- Supported output question types are: multiple_choice, true_false, short_answer, essay.
- multiple_choice questions must have 4 plausible options and correctAnswer must exactly match one option text.
- true_false questions must use options [\"True\", \"False\"] and correctAnswer must be exactly one of them.
- short_answer and essay questions must include a concise expectedAnswer and rubricItems.
- Every question must include an explanation that works as a teacher-facing marking guide summary.
- If reference documents are provided and useful, use them.
- If reference documents are missing, empty, incomplete, or not useful, silently fall back to the provided context, attributes, tags, and your own general knowledge of {subject_name}.
- Never refuse generation because documents are missing.
- Keep output concise and practical for classroom use.
- Keep every question anchored to {subject_name}. Do not drift into another subject.

Return exactly this schema:
{
  \"questions\": [
    {
      \"text\": \"...\",
      \"type\": \"multiple_choice\",
      \"options\": [\"...\", \"...\", \"...\", \"...\"],
      \"correctAnswer\": \"exact option text\",
      \"explanation\": \"teacher-facing marking guide summary\",
      \"difficulty\": \"medium\",
      \"tags\": [\"Key concept\"],
      \"points\": 1,
      \"expectedAnswer\": \"...\",
      \"rubricItems\": [
        {\"description\": \"...\", \"marks\": 1, \"keywords\": [\"...\"]}
      ],
      \"sourceDocumentsUsed\": [\"doc-name\"]
    }
  ]
}"""
    ).replace("{subject_name}", subject_name)

    user_obj = {
        "task": "Generate assessment questions with marking guides.",
        "subject_name": subject_name,
        "question_type_mode": request.questionTypes,
        "question_type_plan": _build_question_type_plan(request.questionTypes, request.numberOfQuestions),
        "number_of_questions": request.numberOfQuestions,
        "difficulty": request.difficulty,
        "context": _trim_text(request.context, MAX_CONTEXT_CHARS),
        "attributes": _normalize_attributes_for_prompt(request.attributes),
        "tags": request.tags,
        "reference_documents": prepared_docs.usable_documents,
        "reference_fallback_used": prepared_docs.fallback_used,
    }
    return system_text, json.dumps(user_obj, ensure_ascii=False)


def _prepare_reference_documents(reference_documents: list[ReferenceDocument]) -> ReferencePreparationResult:
    usable_documents: list[dict[str, str]] = []
    remaining = MAX_REFERENCE_TOTAL_CHARS

    for document in reference_documents[:MAX_REFERENCE_DOCUMENTS]:
        text = _collapse_whitespace(document.markdown)
        if not text:
            continue
        excerpt_limit = min(MAX_REFERENCE_PER_DOC_CHARS, remaining)
        if excerpt_limit <= 0:
            break
        excerpt = _trim_text(text, excerpt_limit)
        if not excerpt:
            continue
        usable_documents.append({
            "documentName": document.documentName,
            "excerpt": excerpt,
        })
        remaining -= len(excerpt)
        if remaining <= 0:
            break

    return ReferencePreparationResult(
        usable_documents=usable_documents,
        fallback_used=len(usable_documents) == 0,
    )


def _normalize_attributes_for_prompt(attributes: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in attributes.items():
        name = _collapse_whitespace(str(key))
        if not name:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = _collapse_whitespace(str(value))
        normalized[name] = rendered or name
    return normalized


def _resolve_assessment_subject_name(request: AssessmentGenerationRequest) -> str:
    explicit = _collapse_whitespace(str(request.subjectName or ""))
    if explicit:
        return explicit
    attribute_names = [
        _collapse_whitespace(str(name))
        for name in request.attributes.keys()
        if _collapse_whitespace(str(name))
    ]
    lowered = {name.lower(): name for name in attribute_names}
    for candidate in ("english language", "mathematics", "computer science", "biology", "chemistry", "physics", "history", "geography"):
        if candidate in lowered:
            return lowered[candidate]
    tag_names = [_collapse_whitespace(str(tag)) for tag in request.tags if _collapse_whitespace(str(tag))]
    for tag in tag_names:
        if " " in tag or tag.lower() in lowered:
            return tag
    return "the subject"


def _build_question_type_plan(question_type_mode: str, count: int) -> list[str]:
    if question_type_mode == "multiple_choice":
        return ["multiple_choice"] * count
    if question_type_mode == "structured":
        return ["short_answer"] * count

    plan: list[str] = []
    for idx in range(count):
        if idx % 3 == 1 and count > 2:
            plan.append("short_answer")
        elif idx % 3 == 2 and count > 3:
            plan.append("true_false")
        else:
            plan.append("multiple_choice" if idx % 2 == 0 else "short_answer")
    return plan[:count]


def _parse_questions_payload(raw_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[Any] = []

    for index, char in enumerate(raw_text):
        if char not in "[{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(obj)

    if not candidates:
        raise ValueError("Model output did not contain JSON")

    for candidate in reversed(candidates):
        if isinstance(candidate, dict) and isinstance(candidate.get("questions"), list):
            return [item for item in candidate["questions"] if isinstance(item, dict)]
    for candidate in reversed(candidates):
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

    raise ValueError("Model output did not contain a questions array")


def _normalize_generated_questions(
    raw_questions: list[dict[str, Any]],
    *,
    request: AssessmentGenerationRequest,
    prepared_docs: ReferencePreparationResult,
) -> list[GeneratedAssessmentQuestion]:
    normalized_questions: list[GeneratedAssessmentQuestion] = []
    requested_tags = _unique_strings(request.tags)
    attribute_tags = _unique_strings(request.attributes.keys())
    type_plan = _build_question_type_plan(request.questionTypes, request.numberOfQuestions)
    source_doc_names = [doc["documentName"] for doc in prepared_docs.usable_documents]

    for index, raw_question in enumerate(raw_questions[: request.numberOfQuestions], start=1):
        question_type = _normalize_question_type(raw_question.get("type"), fallback_type=type_plan[index - 1])
        text = _collapse_whitespace(str(raw_question.get("text") or raw_question.get("question") or ""))
        if not text:
            continue

        difficulty = _normalize_difficulty(raw_question.get("difficulty"), request.difficulty)
        points = _normalize_points(raw_question.get("points"), question_type)
        options = _normalize_options(raw_question.get("options"), question_type)
        correct_answer = _normalize_correct_answer(raw_question.get("correctAnswer"), options, question_type)
        correct_answers = [correct_answer] if correct_answer else []
        expected_answer = _collapse_whitespace(str(raw_question.get("expectedAnswer") or correct_answer or "")) or None
        rubric_items = _normalize_rubric_items(
            raw_question.get("rubricItems"),
            question_type=question_type,
            points=points,
            expected_answer=expected_answer,
            explanation=_collapse_whitespace(str(raw_question.get("explanation") or "")),
        )
        explanation = _build_explanation(
            question_type=question_type,
            explanation=_collapse_whitespace(str(raw_question.get("explanation") or "")),
            correct_answer=correct_answer,
            expected_answer=expected_answer,
            rubric_items=rubric_items,
        )
        tags = _unique_strings(list(raw_question.get("tags") or []) + requested_tags + attribute_tags)
        source_documents_used = _normalize_source_documents(raw_question.get("sourceDocumentsUsed"), source_doc_names)
        marking_guide = _build_marking_guide(question_type, expected_answer, rubric_items)
        rubric_json = {
            "correctAnswer": correct_answer,
            "correctAnswers": correct_answers,
            "expectedAnswer": expected_answer,
            "markingGuide": explanation,
            "rubricItems": [item.model_dump() for item in rubric_items],
        }

        normalized_questions.append(
            GeneratedAssessmentQuestion(
                text=text,
                type=question_type,
                options=options,
                correctAnswer=correct_answer,
                correctAnswers=correct_answers,
                explanation=explanation,
                difficulty=difficulty,
                tags=tags,
                points=points,
                maxMarks=points,
                markingGuide=marking_guide,
                rubricJson=rubric_json,
                referenceFallbackUsed=prepared_docs.fallback_used,
                sourceDocumentsUsed=source_documents_used,
            )
        )

    while len(normalized_questions) < request.numberOfQuestions:
        index = len(normalized_questions) + 1
        fallback_type = type_plan[index - 1]
        normalized_questions.append(
            _build_fallback_question(
                index=index,
                fallback_type=fallback_type,
                request=request,
                prepared_docs=prepared_docs,
            )
        )

    return normalized_questions[: request.numberOfQuestions]


def _normalize_question_type(value: Any, *, fallback_type: str) -> str:
    normalized = _collapse_whitespace(str(value or "")).lower().replace("-", "_").replace(" ", "_")
    if normalized in {"multiple_choice", "mcq"}:
        return "multiple_choice"
    if normalized in {"true_false", "boolean"}:
        return "true_false"
    if normalized in {"short_answer", "structured", "short"}:
        return "short_answer"
    if normalized == "essay":
        return "essay"
    return fallback_type


def _normalize_difficulty(value: Any, fallback: str) -> str:
    normalized = _collapse_whitespace(str(value or "")).lower()
    if normalized in {"easy", "medium", "hard"}:
        return normalized
    return fallback


def _normalize_points(value: Any, question_type: str) -> int:
    default_points = {
        "multiple_choice": 1,
        "true_false": 1,
        "short_answer": 4,
        "essay": 6,
    }.get(question_type, 1)
    try:
        points = int(round(float(value)))
        return max(1, points)
    except (TypeError, ValueError):
        return default_points


def _normalize_options(value: Any, question_type: str) -> list[str]:
    if question_type == "true_false":
        return ["True", "False"]
    if question_type not in {"multiple_choice"}:
        return []

    options: list[str] = []
    if isinstance(value, list):
        options = [_collapse_whitespace(str(item)) for item in value]
    elif isinstance(value, str):
        options = [_collapse_whitespace(item) for item in re.split(r"\n|;|\|", value)]

    options = [item for item in options if item]
    options = _dedupe_preserve_order(options)
    if len(options) >= 4:
        return options[:4]

    filler = [
        "All of the above",
        "None of the above",
        "Both A and B",
        "It depends on the implementation",
    ]
    for option in filler:
        if len(options) >= 4:
            break
        if option not in options:
            options.append(option)
    while len(options) < 4:
        options.append(f"Option {len(options) + 1}")
    return options[:4]


def _normalize_correct_answer(value: Any, options: list[str], question_type: str) -> str | None:
    if question_type == "true_false":
        normalized = _collapse_whitespace(str(value or "")).lower()
        if normalized in {"true", "t", "1", "yes"}:
            return "True"
        if normalized in {"false", "f", "0", "no"}:
            return "False"
        return "True"

    if question_type != "multiple_choice":
        text = _collapse_whitespace(str(value or ""))
        return text or None

    raw = _collapse_whitespace(str(value or ""))
    if not raw:
        return options[0] if options else None

    for option in options:
        if raw.lower() == option.lower():
            return option

    normalized = raw.upper().rstrip(").:")
    labels = ["A", "B", "C", "D", "E", "F"]
    if normalized in labels:
        idx = labels.index(normalized)
        if idx < len(options):
            return options[idx]
    if normalized.startswith("OPTION ") and normalized.split()[-1] in labels:
        idx = labels.index(normalized.split()[-1])
        if idx < len(options):
            return options[idx]
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]

    return options[0] if options else raw


def _normalize_rubric_items(
    value: Any,
    *,
    question_type: str,
    points: int,
    expected_answer: str | None,
    explanation: str,
) -> list[GeneratedRubricItem]:
    if question_type in {"multiple_choice", "true_false"}:
        return []

    items: list[GeneratedRubricItem] = []
    if isinstance(value, list):
        for index, raw_item in enumerate(value, start=1):
            if not isinstance(raw_item, dict):
                continue
            description = _collapse_whitespace(str(raw_item.get("description") or raw_item.get("point") or ""))
            if not description:
                continue
            marks = _normalize_points(raw_item.get("marks"), "short_answer")
            keywords = _unique_strings(raw_item.get("keywords") or [])
            items.append(
                GeneratedRubricItem(
                    index=index,
                    description=description,
                    marks=marks,
                    keywords=keywords,
                )
            )

    if not items:
        seed_text = expected_answer or explanation or "Accurate relevant answer"
        segments = _split_key_points(seed_text)
        if not segments:
            segments = ["Accurate relevant answer"]
        allocated_marks = _allocate_marks(points, len(segments))
        items = [
            GeneratedRubricItem(
                index=index,
                description=segment,
                marks=allocated_marks[index - 1],
                keywords=_extract_keywords(segment),
            )
            for index, segment in enumerate(segments, start=1)
        ]

    return items


def _build_explanation(
    *,
    question_type: str,
    explanation: str,
    correct_answer: str | None,
    expected_answer: str | None,
    rubric_items: list[GeneratedRubricItem],
) -> str:
    if explanation:
        return explanation
    if question_type in {"multiple_choice", "true_false"} and correct_answer:
        return f"Correct answer: {correct_answer}."
    if expected_answer:
        return f"Expected answer: {expected_answer}."
    if rubric_items:
        joined = "; ".join(item.description for item in rubric_items)
        return f"Award marks for: {joined}."
    return "Use the marking guide to award marks for accurate relevant points."


def _build_marking_guide(
    question_type: str,
    expected_answer: str | None,
    rubric_items: list[GeneratedRubricItem],
) -> GeneratedMarkingGuide:
    if question_type in {"multiple_choice", "true_false"}:
        return GeneratedMarkingGuide(mode="objective", expectedAnswer=expected_answer, rubricItems=[])
    if rubric_items:
        return GeneratedMarkingGuide(mode="rubric", expectedAnswer=expected_answer, rubricItems=rubric_items)
    return GeneratedMarkingGuide(mode="holistic", expectedAnswer=expected_answer, rubricItems=[])


def _normalize_source_documents(value: Any, fallback_names: list[str]) -> list[str]:
    if isinstance(value, list):
        names = _unique_strings(value)
        return names or fallback_names
    return fallback_names


def _build_fallback_question(
    *,
    index: int,
    fallback_type: str,
    request: AssessmentGenerationRequest,
    prepared_docs: ReferencePreparationResult,
) -> GeneratedAssessmentQuestion:
    topic_hint = ", ".join(_normalize_attributes_for_prompt(request.attributes).keys()) or "the requested topic"
    base_text = f"Question {index}: Write about {topic_hint}."
    if fallback_type == "multiple_choice":
        options = [
            f"A correct statement about {topic_hint}",
            f"A partially correct statement about {topic_hint}",
            f"An incorrect statement about {topic_hint}",
            f"An unrelated statement about {topic_hint}",
        ]
        correct_answer = options[0]
        explanation = f"Correct answer: {correct_answer}."
        marking_guide = GeneratedMarkingGuide(mode="objective", expectedAnswer=correct_answer, rubricItems=[])
        rubric_json = {
            "correctAnswer": correct_answer,
            "correctAnswers": [correct_answer],
            "expectedAnswer": correct_answer,
            "markingGuide": explanation,
            "rubricItems": [],
        }
        return GeneratedAssessmentQuestion(
            text=base_text,
            type="multiple_choice",
            options=options,
            correctAnswer=correct_answer,
            correctAnswers=[correct_answer],
            explanation=explanation,
            difficulty=request.difficulty,
            tags=_unique_strings(request.tags + list(request.attributes.keys())),
            points=1,
            maxMarks=1,
            markingGuide=marking_guide,
            rubricJson=rubric_json,
            referenceFallbackUsed=prepared_docs.fallback_used,
            sourceDocumentsUsed=[doc["documentName"] for doc in prepared_docs.usable_documents],
        )

    expected_answer = f"A correct response should address {topic_hint} accurately and concisely."
    rubric_items = _normalize_rubric_items(
        None,
        question_type=fallback_type,
        points=4,
        expected_answer=expected_answer,
        explanation="",
    )
    explanation = _build_explanation(
        question_type=fallback_type,
        explanation="",
        correct_answer=None,
        expected_answer=expected_answer,
        rubric_items=rubric_items,
    )
    marking_guide = _build_marking_guide(fallback_type, expected_answer, rubric_items)
    rubric_json = {
        "correctAnswer": None,
        "correctAnswers": [],
        "expectedAnswer": expected_answer,
        "markingGuide": explanation,
        "rubricItems": [item.model_dump() for item in rubric_items],
    }
    return GeneratedAssessmentQuestion(
        text=base_text,
        type=fallback_type if fallback_type in {"short_answer", "essay"} else "short_answer",
        options=[],
        correctAnswer=None,
        correctAnswers=[],
        explanation=explanation,
        difficulty=request.difficulty,
        tags=_unique_strings(request.tags + list(request.attributes.keys())),
        points=4,
        maxMarks=4,
        markingGuide=marking_guide,
        rubricJson=rubric_json,
        referenceFallbackUsed=prepared_docs.fallback_used,
        sourceDocumentsUsed=[doc["documentName"] for doc in prepared_docs.usable_documents],
    )


def _split_key_points(text: str) -> list[str]:
    cleaned = text.replace("\n", "; ")
    segments = [
        _collapse_whitespace(segment)
        for segment in re.split(r";|\.|\n", cleaned)
    ]
    segments = [segment for segment in segments if segment]
    if not segments:
        return []
    return segments[:3]


def _allocate_marks(total_marks: int, item_count: int) -> list[int]:
    if item_count <= 0:
        return []
    base = max(1, total_marks // item_count)
    marks = [base] * item_count
    remainder = max(0, total_marks - (base * item_count))
    index = 0
    while remainder > 0:
        marks[index % item_count] += 1
        remainder -= 1
        index += 1
    return marks


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)
    keywords: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in {"the", "and", "for", "with", "that", "this", "from", "into", "only", "must"}:
            continue
        if normalized not in keywords:
            keywords.append(normalized)
        if len(keywords) >= 4:
            break
    return keywords


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return _dedupe_preserve_order(_collapse_whitespace(str(value)) for value in values if _collapse_whitespace(str(value)))


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _trim_text(value: str, max_chars: int) -> str:
    collapsed = _collapse_whitespace(value)
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _preview(value: str, limit: int = 320) -> str:
    return _collapse_whitespace(value)[:limit]


def _parse_json_candidate(raw_text: str) -> Any:
    decoder = json.JSONDecoder()
    candidates: list[Any] = []

    for index, char in enumerate(raw_text):
        if char not in "[{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(obj)

    if not candidates:
        raise ValueError("Model output did not contain JSON")
    return candidates[-1]


def _sanitize_generated_html(raw_html: str) -> str:
    html_value = _collapse_whitespace(raw_html).replace("</p><p>", "</p>\n<p>")
    html_value = re.sub(r"</?(?:html|body|script|style)[^>]*>", "", html_value, flags=re.IGNORECASE)
    return html_value.strip()


def _coerce_candidate_html(candidate: dict[str, Any]) -> str:
    for key in ("contentHtml", "content", "html", "body", "draftHtml", "draft", "resourceHtml"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_generated_html(value)
        if isinstance(value, list):
            parts = [_collapse_whitespace(str(item)) for item in value if _collapse_whitespace(str(item))]
            if parts:
                return _sanitize_generated_html("".join(f"<p>{html.escape(part)}</p>" for part in parts))

    sections = candidate.get("sections")
    if isinstance(sections, list):
        blocks: list[str] = []
        for section in sections:
            if isinstance(section, dict):
                heading = _collapse_whitespace(str(section.get("heading") or section.get("title") or ""))
                body = _collapse_whitespace(str(section.get("content") or section.get("text") or ""))
                if heading:
                    blocks.append(f"<h2>{html.escape(heading)}</h2>")
                if body:
                    blocks.append(f"<p>{html.escape(body)}</p>")
            else:
                text = _collapse_whitespace(str(section))
                if text:
                    blocks.append(f"<p>{html.escape(text)}</p>")
        if blocks:
            return _sanitize_generated_html("".join(blocks))

    return ""


def _coerce_candidate_text(candidate: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, str):
            text = _collapse_whitespace(value)
            if text:
                return text
    return ""


_GENERIC_FOCUS_SUFFIXES = (
    " resource",
    " resources",
    " draft",
    " content",
    " notes",
    " lesson",
    " lesson notes",
    " practice",
    " quiz",
    " test",
    " exam",
    " assignment",
    " worksheet",
)


def _strip_generic_focus_suffixes(value: str) -> str:
    text = _collapse_whitespace(value)
    if not text:
        return ""
    previous = None
    while previous != text:
        previous = text
        lowered = text.lower()
        for suffix in _GENERIC_FOCUS_SUFFIXES:
            if lowered.endswith(suffix):
                text = text[: -len(suffix)].strip(" -:|")
                break
    return _collapse_whitespace(text)


def _resolve_generation_focus(topic_title: str, title: str | None) -> str:
    topic = _collapse_whitespace(topic_title)
    title_text = _strip_generic_focus_suffixes(title or "")
    if not title_text:
        return topic
    if title_text.lower() == topic.lower():
        return topic
    return title_text


def _fallback_resource_html(
    request: TeacherResourceGenerationRequest,
    source_doc_names: list[str],
    fallback_used: bool,
) -> GeneratedTeacherResource:
    focus_area = _resolve_generation_focus(request.topicTitle, request.title)
    title = _collapse_whitespace(request.title or f"{focus_area} {request.contentType.title()}").strip()
    objective = _collapse_whitespace(request.objective or f"Build mastery in {focus_area}")
    teacher_prompt = _collapse_whitespace(request.teacherPrompt or "Generate a clear, classroom-ready draft.")
    subject_name = _collapse_whitespace(request.subjectName or "the subject")
    grade_level = _collapse_whitespace(request.gradeLevel or "Form 4")
    focus_line = (
        f"<p><strong>Subtopic focus:</strong> {html.escape(focus_area)}</p>"
        if focus_area.lower() != _collapse_whitespace(request.topicTitle).lower()
        else ""
    )
    variant_note = (
        "<p><strong>Variation:</strong> This version uses a slightly different explanation flow and learner task structure.</p>"
        if request.variant
        else ""
    )
    content_html = f"""
<h1>{html.escape(title)}</h1>
<p><strong>Subject:</strong> {html.escape(subject_name)} | <strong>Grade level:</strong> {html.escape(grade_level)}</p>
<p><strong>Topic:</strong> {html.escape(request.topicTitle)}</p>
{focus_line}
<p><strong>Learning objective:</strong> {html.escape(objective)}</p>
{variant_note}
<h2>Overview</h2>
<p>This resource focuses on <strong>{html.escape(focus_area)}</strong>{'' if focus_area.lower() == _collapse_whitespace(request.topicTitle).lower() else f' within {html.escape(request.topicTitle)}'} using clear classroom language and direct teaching points that the teacher can adapt during instruction.</p>
<h2>Teach It</h2>
<p>Start by defining the key idea in <strong>{html.escape(focus_area)}</strong>, then connect it to one practical example drawn from {html.escape(subject_name)}. Keep the explanation concise and check understanding after each section.</p>
<ul>
  <li>Explain the main concept in one clear paragraph.</li>
  <li>Use one worked example linked to {html.escape(focus_area)}.</li>
  <li>Highlight one common misconception learners may have.</li>
</ul>
<h2>Learner Check</h2>
<ol>
  <li>Ask learners to restate the concept in their own words.</li>
  <li>Give a short application prompt based on the topic.</li>
  <li>Review answers and correct misunderstandings immediately.</li>
</ol>
<h2>Teacher Notes</h2>
<p>Use the collaborator direction below to refine or extend the draft:</p>
<blockquote>{html.escape(teacher_prompt)}</blockquote>
""".strip()
    return GeneratedTeacherResource(
        title=title,
        contentHtml=_sanitize_generated_html(content_html),
        summary=f"Generated a {request.contentType} draft focused on {focus_area}.",
        teacherMessage=(
            "I generated a resource draft you can edit, expand, or publish."
            if not fallback_used
            else "I generated a fallback resource draft using the topic context because no strong reference source was available."
        ),
        sourceDocumentsUsed=source_doc_names,
        referenceFallbackUsed=fallback_used,
    )


def generate_teacher_resource(
    request: TeacherResourceGenerationRequest,
    *,
    llm_client,
) -> GeneratedTeacherResource:
    prepared_docs = _prepare_reference_documents(request.referenceDocuments)
    source_doc_names = [doc["documentName"] for doc in prepared_docs.usable_documents]
    fallback = _fallback_resource_html(request, source_doc_names, prepared_docs.fallback_used)
    focus_area = _resolve_generation_focus(request.topicTitle, request.title)

    subject_name = _collapse_whitespace(request.subjectName or "the requested subject")
    system_text = (
        f"You are an AI assistant that generates teacher workspace resources for {subject_name}.\n"
        + """Return JSON ONLY. No markdown fences. No prose outside JSON.

Rules:
- contentHtml must contain valid classroom-ready HTML only.
- Use concise semantic tags such as h1, h2, h3, p, ul, ol, li, table, thead, tbody, tr, th, td, blockquote, pre, code, strong, em.
- Do not include script, style, html, or body tags.
- Default to ZIMSEC O Level high-school expectations unless the input explicitly requests another level.
- Do not generate tertiary, university, or advanced specialist content.
- The content should be ready to render directly in a rich text editor.
- Treat topic_title as the umbrella topic.
- If focus_area or title is narrower than topic_title, center the content on that narrower subtopic while staying under the umbrella topic.
- If teacher_prompt names a narrower subtopic than topic_title, follow that narrower focus.
- summary must be one short teacher-facing sentence.
- teacherMessage must be one short teacher-facing completion sentence.
- If reference documents are missing or weak, silently fall back to the topic, objective, existing content, related records, and your general teaching knowledge of {subject_name}.
- Keep the content anchored to {subject_name}. Do not drift into another subject.

Return exactly this schema:
{
  "title": "...",
  "contentHtml": "<h1>...</h1><p>...</p>",
  "summary": "...",
  "teacherMessage": "...",
  "sourceDocumentsUsed": ["doc-name"]
}"""
    ).replace("{subject_name}", subject_name)

    user_obj = {
        "task": "Generate or revise a teacher workspace resource draft.",
        "subject_name": _collapse_whitespace(request.subjectName or "Subject"),
        "topic_title": _collapse_whitespace(request.topicTitle),
        "focus_area": focus_area,
        "unit_title": _collapse_whitespace(request.unitTitle or ""),
        "grade_level": _collapse_whitespace(request.gradeLevel or ""),
        "content_type": request.contentType,
        "title": _collapse_whitespace(request.title or ""),
        "objective": _collapse_whitespace(request.objective or ""),
        "teacher_prompt": _trim_text(request.teacherPrompt or "", MAX_CONTEXT_CHARS),
        "existing_content": _trim_text(request.existingContent or "", MAX_REFERENCE_PER_DOC_CHARS),
        "variant": request.variant,
        "related_records": _unique_strings(request.relatedRecords)[:6],
        "reference_documents": prepared_docs.usable_documents,
        "reference_fallback_used": prepared_docs.fallback_used,
    }

    raw = llm_client.generate(system_text, json.dumps(user_obj, ensure_ascii=False), max_new_tokens=900)
    try:
        candidate = _parse_json_candidate(raw)
        if not isinstance(candidate, dict):
            raise ValueError("Resource generation response is not an object")
    except Exception as exc:
        print(f"[resource-generation] first pass invalid output: {exc}; preview={_preview(raw)}")
        retry_system = system_text + "\n\nReturn ONLY valid JSON matching the requested schema."
        raw_retry = llm_client.generate(retry_system, json.dumps(user_obj, ensure_ascii=False), max_new_tokens=900)
        try:
            candidate = _parse_json_candidate(raw_retry)
            if not isinstance(candidate, dict):
                raise ValueError("Resource generation retry response is not an object")
        except Exception as retry_exc:
            print(f"[resource-generation] retry invalid output: {retry_exc}; preview={_preview(raw_retry)}")
            return fallback

    title = _coerce_candidate_text(candidate, "title", "resourceTitle") or fallback.title
    content_html = _coerce_candidate_html(candidate)
    summary = _coerce_candidate_text(candidate, "summary", "description", "message") or fallback.summary
    teacher_message = _coerce_candidate_text(candidate, "teacherMessage", "message", "note") or fallback.teacherMessage
    if not content_html:
        return GeneratedTeacherResource(
            title=title or fallback.title,
            contentHtml=fallback.contentHtml,
            summary=summary or fallback.summary,
            teacherMessage=teacher_message or fallback.teacherMessage,
            sourceDocumentsUsed=fallback.sourceDocumentsUsed,
            referenceFallbackUsed=prepared_docs.fallback_used,
        )
    source_used = _normalize_source_documents(candidate.get("sourceDocumentsUsed"), source_doc_names)

    return GeneratedTeacherResource(
        title=title or fallback.title,
        contentHtml=content_html,
        summary=summary or fallback.summary,
        teacherMessage=teacher_message or fallback.teacherMessage,
        sourceDocumentsUsed=source_used,
        referenceFallbackUsed=prepared_docs.fallback_used,
    )


def _existing_question_type_mode(existing_questions: list[TeacherPracticeGenerationRequest | Any]) -> str | None:
    if not existing_questions:
        return None
    has_mcq = False
    has_structured = False
    for question in existing_questions:
        if not isinstance(question, dict):
            question_type = getattr(question, "questionType", None) or getattr(question, "type", None)
        else:
            question_type = question.get("questionType") or question.get("type")
        normalized = _collapse_whitespace(str(question_type or "")).lower().replace("_", "-")
        if normalized == "multiple-choice":
            has_mcq = True
        elif normalized == "short-answer":
            has_structured = True
    if has_mcq and has_structured:
        return "mixed"
    if has_mcq:
        return "multiple_choice"
    if has_structured:
        return "structured"
    return None


def _parse_question_count_token(token: str) -> int | None:
    normalized = _collapse_whitespace(token).lower()
    if not normalized:
        return None
    if normalized.isdigit():
        return max(1, int(normalized))
    return _QUESTION_COUNT_WORDS.get(normalized)


def _resolve_requested_practice_question_count(
    teacher_prompt: str | None,
    *,
    existing_count: int,
    requested_count: int | None,
) -> tuple[int, int]:
    base_count = max(1, requested_count or 0, existing_count)
    prompt_text = _collapse_whitespace(teacher_prompt or "").lower()
    if not prompt_text:
        return base_count, 0

    additive_patterns = (
        r"\b(?:add|append|include|create|generate)\s+(?P<count>a|an|another|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:more|additional|extra|new)?\s*questions?\b",
        r"\b(?P<count>a|an|another|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:more|additional|extra|new)\s+questions?\b",
    )
    for pattern in additive_patterns:
        match = re.search(pattern, prompt_text)
        if not match:
            continue
        parsed_count = _parse_question_count_token(match.group("count") or "")
        if not parsed_count:
            continue
        return max(base_count, existing_count + parsed_count), parsed_count

    return base_count, 0


def _build_practice_description(
    request: TeacherPracticeGenerationRequest,
    question_count: int,
    focus_area: str,
) -> str:
    description = _collapse_whitespace(request.description or "")
    if description:
        return description
    objective = _collapse_whitespace(request.objective or "")
    practice_type = _collapse_whitespace(request.practiceType or "quiz").title()
    topic = _collapse_whitespace(request.topicTitle)
    grade_level = _collapse_whitespace(request.gradeLevel or "")
    if focus_area.lower() != topic.lower():
        parts = [f"{practice_type} on {focus_area} in {topic}."]
    else:
        parts = [f"{practice_type} on {topic}."]
    if objective:
        parts.append(objective)
    if grade_level:
        parts.append(f"Designed for {grade_level}.")
    parts.append(f"Includes {question_count} AI-generated question{'s' if question_count != 1 else ''}.")
    return " ".join(parts)


def _normalize_practice_questions(
    generated_questions: list[GeneratedAssessmentQuestion],
) -> list[GeneratedTeacherPracticeQuestion]:
    normalized_questions: list[GeneratedTeacherPracticeQuestion] = []
    for index, question in enumerate(generated_questions, start=1):
        question_type = _collapse_whitespace(str(question.type or "short_answer")).lower().replace("_", "-")
        is_objective = question_type in {"multiple-choice", "true-false"}
        normalized_type = "multiple-choice" if is_objective else "short-answer"
        options = question.options if normalized_type == "multiple-choice" else []
        correct_answers = _unique_strings(question.correctAnswers)
        marking_guide_payload = question.markingGuide
        correct_answer = _collapse_whitespace(
            str(
                question.correctAnswer
                or (marking_guide_payload.expectedAnswer if marking_guide_payload else "")
                or ""
            )
        )
        if normalized_type == "multiple-choice":
            if not correct_answers and correct_answer:
                correct_answers = [correct_answer]
            if not correct_answer and correct_answers:
                correct_answer = correct_answers[0]
        else:
            correct_answers = []
            correct_answer = _collapse_whitespace(
                str(
                    (marking_guide_payload.expectedAnswer if marking_guide_payload else "")
                    or question.correctAnswer
                    or ""
                )
            )

        rubric_items = marking_guide_payload.rubricItems if marking_guide_payload else []
        marking_points = [
            _collapse_whitespace(str(item.description or ""))
            for item in rubric_items
            if _collapse_whitespace(str(item.description or ""))
        ]
        marking_guide = "\n".join(marking_points) if marking_points else _collapse_whitespace(
            str(
                (marking_guide_payload.expectedAnswer if marking_guide_payload else "")
                or question.explanation
                or correct_answer
                or ""
            )
        )

        normalized_questions.append(
            GeneratedTeacherPracticeQuestion(
                id=f"ai-question-{index}",
                prompt=_collapse_whitespace(str(question.text or f"Question {index}")),
                type=normalized_type,
                marks=max(1, int(question.maxMarks or question.points or 1)),
                options=options,
                correctAnswers=correct_answers,
                correctAnswer=correct_answer,
                markingGuide=marking_guide,
            )
        )
    return normalized_questions


def generate_teacher_practice(
    request: TeacherPracticeGenerationRequest,
    *,
    llm_client,
) -> GeneratedTeacherPractice:
    focus_area = _resolve_generation_focus(request.topicTitle, request.title)
    existing_questions = [
        {
            "prompt": _collapse_whitespace(str(question.prompt or "")),
            "questionType": question.questionType or question.type or "short-answer",
            "marks": question.marks,
            "options": _unique_strings(question.options),
            "correctAnswers": _unique_strings(question.correctAnswers),
            "correctAnswer": _collapse_whitespace(str(question.correctAnswer or "")),
            "markingGuide": _collapse_whitespace(str(question.markingGuide or "")),
        }
        for question in request.existingQuestions
        if _collapse_whitespace(str(question.prompt or ""))
    ]
    question_count, additive_question_count = _resolve_requested_practice_question_count(
        request.teacherPrompt,
        existing_count=len(existing_questions),
        requested_count=request.numberOfQuestions,
    )
    question_type_mode = _existing_question_type_mode(existing_questions) or request.questionTypeMode
    context_parts = [
        f"Generate a teacher-ready {request.practiceType} for the topic {request.topicTitle}.",
        f"Immediate subtopic or lesson focus: {focus_area}.",
        f"Grade level: {request.gradeLevel or 'Not specified'}.",
        f"Learning objective: {request.objective or focus_area}.",
        f"Teacher direction: {request.teacherPrompt or 'Generate a clear, classroom-ready practice set.'}",
        f"Practice description: {request.description or ''}",
        f"Subject anchor: {request.subjectName or 'Use the declared subject only.'}",
        f"Every question must primarily assess {focus_area} while staying within the broader topic {request.topicTitle}.",
        "Return questions with correct answers and marking guidance suitable for the practice canvas.",
    ]
    if request.variant:
        context_parts.append("Produce a distinct alternative version instead of repeating the current structure.")
    if existing_questions:
        context_parts.append(
            "Existing questions to revise or use as style/context:\n" + json.dumps(existing_questions, ensure_ascii=False)
        )
        if additive_question_count > 0:
            context_parts.append(
                f"Keep the existing {len(existing_questions)} question(s) unless the teacher explicitly asked to revise them. "
                f"Return the full updated set with {question_count} question(s) total, including {additive_question_count} newly added question(s)."
            )

    practice_attributes: dict[str, str] = {
        focus_area: request.objective or focus_area,
    }
    if focus_area.lower() != _collapse_whitespace(request.topicTitle).lower():
        practice_attributes[request.topicTitle] = request.practiceType
    if request.unitTitle:
        practice_attributes[request.unitTitle] = request.practiceType
    if request.subjectName:
        practice_attributes[request.subjectName] = request.gradeLevel or request.practiceType

    assessment_request = AssessmentGenerationRequest(
        context="\n".join(part for part in context_parts if _collapse_whitespace(part)),
        subjectName=request.subjectName,
        difficulty=request.difficulty,
        questionTypes=question_type_mode,
        numberOfQuestions=question_count,
        attributes=practice_attributes,
        referenceDocuments=request.referenceDocuments,
        tags=_unique_strings([focus_area, request.topicTitle, request.objective or "", request.subjectName or ""]),
    )
    try:
        generated_questions = generate_teacher_assessment_questions(assessment_request, llm_client=llm_client)
    except AssessmentGenerationError as exc:
        print(f"[practice-generation] using fallback questions: {exc}")
        prepared_docs = _prepare_reference_documents(assessment_request.referenceDocuments)
        generated_questions = _normalize_generated_questions(
            [],
            request=assessment_request,
            prepared_docs=prepared_docs,
        )
    normalized_questions = _normalize_practice_questions(generated_questions)
    source_documents_used = _unique_strings(
        source_name
        for question in generated_questions
        for source_name in (question.sourceDocumentsUsed or [])
    )
    reference_fallback_used = any(question.referenceFallbackUsed for question in generated_questions)
    title = _collapse_whitespace(
        request.title or f"{focus_area} {_collapse_whitespace(request.practiceType).title()}"
    ).strip()
    return GeneratedTeacherPractice(
        title=title or f"{focus_area} Practice",
        description=_build_practice_description(request, len(normalized_questions), focus_area),
        practiceType=request.practiceType,
        difficulty=request.difficulty,
        numberOfQuestions=len(normalized_questions),
        questions=normalized_questions,
        summary=f"Generated {len(normalized_questions)} practice question{'s' if len(normalized_questions) != 1 else ''} focused on {focus_area}.",
        teacherMessage=(
            f"I updated the practice set to {len(normalized_questions)} question"
            f"{'s' if len(normalized_questions) != 1 else ''} with answers ready for the practice canvas."
        ),
        sourceDocumentsUsed=source_documents_used,
        referenceFallbackUsed=reference_fallback_used,
    )


def generate_teacher_development_plan(
    request: PlanGenerationRequest,
    *,
    llm_client,
) -> GeneratedDevelopmentPlan:
    resolved = _resolve_plan_request(request)
    prepared_docs = _prepare_reference_documents(request.referenceDocuments)
    system_text, user_text = _build_plan_generation_prompt(resolved, prepared_docs)
    max_new_tokens = _plan_generation_max_tokens(len(resolved["critical_skills"]))

    raw: str | None = None
    raw_retry: str | None = None
    try:
        raw = llm_client.generate(system_text, user_text, max_new_tokens=max_new_tokens)
        candidate = _parse_plan_payload(raw)
        return _normalize_generated_plan(candidate, resolved=resolved, prepared_docs=prepared_docs)
    except Exception as exc:
        if raw is not None:
            print(f"[plan-generation] first pass invalid output: {exc}; preview={_preview(raw)}")
        try:
            raw_retry = llm_client.generate(
                system_text + "\n\nReturn ONLY valid JSON matching the requested plan schema.",
                user_text,
                max_new_tokens=max_new_tokens,
            )
            candidate = _parse_plan_payload(raw_retry)
            return _normalize_generated_plan(candidate, resolved=resolved, prepared_docs=prepared_docs)
        except Exception as retry_exc:
            if raw_retry is not None:
                print(f"[plan-generation] retry invalid output: {retry_exc}; preview={_preview(raw_retry)}")
            fallback_plan = _build_fallback_plan(resolved=resolved, prepared_docs=prepared_docs)
            if fallback_plan:
                return fallback_plan
            raise AssessmentGenerationError(
                "Model returned invalid development plan JSON after retry.",
                first_raw=raw,
                retry_raw=raw_retry,
            ) from retry_exc


def _plan_generation_max_tokens(critical_skill_count: int) -> int:
    configured = os.getenv("PLAN_GENERATION_MAX_NEW_TOKENS")
    if configured:
        try:
            return max(192, int(configured))
        except ValueError:
            pass
    return min(900, max(320, 220 + (critical_skill_count * 90)))


def _resolve_plan_request(request: PlanGenerationRequest) -> dict[str, Any]:
    if request.studentProfile and request.subject:
        first_name = request.studentProfile.firstName
        last_name = request.studentProfile.lastName
        subject_name = request.subject.name
        subject_id = request.subject.id or request.requestContext.subjectId if request.requestContext else request.subject.id
        overall_score = _clamp_percentage(request.studentProfile.overallScore)
        potential_overall = overall_score
        performance = request.studentProfile.performance or "Average"
        engagement = request.studentProfile.engagement or "Medium"
        plan_name = request.planPreferences.name if request.planPreferences and request.planPreferences.name else f"{subject_name} Development Plan"
        context = (request.planPreferences.context if request.planPreferences and request.planPreferences.context else request.context) or "Focus on actionable steps, varied resources, and clear goals."
        step_count = request.planPreferences.stepCount if request.planPreferences and request.planPreferences.stepCount else None
        objective = request.planPreferences.objective if request.planPreferences else None
        guidance = request.planPreferences.guidance if request.planPreferences else None
        approach = request.planPreferences.stepApproach if request.planPreferences else None
        critical_skills = _resolve_critical_skills_from_canonical(request)
        target_overall = _derive_target_overall_from_skills(critical_skills, fallback=potential_overall)
        potential_overall = max(potential_overall, target_overall)
    else:
        first_name = request.firstName or "Student"
        last_name = request.lastName or ""
        subject_name = request.subjectName or "Subject"
        subject_id = request.subjectID
        overall_score = _parse_percentage(request.currentOverallScore)
        potential_overall = _parse_percentage(request.potentialOverallScore, fallback=overall_score)
        target_overall = _parse_percentage(request.targetScore, fallback=max(85.0, potential_overall))
        performance = request.overallPerformance or "Average"
        engagement = request.overallEngagement or "Medium"
        plan_name = f"{subject_name} Development Plan"
        context = request.context or "Focus on actionable steps, varied resources, and clear goals."
        step_count = None
        objective = None
        guidance = None
        approach = None
        critical_skills = _resolve_critical_skills_from_legacy(request)
        if critical_skills:
            potential_overall = max(potential_overall, _derive_target_overall_from_skills(critical_skills, fallback=potential_overall))
            target_overall = max(target_overall, _derive_target_overall_from_skills(critical_skills, fallback=target_overall))

    if not critical_skills:
        critical_skills = [
            {
                "name": "Overall Performance",
                "attribute_id": "overall-performance",
                "current_score": overall_score,
                "potential_score": max(overall_score, potential_overall),
                "target_score": max(target_overall, potential_overall),
                "gap": max(0.0, max(target_overall, potential_overall) - overall_score),
                "weight": 1.0,
                "reason": "No specific critical skill data was supplied, so the plan focuses on overall performance improvement.",
            }
        ]

    normalized_step_count = step_count or min(5, max(3, len(critical_skills) + 1))
    return {
        "student_name": " ".join(part for part in [first_name, last_name] if part).strip() or "Student",
        "first_name": first_name,
        "last_name": last_name,
        "subject_name": subject_name,
        "subject_id": subject_id,
        "current_overall": _clamp_percentage(overall_score),
        "potential_overall": _clamp_percentage(potential_overall),
        "target_overall": _clamp_percentage(target_overall),
        "performance": performance,
        "engagement": engagement,
        "plan_name": plan_name,
        "context": context,
        "objective": objective or f"Close the learner's gaps in {', '.join(skill['name'] for skill in critical_skills[:3])}.",
        "guidance": guidance or "Use scaffolded instruction, concrete examples, focused practice, and short mastery checks.",
        "step_approach": approach or "scaffolded",
        "step_count": normalized_step_count,
        "critical_skills": critical_skills[:5],
    }


def _resolve_critical_skills_from_legacy(request: PlanGenerationRequest) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for index, item in enumerate(request.attributeDetails, start=1):
        current_score = _parse_percentage(item.currentScore)
        potential_score = _parse_percentage(item.potentialScore, fallback=current_score)
        target_score = _parse_percentage(item.targetScore, fallback=max(current_score, potential_score))
        gap = _parse_percentage(item.gap, fallback=max(0.0, target_score - current_score))
        weight = _parse_float(item.weight, fallback=1.0)
        skills.append(
            {
                "name": item.name,
                "attribute_id": _slugify(item.name),
                "current_score": current_score,
                "potential_score": potential_score,
                "target_score": target_score,
                "gap": gap,
                "weight": weight,
                "reason": f"Current {current_score:.0f}% vs target {target_score:.0f}% (gap {gap:.0f}%).",
                "priority": index,
            }
        )

    positive_gap_skills = [skill for skill in skills if skill["gap"] > 0]
    selected = positive_gap_skills or skills
    selected.sort(key=lambda skill: (skill["gap"], skill["weight"]), reverse=True)
    for rank, skill in enumerate(selected, start=1):
        skill["priority"] = rank
    return selected[:5]


def _resolve_critical_skills_from_canonical(request: PlanGenerationRequest) -> list[dict[str, Any]]:
    critical: list[dict[str, Any]] = []
    if request.skillSnapshot and request.skillSnapshot.criticalSkills:
        for item in request.skillSnapshot.criticalSkills:
            matched_selected = None
            if request.skillSnapshot.selectedSkills:
                matched_selected = next(
                    (
                        selected
                        for selected in request.skillSnapshot.selectedSkills
                        if (item.attributeId and selected.attributeId == item.attributeId)
                        or selected.name.lower() == item.name.lower()
                    ),
                    None,
                )
            current_score = _clamp_percentage(matched_selected.currentScore if matched_selected else None)
            target_score = _clamp_percentage(matched_selected.targetScore if matched_selected else None, fallback=current_score)
            potential_score = _clamp_percentage(matched_selected.potentialScore if matched_selected else None, fallback=target_score)
            gap = float(item.gap if item.gap is not None else max(0.0, target_score - current_score))
            critical.append(
                {
                    "name": item.name,
                    "attribute_id": item.attributeId or _slugify(item.name),
                    "current_score": current_score,
                    "potential_score": potential_score,
                    "target_score": target_score,
                    "gap": max(0.0, gap),
                    "weight": matched_selected.weight if matched_selected and matched_selected.weight is not None else 1.0,
                    "reason": item.reason or f"Priority {item.priority or len(critical) + 1} skill gap requiring targeted intervention.",
                    "priority": item.priority or len(critical) + 1,
                }
            )

    if critical:
        critical.sort(key=lambda skill: (skill["priority"], -skill["gap"]))
        return critical

    if request.skillSnapshot and request.skillSnapshot.selectedSkills:
        derived = []
        for index, item in enumerate(request.skillSnapshot.selectedSkills, start=1):
            current_score = _clamp_percentage(item.currentScore)
            target_score = _clamp_percentage(item.targetScore, fallback=current_score)
            potential_score = _clamp_percentage(item.potentialScore, fallback=target_score)
            gap = item.gap if item.gap is not None else max(0.0, target_score - current_score)
            derived.append(
                {
                    "name": item.name,
                    "attribute_id": item.attributeId or _slugify(item.name),
                    "current_score": current_score,
                    "potential_score": potential_score,
                    "target_score": target_score,
                    "gap": max(0.0, float(gap)),
                    "weight": item.weight if item.weight is not None else 1.0,
                    "reason": f"Current {current_score:.0f}% vs target {target_score:.0f}%.",
                    "priority": index,
                }
            )
        derived = [skill for skill in derived if skill["gap"] > 0] or derived
        derived.sort(key=lambda skill: (skill["gap"], skill["weight"]), reverse=True)
        for rank, skill in enumerate(derived, start=1):
            skill["priority"] = rank
        return derived[:5]

    return []


def _build_plan_generation_prompt(
    resolved: dict[str, Any],
    prepared_docs: ReferencePreparationResult,
) -> tuple[str, str]:
    system_text = (
        f"You are an AI assistant that creates teacher-facing student development plans for {resolved['subject_name']}.\n"
        + """Return JSON ONLY. No markdown. No prose outside JSON.

Rules:
- Build the plan around the CRITICAL SKILLS ONLY.
- Do not add unrelated skills or generic filler topics.
- Default to ZIMSEC O Level high-school expectations unless the input explicitly requests another level.
- Do not generate tertiary, university, or advanced specialist content.
- Every step must directly address one or more critical skills from the input.
- Use practical, teacher-actionable step titles.
- Steps must use only these types: video, document, assessment, assignment, quiz, discussion.
- Keep the number of steps exactly as requested.
- skills must include only the critical skills being targeted.
- progress must be 0 for a newly generated plan.
- potentialOverall should be a realistic improvement target between 0 and 100.
- eta should be the estimated number of days to execute the plan.
- If reference documents are missing or unusable, silently fall back to the learner profile, critical skills, context, and your own subject knowledge of {subject_name}.
- Generate the plan only for {subject_name}. Use the declared subject to disambiguate broad skill names.

Return exactly this schema:
{
  "name": "...",
  "description": "...",
  "progress": 0,
  "potentialOverall": 78,
  "eta": 28,
  "performance": "Average",
  "skills": [
    {
      "name": "Algorithms",
      "score": 75,
      "subskills": [
        { "name": "Binary Search", "score": 72, "color": "yellow" }
      ]
    }
  ],
  "steps": [
    {
      "title": "...",
      "type": "document",
      "content": "<p>...</p>",
      "link": "",
      "additionalResources": [],
      "order": 1
    }
  ],
  "subjectId": "..."
}"""
    ).replace("{subject_name}", resolved["subject_name"])
    user_obj = {
        "student_name": resolved["student_name"],
        "subject_name": resolved["subject_name"],
        "subject_id": resolved["subject_id"],
        "current_overall": resolved["current_overall"],
        "potential_overall": resolved["potential_overall"],
        "target_overall": resolved["target_overall"],
        "performance": resolved["performance"],
        "engagement": resolved["engagement"],
        "plan_name": resolved["plan_name"],
        "context": _trim_text(resolved["context"], MAX_CONTEXT_CHARS),
        "objective": resolved["objective"],
        "guidance": resolved["guidance"],
        "step_approach": resolved["step_approach"],
        "step_count": resolved["step_count"],
        "critical_skills": [
            {
                "name": skill["name"],
                "current_score": skill["current_score"],
                "potential_score": skill["potential_score"],
                "target_score": skill["target_score"],
                "gap": skill["gap"],
                "reason": skill["reason"],
            }
            for skill in resolved["critical_skills"]
        ],
        "reference_documents": prepared_docs.usable_documents,
        "reference_fallback_used": prepared_docs.fallback_used,
    }
    return system_text, json.dumps(user_obj, ensure_ascii=False)


def _parse_plan_payload(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    if not candidates:
        raise ValueError("Model output did not contain a JSON object")

    expected_keys = {"name", "description", "skills", "steps"}
    for candidate in reversed(candidates):
        if expected_keys.issubset(candidate.keys()):
            return candidate
    return candidates[-1]


def _normalize_generated_plan(
    payload: dict[str, Any],
    *,
    resolved: dict[str, Any],
    prepared_docs: ReferencePreparationResult,
) -> GeneratedDevelopmentPlan:
    critical_skills = resolved["critical_skills"]
    critical_names = [skill["name"] for skill in critical_skills]
    normalized_skills = _normalize_plan_skills(payload.get("skills"), critical_skills)
    if not normalized_skills:
        normalized_skills = _fallback_plan_skills(critical_skills)

    step_count = resolved["step_count"]
    normalized_steps = _normalize_plan_steps(
        payload.get("steps"),
        critical_skills=critical_skills,
        step_count=step_count,
        subject_name=resolved["subject_name"],
    )
    if not normalized_steps:
        normalized_steps = _fallback_plan_steps(
            critical_skills=critical_skills,
            step_count=step_count,
            subject_name=resolved["subject_name"],
            objective=resolved["objective"],
            guidance=resolved["guidance"],
        )

    name = _collapse_whitespace(str(payload.get("name") or resolved["plan_name"])) or resolved["plan_name"]
    description = _collapse_whitespace(
        str(
            payload.get("description")
            or f"Personalized plan for {resolved['student_name']} targeting {', '.join(critical_names[:3])}."
        )
    )
    potential_overall = _clamp_percentage(payload.get("potentialOverall"), fallback=resolved["potential_overall"])
    eta = _parse_eta_days(payload.get("eta"), fallback=(len(normalized_steps) * 7))
    performance = _collapse_whitespace(str(payload.get("performance") or resolved["performance"])) or resolved["performance"]

    return GeneratedDevelopmentPlan(
        name=name,
        description=description,
        progress=0,
        potentialOverall=int(round(potential_overall)),
        eta=eta,
        performance=performance,
        skills=normalized_skills,
        steps=normalized_steps,
        subjectId=resolved["subject_id"],
        referenceFallbackUsed=prepared_docs.fallback_used,
        criticalSkillsUsed=critical_names,
    )


def _normalize_plan_skills(raw_skills: Any, critical_skills: list[dict[str, Any]]) -> list[GeneratedPlanSkill]:
    if not isinstance(raw_skills, list):
        return []

    canonical_by_name = {skill["name"].lower(): skill for skill in critical_skills}
    normalized: list[GeneratedPlanSkill] = []

    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            continue
        name = _collapse_whitespace(str(raw_skill.get("name") or ""))
        if not name:
            continue
        matched = canonical_by_name.get(name.lower())
        if not matched:
            continue
        score = int(round(_clamp_percentage(raw_skill.get("score"), fallback=matched["target_score"])))
        subskills = _normalize_plan_subskills(raw_skill.get("subskills"), matched)
        normalized.append(GeneratedPlanSkill(name=matched["name"], score=score, subskills=subskills))

    if normalized:
        return normalized
    return _fallback_plan_skills(critical_skills)


def _normalize_plan_subskills(raw_subskills: Any, matched_skill: dict[str, Any]) -> list[GeneratedPlanSubskill]:
    normalized: list[GeneratedPlanSubskill] = []
    if isinstance(raw_subskills, list):
        for raw_subskill in raw_subskills[:3]:
            if not isinstance(raw_subskill, dict):
                continue
            name = _collapse_whitespace(str(raw_subskill.get("name") or ""))
            if not name:
                continue
            score = int(round(_clamp_percentage(raw_subskill.get("score"), fallback=matched_skill["target_score"])))
            color = _normalize_skill_color(raw_subskill.get("color"))
            normalized.append(GeneratedPlanSubskill(name=name, score=score, color=color))
    if normalized:
        return normalized
    return [
        GeneratedPlanSubskill(
            name=f"{matched_skill['name']} mastery target",
            score=int(round(_clamp_percentage(matched_skill["target_score"]))),
            color=_color_for_gap(matched_skill["gap"]),
        )
    ]


def _normalize_plan_steps(
    raw_steps: Any,
    *,
    critical_skills: list[dict[str, Any]],
    step_count: int,
    subject_name: str,
) -> list[GeneratedPlanStep]:
    if not isinstance(raw_steps, list):
        return []

    normalized: list[GeneratedPlanStep] = []
    critical_names = [skill["name"] for skill in critical_skills]
    for index, raw_step in enumerate(raw_steps[:step_count], start=1):
        if not isinstance(raw_step, dict):
            continue
        title = _collapse_whitespace(str(raw_step.get("title") or ""))
        step_type = _normalize_plan_step_type(raw_step.get("type"))
        content = str(raw_step.get("content") or "")
        link = _collapse_whitespace(str(raw_step.get("link") or ""))
        resources = _normalize_string_list(raw_step.get("additionalResources"))
        focus_skill = critical_skills[(index - 1) % len(critical_skills)]

        if not title:
            title = _fallback_step_title(step_type, focus_skill["name"], index)
        if not _mentions_any_skill(title, critical_names) and not _mentions_any_skill(content, critical_names):
            content = _augment_step_content(content, focus_skill, subject_name)
            if focus_skill["name"].lower() not in title.lower():
                title = f"{title} for {focus_skill['name']}"
        elif not content.strip():
            content = _augment_step_content(content, focus_skill, subject_name)

        normalized.append(
            GeneratedPlanStep(
                title=title,
                type=step_type,
                content=content,
                link=link,
                additionalResources=resources,
                order=index,
            )
        )

    return normalized


def _build_fallback_plan(
    *,
    resolved: dict[str, Any],
    prepared_docs: ReferencePreparationResult,
) -> GeneratedDevelopmentPlan:
    critical_skills = resolved["critical_skills"]
    return GeneratedDevelopmentPlan(
        name=resolved["plan_name"],
        description=f"Personalized plan for {resolved['student_name']} targeting {', '.join(skill['name'] for skill in critical_skills[:3])}.",
        progress=0,
        potentialOverall=int(round(resolved["potential_overall"])),
        eta=max(1, resolved["step_count"] * 7),
        performance=resolved["performance"],
        skills=_fallback_plan_skills(critical_skills),
        steps=_fallback_plan_steps(
            critical_skills=critical_skills,
            step_count=resolved["step_count"],
            subject_name=resolved["subject_name"],
            objective=resolved["objective"],
            guidance=resolved["guidance"],
        ),
        subjectId=resolved["subject_id"],
        referenceFallbackUsed=prepared_docs.fallback_used,
        criticalSkillsUsed=[skill["name"] for skill in critical_skills],
    )


def _fallback_plan_skills(critical_skills: list[dict[str, Any]]) -> list[GeneratedPlanSkill]:
    return [
        GeneratedPlanSkill(
            name=skill["name"],
            score=int(round(_clamp_percentage(skill["target_score"]))),
            subskills=[
                GeneratedPlanSubskill(
                    name=f"{skill['name']} mastery target",
                    score=int(round(_clamp_percentage(skill["target_score"]))),
                    color=_color_for_gap(skill["gap"]),
                )
            ],
        )
        for skill in critical_skills
    ]


def _fallback_plan_steps(
    *,
    critical_skills: list[dict[str, Any]],
    step_count: int,
    subject_name: str,
    objective: str,
    guidance: str,
) -> list[GeneratedPlanStep]:
    steps: list[GeneratedPlanStep] = []
    step_types = ["document", "assignment", "discussion", "quiz", "assessment"]
    for index in range(1, step_count + 1):
        focus_skill = critical_skills[(index - 1) % len(critical_skills)]
        step_type = step_types[(index - 1) % len(step_types)]
        steps.append(
            GeneratedPlanStep(
                title=_fallback_step_title(step_type, focus_skill["name"], index),
                type=step_type,
                content=_augment_step_content("", focus_skill, subject_name, objective=objective, guidance=guidance),
                link="",
                additionalResources=[],
                order=index,
            )
        )
    return steps


def _fallback_step_title(step_type: str, skill_name: str, index: int) -> str:
    if step_type == "document":
        return f"Review core concepts for {skill_name}"
    if step_type == "assignment":
        return f"Practice targeted tasks on {skill_name}"
    if step_type == "discussion":
        return f"Discuss misconceptions in {skill_name}"
    if step_type == "quiz":
        return f"Check mastery of {skill_name}"
    if step_type == "assessment":
        return f"Assess progress in {skill_name}"
    return f"Step {index}: Improve {skill_name}"


def _augment_step_content(
    content: str,
    focus_skill: dict[str, Any],
    subject_name: str,
    *,
    objective: str | None = None,
    guidance: str | None = None,
) -> str:
    base_content = _collapse_whitespace(content)
    objective_text = objective or f"Improve learner performance in {focus_skill['name']}."
    guidance_text = guidance or "Use scaffolded practice, examples, and quick formative checks."
    parts = [
        f"<p><strong>Critical Skill Focus:</strong> {focus_skill['name']}</p>",
        f"<p><strong>Subject:</strong> {subject_name}</p>",
        f"<p><strong>Teacher Objective:</strong> {objective_text}</p>",
        f"<p><strong>Guidance:</strong> {guidance_text}</p>",
        f"<p><strong>Why this matters:</strong> {focus_skill['reason']}</p>",
    ]
    if base_content:
        parts.insert(0, f"<p>{base_content}</p>")
    return "".join(parts)


def _normalize_plan_step_type(value: Any) -> str:
    normalized = _collapse_whitespace(str(value or "")).lower().replace("-", "_").replace(" ", "_")
    allowed = {"video", "document", "assessment", "assignment", "quiz", "discussion"}
    if normalized in allowed:
        return normalized
    return "document"


def _normalize_skill_color(value: Any) -> str:
    normalized = _collapse_whitespace(str(value or "")).lower()
    if normalized in {"yellow", "cyan", "blue", "green", "red"}:
        return normalized
    return "blue"


def _color_for_gap(gap: float) -> str:
    if gap >= 25:
        return "red"
    if gap >= 15:
        return "yellow"
    if gap > 0:
        return "cyan"
    return "green"


def _derive_target_overall_from_skills(critical_skills: list[dict[str, Any]], *, fallback: float) -> float:
    if not critical_skills:
        return fallback
    return sum(_clamp_percentage(skill["target_score"]) for skill in critical_skills) / len(critical_skills)


def _parse_percentage(value: Any, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return _clamp_percentage(float(value))
    text = str(value).strip().replace("%", "")
    try:
        return _clamp_percentage(float(text))
    except ValueError:
        return fallback


def _parse_float(value: Any, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return fallback


def _parse_eta_days(value: Any, *, fallback: int) -> int:
    if value is None:
        return max(1, int(fallback))
    if isinstance(value, (int, float)):
        return max(1, int(round(float(value))))

    text = str(value).strip().lower()
    if not text:
        return max(1, int(fallback))

    direct_value = _parse_float(text, fallback=float("nan"))
    if not math.isnan(direct_value):
        return max(1, int(round(direct_value)))

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return max(1, int(fallback))

    numeric_value = float(match.group(0))
    if "week" in text:
        numeric_value *= 7
    elif "month" in text:
        numeric_value *= 30

    return max(1, int(round(numeric_value)))


def _clamp_percentage(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, min(float(value), 100.0))
    except (TypeError, ValueError):
        return max(0.0, min(float(fallback), 100.0))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "skill"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in (_collapse_whitespace(str(entry)) for entry in value) if item]
    if isinstance(value, str):
        cleaned = _collapse_whitespace(value)
        return [cleaned] if cleaned else []
    return []


def _mentions_any_skill(text: str, skill_names: list[str]) -> bool:
    normalized = text.lower()
    return any(skill.lower() in normalized for skill in skill_names)
