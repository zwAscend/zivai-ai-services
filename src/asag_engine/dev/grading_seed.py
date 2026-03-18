from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


DEFAULT_RUBRIC_ITEMS = [
    {"description": "Defines binary search as checking the middle element and halving the search space", "mark_value": 3},
    {"description": "States that the data must be sorted", "mark_value": 1},
    {"description": "States that direct/index access is typically needed", "mark_value": 1},
]


DEFAULTS = {
    "rubric": {
        "question_type_code": "short_answer",
        "question_stem": "Explain what binary search is and state two conditions required before it can be used.",
        "max_mark": 5.0,
        "student_answer_text": "Binary search checks the middle item and keeps searching in the half where the answer could be. It works when the data is sorted.",
        "rubric_items": DEFAULT_RUBRIC_ITEMS,
        "expected_answer": None,
    },
    "holistic": {
        "question_type_code": "short_answer",
        "question_stem": "Explain why prototyping is useful during systems development.",
        "max_mark": 4.0,
        "student_answer_text": "A prototype helps show how the system may work before the final version is built.",
        "rubric_items": [],
        "expected_answer": "Prototyping helps validate requirements early, gather feedback, reduce design risk, and reveal usability issues before full development.",
    },
    "objective": {
        "question_type_code": "mcq",
        "question_stem": "Which number is equal to binary 1010?",
        "max_mark": 1.0,
        "student_answer_text": "10",
        "rubric_items": [],
        "expected_answer": "10",
    },
}


@dataclass
class SeedResult:
    school_id: str
    teacher_user_id: str
    student_user_id: str
    subject_id: str
    topic_id: str | None
    question_id: str
    marking_scheme_id: str | None
    assessment_id: str
    assessment_question_id: str
    assessment_assignment_id: str
    assessment_enrollment_id: str
    assessment_attempt_id: str
    attempt_answer_id: str
    scenario: str



def bootstrap_grading_scenario(session: Session, payload: dict[str, Any] | None = None) -> SeedResult:
    payload = payload or {}
    scenario = (payload.get("scenario") or "rubric").strip().lower()
    if scenario not in DEFAULTS:
        raise ValueError("scenario must be one of: rubric, holistic, objective")

    defaults = DEFAULTS[scenario]
    ensure_lookup_values(session)

    school_id = get_or_create_school(
        session,
        code=payload.get("school_code") or "ZVHS",
        name=payload.get("school_name") or "zivAI High School",
        country_code=payload.get("country_code") or "ZW",
    )
    teacher_user_id = get_or_create_user(
        session,
        email=payload.get("teacher_email") or "teacher@zivai.local",
        phone_number=payload.get("teacher_phone") or "263712000001",
        first_name=payload.get("teacher_first_name") or "Tariro",
        last_name=payload.get("teacher_last_name") or "Moyo",
        username=payload.get("teacher_username") or "teacher1",
    )
    student_user_id = get_or_create_user(
        session,
        email=payload.get("student_email") or "student@zivai.local",
        phone_number=payload.get("student_phone") or "263712000002",
        first_name=payload.get("student_first_name") or "Tinashe",
        last_name=payload.get("student_last_name") or "Dube",
        username=payload.get("student_username") or "student1",
    )
    ensure_school_user(session, school_id, teacher_user_id)
    ensure_school_user(session, school_id, student_user_id)

    subject_id = get_or_create_subject(
        session,
        code=payload.get("subject_code") or "CS",
        name=payload.get("subject_name") or "Computer Science",
        description=payload.get("subject_description") or "Core computer science for demo grading flows.",
    )
    topic_id = get_or_create_topic(
        session,
        subject_id=subject_id,
        code=payload.get("topic_code") or "CS.BS",
        name=payload.get("topic_name") or "Search Algorithms",
        description=payload.get("topic_description") or "Short-answer grading demo topic.",
    )

    question_type_code = payload.get("question_type_code") or defaults["question_type_code"]
    question_stem = payload.get("question_stem") or defaults["question_stem"]
    max_mark = float(payload.get("max_mark") or defaults["max_mark"])
    student_answer_text = payload.get("student_answer_text") or defaults["student_answer_text"]
    rubric_items = payload.get("rubric_items") if payload.get("rubric_items") is not None else defaults["rubric_items"]
    expected_answer = payload.get("expected_answer") if payload.get("expected_answer") is not None else defaults["expected_answer"]

    question_code = payload.get("question_code") or f"{scenario.upper()}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    rubric_json = build_rubric_json(scenario, rubric_items, expected_answer)
    question_id = create_question(
        session,
        subject_id=subject_id,
        topic_id=topic_id,
        author_id=teacher_user_id,
        code=question_code,
        stem=question_stem,
        question_type_code=question_type_code,
        max_mark=max_mark,
        rubric_json=rubric_json,
    )

    marking_scheme_id = None
    if scenario == "rubric" and rubric_items:
        marking_scheme_id = create_marking_scheme(session, question_id, max_mark, rubric_items)

    assessment_name = payload.get("assessment_name") or f"{scenario.title()} Grading Demo"
    assessment_id = create_assessment(
        session,
        school_id=school_id,
        subject_id=subject_id,
        created_by=teacher_user_id,
        name=assessment_name,
        description=payload.get("assessment_description") or "Demo assessment for grading API testing.",
        assessment_type=payload.get("assessment_type") or "quiz",
        max_score=max_mark,
        weight_pct=float(payload.get("weight_pct") or 10.0),
    )
    assessment_question_id = create_assessment_question(
        session,
        assessment_id=assessment_id,
        question_id=question_id,
        points=max_mark,
        rubric_scheme_id=marking_scheme_id,
    )
    assignment_id = create_assessment_assignment(
        session,
        assessment_id=assessment_id,
        assigned_by=teacher_user_id,
        title=payload.get("assignment_title") or assessment_name,
        instructions=payload.get("assignment_instructions") or "Answer the question and submit.",
        due_days=int(payload.get("due_days") or 7),
    )
    enrollment_id = get_or_create_assessment_enrollment(session, assignment_id, student_user_id)
    attempt_id = create_assessment_attempt(session, enrollment_id, max_mark)
    attempt_answer_id = create_attempt_answer(
        session,
        attempt_id=attempt_id,
        assessment_question_id=assessment_question_id,
        student_answer_text=student_answer_text,
        max_score=max_mark,
    )

    return SeedResult(
        school_id=school_id,
        teacher_user_id=teacher_user_id,
        student_user_id=student_user_id,
        subject_id=subject_id,
        topic_id=topic_id,
        question_id=question_id,
        marking_scheme_id=marking_scheme_id,
        assessment_id=assessment_id,
        assessment_question_id=assessment_question_id,
        assessment_assignment_id=assignment_id,
        assessment_enrollment_id=enrollment_id,
        assessment_attempt_id=attempt_id,
        attempt_answer_id=attempt_answer_id,
        scenario=scenario,
    )



def list_grading_targets(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT
              aa.id AS attempt_answer_id,
              aa.assessment_attempt_id,
              aa.assessment_question_id,
              aq.question_id,
              aq.sequence_index,
              q.stem,
              aa.student_answer_text,
              aa.ai_score,
              aa.human_score,
              aa.requires_review,
              aa.graded_at,
              aq.rubric_scheme_id,
              at.grading_status_code
            FROM lms.attempt_answers aa
            JOIN lms.assessment_questions aq ON aq.id = aa.assessment_question_id
            JOIN lms.questions q ON q.id = aq.question_id
            JOIN lms.assessment_attempts at ON at.id = aa.assessment_attempt_id
            WHERE aa.deleted_at IS NULL
            ORDER BY aa.created_at DESC
            LIMIT :limit
            """
        ),
        {"limit": max(1, min(limit, 100))},
    ).mappings().all()
    return [{key: _json_safe(value) for key, value in dict(row).items()} for row in rows]



def ensure_lookup_values(session: Session) -> None:
    statements = [
        "INSERT INTO lookups.exam_board (code, name) VALUES ('zimsec', 'ZIMSEC') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.exam_board (code, name) VALUES ('cambridge', 'CAMBRIDGE') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('short_answer', 'Short Answer') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('structured', 'Structured') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('mcq', 'Multiple Choice') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('multiple_choice', 'Multiple Choice') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('true_false', 'True/False') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.question_type (code, name) VALUES ('essay', 'Essay') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.grading_status (code, name) VALUES ('pending', 'Pending') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.grading_status (code, name) VALUES ('auto_graded', 'Auto Graded') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.grading_status (code, name) VALUES ('reviewed', 'Reviewed') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.assessment_enrollment_status (code, name) VALUES ('assigned', 'Assigned') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.assessment_enrollment_status (code, name) VALUES ('completed', 'Completed') ON CONFLICT (code) DO NOTHING",
        "INSERT INTO lookups.assessment_enrollment_status (code, name) VALUES ('late', 'Late') ON CONFLICT (code) DO NOTHING",
    ]
    for statement in statements:
        session.execute(text(statement))
    session.flush()



def get_or_create_school(session: Session, code: str, name: str, country_code: str) -> str:
    row = session.execute(
        text("SELECT id FROM lms.schools WHERE code = :code AND deleted_at IS NULL LIMIT 1"),
        {"code": code},
    ).mappings().first()
    if row:
        return str(row["id"])
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.schools (code, name, country_code)
                VALUES (:code, :name, :country_code)
                RETURNING id
                """
            ),
            {"code": code, "name": name, "country_code": country_code},
        ).scalar_one()
    )



def get_or_create_user(
    session: Session,
    email: str,
    phone_number: str,
    first_name: str,
    last_name: str,
    username: str,
) -> str:
    row = session.execute(
        text("SELECT id FROM lms.users WHERE email = :email AND deleted_at IS NULL LIMIT 1"),
        {"email": email},
    ).mappings().first()
    if row:
        return str(row["id"])
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.users (
                    email, phone_number, first_name, last_name, username, is_active
                ) VALUES (
                    :email, :phone_number, :first_name, :last_name, :username, TRUE
                )
                RETURNING id
                """
            ),
            {
                "email": email,
                "phone_number": phone_number,
                "first_name": first_name,
                "last_name": last_name,
                "username": username,
            },
        ).scalar_one()
    )



def ensure_school_user(session: Session, school_id: str, user_id: str) -> None:
    session.execute(
        text(
            """
            INSERT INTO lms.school_users (school_id, user_id, is_active)
            VALUES (:school_id, :user_id, TRUE)
            ON CONFLICT (school_id, user_id) DO NOTHING
            """
        ),
        {"school_id": school_id, "user_id": user_id},
    )



def get_or_create_subject(session: Session, code: str, name: str, description: str) -> str:
    row = session.execute(
        text("SELECT id FROM lms.subjects WHERE code = :code AND deleted_at IS NULL LIMIT 1"),
        {"code": code},
    ).mappings().first()
    if row:
        return str(row["id"])
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.subjects (code, name, description, is_active)
                VALUES (:code, :name, :description, TRUE)
                RETURNING id
                """
            ),
            {"code": code, "name": name, "description": description},
        ).scalar_one()
    )



def get_or_create_topic(session: Session, subject_id: str, code: str, name: str, description: str) -> str:
    row = session.execute(
        text(
            """
            SELECT id FROM lms.topics
            WHERE subject_id = :subject_id AND code = :code AND deleted_at IS NULL
            LIMIT 1
            """
        ),
        {"subject_id": subject_id, "code": code},
    ).mappings().first()
    if row:
        return str(row["id"])
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.topics (subject_id, code, name, description, sequence_index)
                VALUES (:subject_id, :code, :name, :description, 1)
                RETURNING id
                """
            ),
            {"subject_id": subject_id, "code": code, "name": name, "description": description},
        ).scalar_one()
    )



def create_question(
    session: Session,
    subject_id: str,
    topic_id: str | None,
    author_id: str,
    code: str,
    stem: str,
    question_type_code: str,
    max_mark: float,
    rubric_json: dict[str, Any] | None,
) -> str:
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.questions (
                    subject_id, topic_id, author_id, code, stem, question_type_code,
                    max_mark, rubric_json, is_active
                ) VALUES (
                    :subject_id, :topic_id, :author_id, :code, :stem, :question_type_code,
                    :max_mark, CAST(:rubric_json AS jsonb), TRUE
                )
                RETURNING id
                """
            ),
            {
                "subject_id": subject_id,
                "topic_id": topic_id,
                "author_id": author_id,
                "code": code,
                "stem": stem,
                "question_type_code": question_type_code,
                "max_mark": max_mark,
                "rubric_json": json.dumps(rubric_json) if rubric_json is not None else None,
            },
        ).scalar_one()
    )



def create_marking_scheme(session: Session, question_id: str, total_mark: float, rubric_items: list[dict[str, Any]]) -> str:
    marking_scheme_id = str(
        session.execute(
            text(
                """
                INSERT INTO lms.marking_schemes (question_id, version, total_mark, scheme_source, is_active)
                VALUES (:question_id, 1, :total_mark, 'dev-bootstrap', TRUE)
                RETURNING id
                """
            ),
            {"question_id": question_id, "total_mark": total_mark},
        ).scalar_one()
    )
    for idx, item in enumerate(rubric_items, start=1):
        session.execute(
            text(
                """
                INSERT INTO lms.marking_scheme_items (marking_scheme_id, step_index, description, mark_value, rubric_code)
                VALUES (:marking_scheme_id, :step_index, :description, :mark_value, :rubric_code)
                """
            ),
            {
                "marking_scheme_id": marking_scheme_id,
                "step_index": idx,
                "description": item.get("description") or f"Rubric item {idx}",
                "mark_value": float(item.get("mark_value") or 1.0),
                "rubric_code": item.get("rubric_code") or f"R{idx}",
            },
        )
    return marking_scheme_id



def create_assessment(
    session: Session,
    school_id: str,
    subject_id: str,
    created_by: str,
    name: str,
    description: str,
    assessment_type: str,
    max_score: float,
    weight_pct: float,
) -> str:
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.assessments (
                    school_id, subject_id, name, description, assessment_type,
                    visibility, max_score, weight_pct, is_ai_enhanced, status,
                    created_by, last_modified_by
                ) VALUES (
                    :school_id, :subject_id, :name, :description, :assessment_type,
                    'private', :max_score, :weight_pct, TRUE, 'published',
                    :created_by, :created_by
                )
                RETURNING id
                """
            ),
            {
                "school_id": school_id,
                "subject_id": subject_id,
                "name": name,
                "description": description,
                "assessment_type": assessment_type,
                "max_score": max_score,
                "weight_pct": weight_pct,
                "created_by": created_by,
            },
        ).scalar_one()
    )



def create_assessment_question(
    session: Session,
    assessment_id: str,
    question_id: str,
    points: float,
    rubric_scheme_id: str | None,
) -> str:
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.assessment_questions (
                    assessment_id, question_id, sequence_index, points, rubric_scheme_id, rubric_scheme_version
                ) VALUES (
                    :assessment_id, :question_id, 1, :points, :rubric_scheme_id, :rubric_scheme_version
                )
                RETURNING id
                """
            ),
            {
                "assessment_id": assessment_id,
                "question_id": question_id,
                "points": points,
                "rubric_scheme_id": rubric_scheme_id,
                "rubric_scheme_version": 1 if rubric_scheme_id else None,
            },
        ).scalar_one()
    )



def create_assessment_assignment(
    session: Session,
    assessment_id: str,
    assigned_by: str,
    title: str,
    instructions: str,
    due_days: int,
) -> str:
    start_time = datetime.now(timezone.utc) - timedelta(days=1)
    due_time = datetime.now(timezone.utc) + timedelta(days=due_days)
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.assessment_assignments (
                    assessment_id, assigned_by, title, instructions, start_time, due_time, is_published
                ) VALUES (
                    :assessment_id, :assigned_by, :title, :instructions, :start_time, :due_time, TRUE
                )
                RETURNING id
                """
            ),
            {
                "assessment_id": assessment_id,
                "assigned_by": assigned_by,
                "title": title,
                "instructions": instructions,
                "start_time": start_time,
                "due_time": due_time,
            },
        ).scalar_one()
    )



def get_or_create_assessment_enrollment(session: Session, assignment_id: str, student_id: str) -> str:
    row = session.execute(
        text(
            """
            SELECT id FROM lms.assessment_enrollments
            WHERE assessment_assignment_id = :assignment_id AND student_id = :student_id AND deleted_at IS NULL
            LIMIT 1
            """
        ),
        {"assignment_id": assignment_id, "student_id": student_id},
    ).mappings().first()
    if row:
        return str(row["id"])
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.assessment_enrollments (assessment_assignment_id, student_id, status_code)
                VALUES (:assignment_id, :student_id, 'assigned')
                RETURNING id
                """
            ),
            {"assignment_id": assignment_id, "student_id": student_id},
        ).scalar_one()
    )



def create_assessment_attempt(session: Session, enrollment_id: str, max_score: float) -> str:
    next_attempt_number = session.execute(
        text(
            """
            SELECT COALESCE(MAX(attempt_number), 0) + 1
            FROM lms.assessment_attempts
            WHERE assessment_enrollment_id = :enrollment_id AND deleted_at IS NULL
            """
        ),
        {"enrollment_id": enrollment_id},
    ).scalar_one()
    submitted_at = datetime.now(timezone.utc)
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.assessment_attempts (
                    assessment_enrollment_id, attempt_number, submitted_at, max_score, grading_status_code
                ) VALUES (
                    :enrollment_id, :attempt_number, :submitted_at, :max_score, 'pending'
                )
                RETURNING id
                """
            ),
            {
                "enrollment_id": enrollment_id,
                "attempt_number": int(next_attempt_number),
                "submitted_at": submitted_at,
                "max_score": max_score,
            },
        ).scalar_one()
    )



def create_attempt_answer(
    session: Session,
    attempt_id: str,
    assessment_question_id: str,
    student_answer_text: str,
    max_score: float,
) -> str:
    return str(
        session.execute(
            text(
                """
                INSERT INTO lms.attempt_answers (
                    assessment_attempt_id, assessment_question_id, student_answer_text,
                    text_content, max_score, requires_review
                ) VALUES (
                    :attempt_id, :assessment_question_id, :student_answer_text,
                    :student_answer_text, :max_score, FALSE
                )
                RETURNING id
                """
            ),
            {
                "attempt_id": attempt_id,
                "assessment_question_id": assessment_question_id,
                "student_answer_text": student_answer_text,
                "max_score": max_score,
            },
        ).scalar_one()
    )



def build_rubric_json(scenario: str, rubric_items: list[dict[str, Any]], expected_answer: str | None) -> dict[str, Any] | None:
    if scenario == "rubric" and rubric_items:
        return {
            "markingPoints": [
                {
                    "description": item.get("description"),
                    "marks": float(item.get("mark_value") or 1.0),
                    "rubric_code": item.get("rubric_code") or f"R{idx}",
                }
                for idx, item in enumerate(rubric_items, start=1)
            ]
        }
    if scenario == "objective" and expected_answer:
        return {"correctAnswer": expected_answer}
    if scenario == "holistic" and expected_answer:
        return {"expectedAnswer": expected_answer}
    return None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
