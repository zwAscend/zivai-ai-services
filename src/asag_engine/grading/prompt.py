import json


RUBRIC_SYSTEM_INSTRUCTIONS = """You are an automated grading assistant for Computer Science short-answer questions.
You will grade ONE student answer against a rubric.
Return JSON ONLY. No markdown. No prose outside JSON.

Rules:
- Evaluate each rubric item independently.
- Award marks only when the student's answer clearly matches that rubric item.
- awarded must be between 0 and the rubric item's max_marks.
- Keep each reason short and concrete.
- feedback_text must explain what the student got right or missed.
- missing_points should contain the main missing ideas.
- confidence must be a number between 0 and 1.

Return exactly this schema:
{
  "items": [{"rubric_index": 1, "awarded": 0, "reason": "..."}],
  "missing_points": ["..."],
  "feedback_text": "...",
  "confidence": 0.0
}
"""


HOLISTIC_SYSTEM_INSTRUCTIONS = """You are an automated grading assistant for Computer Science short-answer questions.
There is no usable rubric for this question, so grade holistically.
Return JSON ONLY. No markdown. No prose outside JSON.

Rules:
- Use the question, student answer, expected answer hints, and max_score.
- If the answer is blank or irrelevant, award 0.
- feedback_text must tell the student what to improve.
- missing_points should list the major missing ideas.
- confidence must be a number between 0 and 1.

Return exactly this schema:
{
  "score_awarded": 0,
  "feedback_text": "...",
  "missing_points": ["..."],
  "confidence": 0.0,
  "reason": "..."
}
"""


def build_rubric_grading_prompt(question_text: str, max_score: float, rubric_items, student_answer: str):
    rubric_payload = [
        {
            "rubric_index": item["rubric_index"],
            "description": item["description"],
            "max_marks": item["max_marks"],
        }
        for item in rubric_items
    ]
    user_obj = {
        "question_text": question_text,
        "max_score": max_score,
        "rubric": rubric_payload,
        "student_answer": student_answer,
    }
    return RUBRIC_SYSTEM_INSTRUCTIONS, json.dumps(user_obj, ensure_ascii=False)


def build_holistic_grading_prompt(
    question_text: str,
    max_score: float,
    student_answer: str,
    expected_answer: str | None = None,
    expected_points: list[str] | None = None,
):
    user_obj = {
        "question_text": question_text,
        "max_score": max_score,
        "student_answer": student_answer,
        "expected_answer": expected_answer,
        "expected_points": expected_points or [],
    }
    return HOLISTIC_SYSTEM_INSTRUCTIONS, json.dumps(user_obj, ensure_ascii=False)
