import time

from .prompt import build_holistic_grading_prompt, build_rubric_grading_prompt
from .validators import parse_holistic_grade, parse_rubric_grade


class GradeGenerationError(RuntimeError):
    def __init__(self, message: str, first_raw: str, retry_raw: str | None = None):
        super().__init__(message)
        self.first_raw = first_raw
        self.retry_raw = retry_raw


class GraderOutput(dict):
    pass


def _preview(raw_text: str | None, limit: int = 240) -> str:
    if not raw_text:
        return ""
    compact = " ".join(raw_text.split())
    return compact[:limit]


def _run_with_retry(system_text: str, user_text: str, llm_client, parser, retry_on_fail: bool = True):
    started = time.perf_counter()
    raw = llm_client.generate(system_text, user_text)
    try:
        parsed = parser(raw)
    except Exception as first_exc:
        print(f"[grade] first pass invalid output: {first_exc}; preview={_preview(raw)}")
        if not retry_on_fail:
            raise GradeGenerationError(
                "Model returned invalid grading JSON on the first pass.",
                first_raw=raw,
            ) from first_exc
        retry_system = system_text + "\n\nReturn ONLY valid JSON matching the schema. No extra words."
        raw2 = llm_client.generate(retry_system, user_text)
        try:
            parsed = parser(raw2)
            raw = raw2
        except Exception as retry_exc:
            print(f"[grade] retry invalid output: {retry_exc}; preview={_preview(raw2)}")
            raise GradeGenerationError(
                "Model returned invalid grading JSON after retry.",
                first_raw=raw,
                retry_raw=raw2,
            ) from retry_exc
    elapsed = time.perf_counter() - started
    return parsed, raw, elapsed


def grade_with_rubric(
    question_text: str,
    max_score: float,
    rubric_items,
    student_answer: str,
    llm_client,
    subject_name: str | None = None,
    retry_on_fail: bool = True,
):
    system_text, user_text = build_rubric_grading_prompt(
        question_text,
        max_score,
        rubric_items,
        student_answer,
        subject_name=subject_name,
    )
    parsed, raw, elapsed = _run_with_retry(system_text, user_text, llm_client, parse_rubric_grade, retry_on_fail)
    parsed.confidence = max(0.0, min(float(parsed.confidence), 1.0))
    print(f"[grade] rubric grading completed seconds={elapsed:.2f} items={len(parsed.items)}")
    return parsed, raw, elapsed, system_text, user_text


def grade_holistically(
    question_text: str,
    max_score: float,
    student_answer: str,
    llm_client,
    subject_name: str | None = None,
    expected_answer: str | None = None,
    expected_points: list[str] | None = None,
    retry_on_fail: bool = True,
):
    system_text, user_text = build_holistic_grading_prompt(
        question_text,
        max_score,
        student_answer,
        subject_name=subject_name,
        expected_answer=expected_answer,
        expected_points=expected_points,
    )
    parsed, raw, elapsed = _run_with_retry(system_text, user_text, llm_client, parse_holistic_grade, retry_on_fail)
    parsed.score_awarded = max(0.0, min(float(parsed.score_awarded), float(max_score)))
    parsed.confidence = max(0.0, min(float(parsed.confidence), 1.0))
    print(f"[grade] holistic grading completed seconds={elapsed:.2f} score_awarded={parsed.score_awarded}")
    return parsed, raw, elapsed, system_text, user_text
