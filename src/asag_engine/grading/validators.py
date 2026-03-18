import json

from .schema import HolisticLLMResult, RubricLLMResult


RUBRIC_KEYS = {"items", "missing_points", "feedback_text", "confidence"}
HOLISTIC_KEYS = {"score_awarded", "missing_points", "feedback_text", "confidence", "reason"}


def _extract_json(raw_text: str, expected_keys: set[str]) -> str:
    decoder = json.JSONDecoder()
    dict_candidates: list[tuple[dict, str]] = []

    for idx, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(raw_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            dict_candidates.append((obj, raw_text[idx:idx + end]))

    if not dict_candidates:
        raise ValueError("Model output did not contain a JSON object")

    for obj, raw_obj in reversed(dict_candidates):
        if expected_keys.issubset(obj.keys()):
            return raw_obj

    return dict_candidates[-1][1]


def parse_rubric_grade(raw_text: str) -> RubricLLMResult:
    obj = json.loads(_extract_json(raw_text, RUBRIC_KEYS))
    return RubricLLMResult(**obj)


def parse_holistic_grade(raw_text: str) -> HolisticLLMResult:
    obj = json.loads(_extract_json(raw_text, HOLISTIC_KEYS))
    return HolisticLLMResult(**obj)
