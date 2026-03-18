from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

import mindspore as ms
import numpy as np
from sqlalchemy.orm import Session

from asag_engine.db.shared_models import (
    AiInferenceRun,
    AiModel,
    AiModelVersion,
    LmsInteractionEvent,
    LmsMasterySnapshot,
    LmsMasterySnapshotSkill,
    LmsSchoolUser,
    LmsSkill,
    LmsStudentAttribute,
    LmsSubject,
    LmsUser,
)

from .model import DKTNetLSTM
from .schema import DktMasteryResponse, DktUpdateRequest, DktUpdateResponse, DktWeakSkill


class DktConfigurationError(RuntimeError):
    pass


class DktValidationError(ValueError):
    pass


class DktNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class DktArtifacts:
    model: DKTNetLSTM
    ckpt_path: Path
    skill_map_path: Path
    model_meta_path: Path
    skill_to_idx: dict[str, int]
    idx_to_skill: dict[int, str]
    num_skills: int
    max_len: int
    model_name: str
    model_version: str
    metrics: dict
    config: dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(env_key: str, filename: str) -> Path:
    configured = os.getenv(env_key)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.exists():
            return path
        raise DktConfigurationError(f"Configured path for {env_key} does not exist: {path}")

    candidates = [
        _project_root() / "models" / "dkt" / filename,
        _project_root().parent / "msmodels" / "workspace" / "dkt" / "update" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise DktConfigurationError(f"Could not find DKT artifact '{filename}'. Checked: {', '.join(str(p) for p in candidates)}")


@lru_cache(maxsize=1)
def load_dkt_artifacts() -> DktArtifacts:
    skill_map_path = _resolve_artifact_path("DKT_SKILL_MAP_PATH", "skill_map_v1.json")
    model_meta_path = _resolve_artifact_path("DKT_MODEL_META_PATH", "model_meta.json")
    ckpt_path = _resolve_artifact_path("DKT_CLOUD_CKPT_PATH", "dkt_lstm_cloud.ckpt")

    meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    skill_map = json.loads(skill_map_path.read_text(encoding="utf-8"))
    skill_to_idx = dict(skill_map.get("skill_code_to_idx") or {})
    if not skill_to_idx:
        raise DktConfigurationError(f"No skill_code_to_idx mapping found in {skill_map_path}")

    num_skills = int(skill_map.get("num_skills") or len(skill_to_idx))
    idx_to_skill = {int(idx): code for code, idx in skill_to_idx.items()}
    architecture = meta.get("architecture") or {}
    embed_dim = int(architecture.get("embedding_dim", 96))
    hidden_dim = int(architecture.get("hidden_dim", 192))
    lstm_layers = int(architecture.get("lstm_layers", 2))
    lstm_dropout = float(architecture.get("lstm_dropout", 0.2))
    max_len = int(os.getenv("DKT_MAX_LEN", str(meta.get("max_len", 50))))

    model = DKTNetLSTM(
        num_skills=num_skills,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=lstm_layers,
        dropout=lstm_dropout,
    )
    param_dict = ms.load_checkpoint(str(ckpt_path))
    ms.load_param_into_net(model, param_dict, strict_load=False)
    model.set_train(False)

    model_version = str(meta.get("saved_at_utc") or meta.get("skill_map_version") or "v1")
    model_name = os.getenv("DKT_MODEL_NAME", "DKT Computer Science Form 3-4")
    config = {
        "skill_map_version": skill_map.get("mapping_version"),
        "num_skills": num_skills,
        "max_len": max_len,
        "checkpoint": str(ckpt_path),
        "skill_map": str(skill_map_path),
        "model_meta": str(model_meta_path),
        "architecture": architecture,
    }

    return DktArtifacts(
        model=model,
        ckpt_path=ckpt_path,
        skill_map_path=skill_map_path,
        model_meta_path=model_meta_path,
        skill_to_idx=skill_to_idx,
        idx_to_skill=idx_to_skill,
        num_skills=num_skills,
        max_len=max_len,
        model_name=model_name,
        model_version=model_version,
        metrics=meta.get("metrics") or {},
        config=config,
    )


def _encode_event(skill_idx: int, is_correct: int, num_skills: int) -> int:
    return (skill_idx + 1) + (num_skills if int(is_correct) == 1 else 0)


def _infer_mastery(artifacts: DktArtifacts, tokens: list[int]) -> np.ndarray:
    if not tokens:
        return np.zeros(artifacts.num_skills, dtype=np.float32)
    trimmed = tokens[-artifacts.max_len :]
    x = np.asarray([trimmed], dtype=np.int32)
    probs = artifacts.model(ms.Tensor(x)).asnumpy()[0]
    return probs[-1]


def _risk_level(average_mastery: float) -> str:
    if average_mastery < 0.45:
        return "high"
    if average_mastery < 0.65:
        return "medium"
    return "low"


def _parse_uuid(value: str | UUID | None) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_student(session: Session, student_id: str) -> LmsUser:
    uuid_value = _parse_uuid(student_id)
    student = None
    if uuid_value is not None:
        student = session.get(LmsUser, uuid_value)
    if student is None:
        student = session.query(LmsUser).filter(LmsUser.external_id == student_id).one_or_none()
    if student is None:
        raise DktNotFoundError(f"Student not found: {student_id}")
    return student


def _resolve_subject(session: Session, *, subject_id: UUID | None, subject_code: str | None, default_subject_code: str | None) -> LmsSubject:
    subject = None
    if subject_id is not None:
        subject = session.get(LmsSubject, subject_id)
    elif subject_code:
        subject = session.query(LmsSubject).filter(LmsSubject.code == subject_code).one_or_none()
    elif default_subject_code:
        subject = session.query(LmsSubject).filter(LmsSubject.code == default_subject_code).one_or_none()

    if subject is None:
        label = subject_code or (str(subject_id) if subject_id else (default_subject_code or "<unspecified>"))
        raise DktNotFoundError(f"Subject not found: {label}")
    return subject


def _resolve_school_id(session: Session, student_id: UUID, requested_school_id: UUID | None) -> UUID | None:
    if requested_school_id is not None:
        return requested_school_id
    school_user = (
        session.query(LmsSchoolUser)
        .filter(
            LmsSchoolUser.user_id == student_id,
            LmsSchoolUser.is_active.is_(True),
            LmsSchoolUser.deleted_at.is_(None),
        )
        .order_by(LmsSchoolUser.id.asc())
        .first()
    )
    return school_user.school_id if school_user else None


def _load_subject_skills(session: Session, subject_id: UUID) -> dict[str, LmsSkill]:
    skills = (
        session.query(LmsSkill)
        .filter(LmsSkill.subject_id == subject_id, LmsSkill.deleted_at.is_(None))
        .all()
    )
    return {skill.code: skill for skill in skills if skill.code}


def _load_history_tokens(session: Session, student_id: UUID, subject_id: UUID, artifacts: DktArtifacts) -> list[int]:
    events = (
        session.query(LmsInteractionEvent)
        .join(LmsSkill, LmsSkill.id == LmsInteractionEvent.skill_id)
        .filter(
            LmsInteractionEvent.student_id == student_id,
            LmsInteractionEvent.subject_id == subject_id,
            LmsInteractionEvent.deleted_at.is_(None),
            LmsSkill.deleted_at.is_(None),
        )
        .order_by(LmsInteractionEvent.event_time.asc(), LmsInteractionEvent.created_at.asc())
        .all()
    )
    tokens: list[int] = []
    for event in events:
        skill_code = event.skill.code if event.skill else None
        if not skill_code:
            continue
        skill_idx = artifacts.skill_to_idx.get(skill_code)
        if skill_idx is None:
            continue
        tokens.append(_encode_event(skill_idx, event.is_correct, artifacts.num_skills))
    return tokens


def _build_response(
    *,
    artifacts: DktArtifacts,
    student_id: UUID,
    subject: LmsSubject,
    mastery: np.ndarray,
    trace_id: str,
    persisted: bool,
    events_applied: int,
    ignored_skill_codes: list[str],
    snapshot_id: UUID | None,
    snapshot_time: datetime,
    include_mastery_vector: bool,
    skill_records_by_code: dict[str, LmsSkill],
) -> DktUpdateResponse:
    avg = float(np.mean(mastery)) if len(mastery) else 0.0
    risk_level = _risk_level(avg)
    weak_limit = max(1, int(os.getenv("DKT_WEAK_SKILL_LIMIT", "5")))
    ranked = sorted(((artifacts.idx_to_skill[idx], float(prob)) for idx, prob in enumerate(mastery)), key=lambda item: item[1])
    weak_skills = []
    for code, prob in ranked[:weak_limit]:
        skill = skill_records_by_code.get(code)
        weak_skills.append(
            DktWeakSkill(
                skill_code=code,
                mastery_prob=round(prob, 6),
                skill_name=skill.name if skill else None,
                skill_id=skill.id if skill else None,
            )
        )

    mastery_vector = None
    if include_mastery_vector:
        mastery_vector = {
            artifacts.idx_to_skill[idx]: round(float(prob), 6)
            for idx, prob in enumerate(mastery)
        }

    return DktUpdateResponse(
        student_id=student_id,
        subject_id=subject.id,
        subject_code=subject.code,
        average_mastery=round(avg, 6),
        risk_level=risk_level,
        weak_skills=weak_skills,
        mastery_vector=mastery_vector,
        snapshot_id=snapshot_id,
        trace_id=trace_id,
        persisted=persisted,
        events_applied=events_applied,
        ignored_skill_codes=ignored_skill_codes,
        snapshot_time=snapshot_time,
        model_name=artifacts.model_name,
        model_version=artifacts.model_version,
    )


def _ensure_model_version(session: Session, artifacts: DktArtifacts) -> AiModelVersion:
    model = (
        session.query(AiModel)
        .filter(AiModel.name == artifacts.model_name, AiModel.model_type == "dkt", AiModel.deleted_at.is_(None))
        .order_by(AiModel.is_active.desc())
        .first()
    )
    if model is None:
        model = AiModel(
            name=artifacts.model_name,
            model_type="dkt",
            description="Computer Science Form 3-4 DKT LSTM model",
            is_active=True,
        )
        session.add(model)
        session.flush()

    version = (
        session.query(AiModelVersion)
        .filter(
            AiModelVersion.model_id == model.id,
            AiModelVersion.version == artifacts.model_version,
            AiModelVersion.deleted_at.is_(None),
        )
        .order_by(AiModelVersion.is_active.desc())
        .first()
    )
    if version is None:
        version = AiModelVersion(
            model_id=model.id,
            version=artifacts.model_version,
            artifact_uri=str(artifacts.ckpt_path),
            metrics=artifacts.metrics,
            config=artifacts.config,
            is_active=True,
        )
        session.add(version)
        session.flush()
    return version


def _persist_snapshot(
    session: Session,
    *,
    student_id: UUID,
    subject: LmsSubject,
    mastery: np.ndarray,
    skill_records_by_code: dict[str, LmsSkill],
    snapshot_time: datetime,
) -> LmsMasterySnapshot:
    snapshot = LmsMasterySnapshot(
        student_id=student_id,
        subject_id=subject.id,
        snapshot_time=snapshot_time,
        source="dkt_update",
        average_mastery=float(np.mean(mastery)) if len(mastery) else 0.0,
        risk_level_code=_risk_level(float(np.mean(mastery)) if len(mastery) else 0.0),
    )
    session.add(snapshot)
    session.flush()

    for code, skill in skill_records_by_code.items():
        idx = load_dkt_artifacts().skill_to_idx.get(code)
        if idx is None:
            continue
        session.add(
            LmsMasterySnapshotSkill(
                mastery_snapshot_id=snapshot.id,
                skill_id=skill.id,
                mastery_prob=float(mastery[idx]),
            )
        )
    return snapshot


def _persist_student_attributes(
    session: Session,
    *,
    student_id: UUID,
    mastery: np.ndarray,
    skill_records_by_code: dict[str, LmsSkill],
    assessed_at: datetime,
) -> None:
    skill_ids = [skill.id for skill in skill_records_by_code.values()]
    existing = {
        item.skill_id: item
        for item in session.query(LmsStudentAttribute)
        .filter(
            LmsStudentAttribute.student_id == student_id,
            LmsStudentAttribute.skill_id.in_(skill_ids),
            LmsStudentAttribute.deleted_at.is_(None),
        )
        .all()
    }

    artifacts = load_dkt_artifacts()
    for code, skill in skill_records_by_code.items():
        idx = artifacts.skill_to_idx.get(code)
        if idx is None:
            continue
        current_score = round(float(mastery[idx]) * 100.0, 2)
        attribute = existing.get(skill.id)
        if attribute is None:
            attribute = LmsStudentAttribute(
                student_id=student_id,
                skill_id=skill.id,
                current_score=current_score,
                potential_score=current_score,
                last_assessed=assessed_at,
            )
            session.add(attribute)
            continue
        attribute.current_score = current_score
        attribute.last_assessed = assessed_at
        if attribute.potential_score is None:
            attribute.potential_score = current_score


def update_student_mastery(session: Session, payload: DktUpdateRequest) -> DktUpdateResponse:
    artifacts = load_dkt_artifacts()
    student = _resolve_student(session, payload.student_id)
    default_subject_code = os.getenv("DKT_DEFAULT_SUBJECT_CODE", "computer_science")
    subject = _resolve_subject(
        session,
        subject_id=payload.subject_id,
        subject_code=payload.subject_code,
        default_subject_code=default_subject_code,
    )
    school_id = _resolve_school_id(session, student.id, payload.school_id)
    if payload.persist and school_id is None:
        raise DktValidationError("school_id could not be resolved for this student; provide school_id explicitly.")

    skill_records_by_code = _load_subject_skills(session, subject.id)
    ignored_skill_codes: list[str] = []
    valid_events = []
    for event in sorted(payload.events, key=lambda item: item.event_time):
        if event.skill_code not in artifacts.skill_to_idx:
            ignored_skill_codes.append(event.skill_code)
            continue
        if payload.persist and event.skill_code not in skill_records_by_code:
            raise DktNotFoundError(
                f"Skill code not found in lms.skills for subject '{subject.code}': {event.skill_code}"
            )
        valid_events.append(event)

    if not valid_events:
        raise DktValidationError("No valid DKT events were supplied after filtering unknown skills.")

    tokens = _load_history_tokens(session, student.id, subject.id, artifacts)
    for event in valid_events:
        skill_idx = artifacts.skill_to_idx[event.skill_code]
        tokens.append(_encode_event(skill_idx, event.is_correct, artifacts.num_skills))

    started = time.perf_counter()
    mastery = _infer_mastery(artifacts, tokens)
    latency_ms = int((time.perf_counter() - started) * 1000)
    trace_id = f"dkt-{uuid4().hex[:16]}"
    snapshot_time = _utcnow()
    response = _build_response(
        artifacts=artifacts,
        student_id=student.id,
        subject=subject,
        mastery=mastery,
        trace_id=trace_id,
        persisted=payload.persist,
        events_applied=len(valid_events),
        ignored_skill_codes=ignored_skill_codes,
        snapshot_id=None,
        snapshot_time=snapshot_time,
        include_mastery_vector=payload.include_mastery_vector,
        skill_records_by_code=skill_records_by_code,
    )

    if not payload.persist:
        return response

    for event in valid_events:
        session.add(
            LmsInteractionEvent(
                school_id=school_id,
                student_id=student.id,
                subject_id=subject.id,
                skill_id=skill_records_by_code[event.skill_code].id,
                assessment_attempt_id=event.assessment_attempt_id,
                attempt_answer_id=event.attempt_answer_id,
                is_correct=event.is_correct,
                score=event.score,
                max_score=event.max_score,
                event_time=_normalize_timestamp(event.event_time),
                trace_id=trace_id,
                created_at=snapshot_time,
            )
        )

    snapshot = _persist_snapshot(
        session,
        student_id=student.id,
        subject=subject,
        mastery=mastery,
        skill_records_by_code=skill_records_by_code,
        snapshot_time=snapshot_time,
    )
    _persist_student_attributes(
        session,
        student_id=student.id,
        mastery=mastery,
        skill_records_by_code=skill_records_by_code,
        assessed_at=snapshot_time,
    )

    model_version = _ensure_model_version(session, artifacts)
    final_response = response.model_copy(update={"snapshot_id": snapshot.id, "persisted": True})
    session.add(
        AiInferenceRun(
            trace_id=trace_id,
            model_version_id=model_version.id,
            school_id=school_id,
            student_id=student.id,
            assessment_attempt_id=valid_events[-1].assessment_attempt_id,
            attempt_answer_id=valid_events[-1].attempt_answer_id,
            prompt_text=None,
            context_json={
                "subject_code": subject.code,
                "artifact": str(artifacts.ckpt_path),
                "skill_map": str(artifacts.skill_map_path),
            },
            rubric_scheme_id=None,
            rubric_scheme_version=None,
            request_json=payload.model_dump(mode="json"),
            response_json=final_response.model_dump(mode="json"),
            latency_ms=latency_ms,
            created_at=snapshot_time,
        )
    )

    return final_response


def get_student_mastery(
    session: Session,
    *,
    student_id: str,
    subject_id: UUID | None = None,
    subject_code: str | None = None,
    include_mastery_vector: bool = False,
    refresh: bool = False,
) -> DktMasteryResponse:
    artifacts = load_dkt_artifacts()
    student = _resolve_student(session, student_id)
    default_subject_code = os.getenv("DKT_DEFAULT_SUBJECT_CODE", "computer_science")
    subject = _resolve_subject(
        session,
        subject_id=subject_id,
        subject_code=subject_code,
        default_subject_code=default_subject_code,
    )

    latest_snapshot = None
    if not refresh:
        latest_snapshot = (
            session.query(LmsMasterySnapshot)
            .filter(
                LmsMasterySnapshot.student_id == student.id,
                LmsMasterySnapshot.subject_id == subject.id,
                LmsMasterySnapshot.deleted_at.is_(None),
            )
            .order_by(LmsMasterySnapshot.snapshot_time.desc())
            .first()
        )

    if latest_snapshot is not None:
        rows = (
            session.query(LmsMasterySnapshotSkill, LmsSkill)
            .join(LmsSkill, LmsSkill.id == LmsMasterySnapshotSkill.skill_id)
            .filter(LmsMasterySnapshotSkill.mastery_snapshot_id == latest_snapshot.id)
            .all()
        )
        mastery_vector = None
        if include_mastery_vector:
            mastery_vector = {skill.code: round(float(row.mastery_prob), 6) for row, skill in rows if skill.code}
        weak_skills = [
            DktWeakSkill(
                skill_code=skill.code,
                mastery_prob=round(float(row.mastery_prob), 6),
                skill_name=skill.name,
                skill_id=skill.id,
            )
            for row, skill in sorted(rows, key=lambda item: item[0].mastery_prob)[: max(1, int(os.getenv("DKT_WEAK_SKILL_LIMIT", "5")))]
            if skill.code
        ]
        return DktMasteryResponse(
            student_id=student.id,
            subject_id=subject.id,
            subject_code=subject.code,
            average_mastery=round(float(latest_snapshot.average_mastery or 0.0), 6),
            risk_level=latest_snapshot.risk_level_code or _risk_level(float(latest_snapshot.average_mastery or 0.0)),
            weak_skills=weak_skills,
            mastery_vector=mastery_vector,
            snapshot_id=latest_snapshot.id,
            snapshot_time=latest_snapshot.snapshot_time,
            source=latest_snapshot.source,
            trace_id=None,
        )

    tokens = _load_history_tokens(session, student.id, subject.id, artifacts)
    if not tokens:
        raise DktNotFoundError(f"No mastery snapshot or interaction history found for student {student_id} and subject {subject.code}")

    mastery = _infer_mastery(artifacts, tokens)
    skill_records_by_code = _load_subject_skills(session, subject.id)
    weak_limit = max(1, int(os.getenv("DKT_WEAK_SKILL_LIMIT", "5")))
    ranked = sorted(((artifacts.idx_to_skill[idx], float(prob)) for idx, prob in enumerate(mastery)), key=lambda item: item[1])
    weak_skills = []
    for code, prob in ranked[:weak_limit]:
        skill = skill_records_by_code.get(code)
        weak_skills.append(
            DktWeakSkill(
                skill_code=code,
                mastery_prob=round(prob, 6),
                skill_name=skill.name if skill else None,
                skill_id=skill.id if skill else None,
            )
        )

    mastery_vector = None
    if include_mastery_vector:
        mastery_vector = {
            artifacts.idx_to_skill[idx]: round(float(prob), 6)
            for idx, prob in enumerate(mastery)
        }

    snapshot_time = _utcnow()
    avg = float(np.mean(mastery)) if len(mastery) else 0.0
    return DktMasteryResponse(
        student_id=student.id,
        subject_id=subject.id,
        subject_code=subject.code,
        average_mastery=round(avg, 6),
        risk_level=_risk_level(avg),
        weak_skills=weak_skills,
        mastery_vector=mastery_vector,
        snapshot_id=None,
        snapshot_time=snapshot_time,
        source="computed_from_history",
        trace_id=None,
    )
