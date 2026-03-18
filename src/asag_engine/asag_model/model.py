from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
}


class AsagConfigurationError(RuntimeError):
    pass


class AsagInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class AsagArtifacts:
    model_dir: Path
    ckpt_path: Path
    mindir_path: Path
    vocab_path: Path
    results_path: Path
    max_length: int
    hidden_size: int
    dropout_prob: float
    raw_score_max: float

    @classmethod
    def from_env(cls) -> "AsagArtifacts":
        default_model_dir = Path(__file__).resolve().parents[3] / "models" / "asag"
        model_dir = Path(os.getenv("ASAG_MODEL_DIR", default_model_dir))
        ckpt_path = Path(os.getenv("ASAG_CKPT_PATH", model_dir / "asag_mohler_best.ckpt"))
        mindir_path = Path(os.getenv("ASAG_MINDIR_PATH", model_dir / "asag_mohler.mindir"))
        vocab_path = Path(os.getenv("ASAG_VOCAB_PATH", model_dir / "tokenizer" / "vocab.json"))
        results_path = Path(os.getenv("ASAG_RESULTS_PATH", model_dir / "results.json"))
        max_length = int(os.getenv("ASAG_MAX_LENGTH", "256"))
        hidden_size = int(os.getenv("ASAG_HIDDEN_SIZE", "256"))
        dropout_prob = float(os.getenv("ASAG_DROPOUT_PROB", "0.2"))
        raw_score_max = float(os.getenv("ASAG_RAW_SCORE_MAX", "5.0"))
        return cls(
            model_dir=model_dir,
            ckpt_path=ckpt_path,
            mindir_path=mindir_path,
            vocab_path=vocab_path,
            results_path=results_path,
            max_length=max_length,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            raw_score_max=raw_score_max,
        )


class SimpleBertLikeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.pad_token_id = vocab[SPECIAL_TOKENS["pad"]]
        self.unk_token_id = vocab[SPECIAL_TOKENS["unk"]]
        self.cls_token_id = vocab[SPECIAL_TOKENS["cls"]]
        self.sep_token_id = vocab[SPECIAL_TOKENS["sep"]]
        self.vocab_size = len(vocab)

    @classmethod
    def from_vocab_path(cls, vocab_path: Path) -> "SimpleBertLikeTokenizer":
        try:
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AsagConfigurationError(f"ASAG vocab file not found: {vocab_path}") from exc
        except PermissionError as exc:
            raise AsagConfigurationError(
                f"ASAG vocab file is not readable: {vocab_path}. Fix file permissions or copy the artifacts into models/asag/."
            ) from exc
        except json.JSONDecodeError as exc:
            raise AsagConfigurationError(f"ASAG vocab file is invalid JSON: {vocab_path}") from exc
        return cls({str(k): int(v) for k, v in vocab.items()})

    @staticmethod
    def _basic_tokenize(text: str) -> list[str]:
        return str(text or "").lower().strip().split()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in self._basic_tokenize(text)]
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        return token_ids


def truncate_three_sequences(a: list[int], b: list[int], c: list[int], max_total: int) -> tuple[list[int], list[int], list[int]]:
    while len(a) + len(b) + len(c) > max_total:
        if len(a) >= len(b) and len(a) >= len(c) and a:
            a.pop()
        elif len(b) >= len(a) and len(b) >= len(c) and b:
            b.pop()
        elif c:
            c.pop()
        else:
            break
    return a, b, c


def encode_triplet(
    question: str,
    reference_answer: str,
    student_answer: str,
    tokenizer: SimpleBertLikeTokenizer,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    r_ids = tokenizer.encode(reference_answer, add_special_tokens=False)
    s_ids = tokenizer.encode(student_answer, add_special_tokens=False)
    max_payload = max_length - 4
    q_ids, r_ids, s_ids = truncate_three_sequences(q_ids, r_ids, s_ids, max_payload)

    input_ids = [tokenizer.cls_token_id] + q_ids + [tokenizer.sep_token_id] + r_ids + [tokenizer.sep_token_id] + s_ids + [tokenizer.sep_token_id]
    token_type_ids = [0] * (1 + len(q_ids) + 1) + [1] * (len(r_ids) + 1) + [0] * (len(s_ids) + 1)
    attention_mask = [1] * len(input_ids)

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    return (
        np.array(input_ids[:max_length], dtype=np.int32),
        np.array(attention_mask[:max_length], dtype=np.int32),
        np.array(token_type_ids[:max_length], dtype=np.int32),
    )


class FallbackEncoderFactory:
    @staticmethod
    def build(vocab_size: int, hidden_size: int):
        import mindspore.nn as nn

        class FallbackEncoder(nn.Cell):
            def __init__(self, vocab_size: int, hidden_size: int = 256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size=vocab_size, embedding_size=hidden_size, padding_idx=0)
                self.proj = nn.Dense(hidden_size, hidden_size)
                self.tanh = nn.Tanh()

            def construct(self, input_ids, attention_mask=None, token_type_ids=None):
                x = self.embedding(input_ids)
                x = self.tanh(self.proj(x))
                return (x,)

        return FallbackEncoder(vocab_size=vocab_size, hidden_size=hidden_size)


class ASAGRegressorFactory:
    @staticmethod
    def build(encoder, hidden_size: int, dropout_prob: float):
        import mindspore as ms
        import mindspore.nn as nn
        import mindspore.ops as ops

        class ASAGRegressor(nn.Cell):
            def __init__(self, encoder, hidden_size: int, dropout_prob: float = 0.2):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(keep_prob=1.0 - dropout_prob)
                self.regressor = nn.Dense(hidden_size, 1)
                self.cast = ops.Cast()
                self.expand_dims = ops.ExpandDims()
                self.reduce_sum = ops.ReduceSum(keep_dims=False)
                self.clip_by_value = ops.clip_by_value
                self.squeeze = ops.Squeeze(axis=-1)

            def masked_mean_pool(self, token_embeddings, attention_mask):
                mask = self.cast(attention_mask, ms.float32)
                mask = self.expand_dims(mask, -1)
                masked_embeddings = token_embeddings * mask
                summed = self.reduce_sum(masked_embeddings, 1)
                counts = self.reduce_sum(mask, 1)
                counts = self.clip_by_value(counts, ms.Tensor(1e-6, ms.float32), ms.Tensor(1e6, ms.float32))
                return summed / counts

            def construct(self, input_ids, attention_mask, token_type_ids):
                encoder_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
                token_embeddings = encoder_outputs[0]
                pooled = self.masked_mean_pool(token_embeddings, attention_mask)
                pooled = self.dropout(pooled)
                score = self.regressor(pooled)
                return self.squeeze(score)

        return ASAGRegressor(encoder=encoder, hidden_size=hidden_size, dropout_prob=dropout_prob)


@dataclass
class AsagScore:
    raw_score_0_5: float
    normalized_score: float
    scaled_score: float
    band: str
    confidence: float
    backend: str
    artifact_path: str
    max_score: float


class AsagScorer:
    def __init__(self, artifacts: AsagArtifacts):
        self.artifacts = artifacts
        self.tokenizer = SimpleBertLikeTokenizer.from_vocab_path(artifacts.vocab_path)
        self._runner = self._load_runner()

    def _assert_readable(self, path: Path, label: str) -> None:
        if not path.exists():
            raise AsagConfigurationError(f"ASAG {label} file not found: {path}")
        if not os.access(path, os.R_OK):
            raise AsagConfigurationError(
                f"ASAG {label} file is not readable: {path}. Fix file permissions or copy the artifacts into models/asag/."
            )

    def _load_runner(self):
        import mindspore as ms
        import mindspore.nn as nn

        if self.artifacts.ckpt_path.exists() and os.access(self.artifacts.ckpt_path, os.R_OK):
            encoder = FallbackEncoderFactory.build(self.tokenizer.vocab_size, self.artifacts.hidden_size)
            model = ASAGRegressorFactory.build(encoder, self.artifacts.hidden_size, self.artifacts.dropout_prob)
            try:
                params = ms.load_checkpoint(str(self.artifacts.ckpt_path))
                not_loaded = ms.load_param_into_net(model, params)
            except Exception as exc:
                raise AsagConfigurationError(f"Failed to load ASAG checkpoint {self.artifacts.ckpt_path}: {exc}") from exc
            if not_loaded:
                raise AsagConfigurationError(f"ASAG checkpoint did not fully load: {not_loaded}")
            model.set_train(False)
            return ("checkpoint", str(self.artifacts.ckpt_path), model)

        if self.artifacts.mindir_path.exists() and os.access(self.artifacts.mindir_path, os.R_OK):
            try:
                graph = ms.load(str(self.artifacts.mindir_path))
                model = nn.GraphCell(graph)
            except Exception as exc:
                raise AsagConfigurationError(f"Failed to load ASAG MindIR {self.artifacts.mindir_path}: {exc}") from exc
            return ("mindir", str(self.artifacts.mindir_path), model)

        ckpt_exists = self.artifacts.ckpt_path.exists()
        mindir_exists = self.artifacts.mindir_path.exists()
        if ckpt_exists or mindir_exists:
            unreadable = []
            if ckpt_exists:
                unreadable.append(str(self.artifacts.ckpt_path))
            if mindir_exists:
                unreadable.append(str(self.artifacts.mindir_path))
            raise AsagConfigurationError(
                "ASAG model artifacts exist but are not readable by the current user: " + ", ".join(unreadable)
            )
        raise AsagConfigurationError(
            f"No ASAG model artifact found. Expected checkpoint at {self.artifacts.ckpt_path} or MindIR at {self.artifacts.mindir_path}."
        )

    def score(self, question: str, reference_answer: str, student_answer: str, max_score: float) -> AsagScore:
        import mindspore as ms

        try:
            input_ids, attention_mask, token_type_ids = encode_triplet(
                question=question,
                reference_answer=reference_answer,
                student_answer=student_answer,
                tokenizer=self.tokenizer,
                max_length=self.artifacts.max_length,
            )
            input_tensor = ms.Tensor(input_ids[None, :], ms.int32)
            mask_tensor = ms.Tensor(attention_mask[None, :], ms.int32)
            type_tensor = ms.Tensor(token_type_ids[None, :], ms.int32)
            backend, model_path, model = self._runner
            raw = float(model(input_tensor, mask_tensor, type_tensor).asnumpy().reshape(-1)[0])
        except AsagConfigurationError:
            raise
        except Exception as exc:
            raise AsagInferenceError(f"ASAG inference failed: {exc}") from exc

        raw_score = min(max(raw, 0.0), self.artifacts.raw_score_max)
        normalized = raw_score / self.artifacts.raw_score_max if self.artifacts.raw_score_max > 0 else 0.0
        scaled = min(max(normalized * max_score, 0.0), max_score)
        confidence = min(max(0.55 + 0.35 * abs(normalized - 0.5) * 2, 0.0), 0.9)
        return AsagScore(
            raw_score_0_5=round(raw_score, 4),
            normalized_score=round(normalized, 4),
            scaled_score=round(scaled, 2),
            band=_score_band(raw_score, self.artifacts.raw_score_max),
            confidence=round(confidence, 4),
            backend=backend,
            artifact_path=model_path,
            max_score=round(max_score, 2),
        )


def _score_band(raw_score: float, raw_score_max: float) -> str:
    ratio = 0.0 if raw_score_max <= 0 else raw_score / raw_score_max
    if ratio < 0.25:
        return "very_weak"
    if ratio < 0.5:
        return "weak"
    if ratio < 0.7:
        return "partial"
    if ratio < 0.85:
        return "good"
    return "strong"


@lru_cache(maxsize=1)
def build_asag_scorer() -> AsagScorer:
    artifacts = AsagArtifacts.from_env()
    return AsagScorer(artifacts)
