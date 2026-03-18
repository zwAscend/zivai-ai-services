from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class OcrReferenceDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    documentName: str = Field(..., min_length=1)
    markdown: str = ""


class OcrGeneralRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    source: str | None = None
    module: str | None = None
    url: str | None = None
    language: str | None = None
    preferDigitalExtraction: bool = True
    forceOcr: bool = False
    detectDirection: bool = True
    quickMode: bool = False
    characterMode: bool = False
    singleOrientationMode: bool = False
    returnMarkdownResult: bool = True
    maxPages: int | None = Field(default=None, ge=1, le=100)

    @field_validator("source", "module", "url", "language", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator(
        "preferDigitalExtraction",
        "forceOcr",
        "detectDirection",
        "quickMode",
        "characterMode",
        "singleOrientationMode",
        "returnMarkdownResult",
        mode="before",
    )
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_VALUES:
                return True
            if normalized in _FALSE_VALUES:
                return False
        return bool(value)


class OcrDocumentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    documentName: str = Field(..., min_length=1)
    fileFormat: str = Field(..., min_length=1)
    pageNumber: int = Field(..., ge=1)
    totalPages: int = Field(..., ge=1)
    engine: str = Field(..., min_length=1)
    mode: Literal["digital_text", "ocr"]
    fullText: str = ""
    markdown: str = ""
    averageConfidence: float | None = None
    wordsBlockCount: int = Field(default=0, ge=0)
    module: str | None = None
    source: str | None = None
    referenceDocument: OcrReferenceDocument
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcrGeneralResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok"] = "ok"
    request: OcrGeneralRequest
    documents: list[OcrDocumentResult] = Field(default_factory=list)
    combinedText: str = ""
    referenceDocuments: list[OcrReferenceDocument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
