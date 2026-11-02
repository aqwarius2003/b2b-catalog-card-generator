from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValidationStatus(str, Enum):
    VALID = "valid"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class SpecItem(BaseModel):
    name: str = Field(..., min_length=1)
    value: str = Field(...)


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    manufacturer: str = Field(..., min_length=1)
    specs: dict[str, Any] = Field(default_factory=dict)
    description: str = Field(default="")

    @field_validator("name", "manufacturer", mode="before")
    @classmethod
    def check_non_empty_str(cls, v: Any) -> str:
        if v is None or not isinstance(v, str) or not v.strip():
            raise ValueError("Поле обязательно для заполнения и не может быть пустым.")
        return v.strip()


class ValidationResult(BaseModel):
    status: ValidationStatus
    cleaned_input: dict[str, Any] = Field(default_factory=dict)
    specs_list: list[SpecItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.status == ValidationStatus.ERROR

    @property
    def requires_confirmation(self) -> bool:
        return self.status == ValidationStatus.NEEDS_REVIEW


class LLMGeneratedCard(BaseModel):
    seo_title: str = Field(default="")
    seo_description: str = Field(default="")
    search_queries: list[str] = Field(default_factory=list)
    commercial_block: str = Field(default="")
    insufficient_data_reason: str | None = None

    @field_validator("seo_title", "seo_description", "commercial_block", mode="before")
    @classmethod
    def strip_text(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return "" if v is None else str(v)

    @field_validator("search_queries", mode="before")
    @classmethod
    def clean_queries(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s and s not in cleaned:
                    cleaned.append(s)
        return cleaned


class ProductCard(BaseModel):
    seo_title: str
    seo_description: str
    specs_list: list[SpecItem]
    search_queries: list[str]
    commercial_block: str


class ExportMetadata(BaseModel):
    provider: str
    model: str
    generation_timestamp: str
    fallback_used: bool = False
    retries_count: int = 0
    duration_seconds: float = 0.0


class ExportPayload(BaseModel):
    status: str
    card: ProductCard
    warnings: list[str]
    metadata: ExportMetadata
