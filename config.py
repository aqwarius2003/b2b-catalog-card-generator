from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

ProviderType = Literal["gemini", "openai_compatible"]
OutputMode = Literal["json_schema", "json_prompt"]


@dataclass(frozen=True)
class ProviderConfig:
    provider: ProviderType
    model: str
    api_key: str
    base_url: str
    output_mode: OutputMode
    timeout_seconds: int = 120
    max_retries: int = 2

    def validate(self, prefix: str = "") -> list[str]:
        errors: list[str] = []
        route_name = f" [{prefix}]" if prefix else ""

        if self.provider not in ("gemini", "openai_compatible"):
            errors.append(f"Неизвестный провайдер{route_name}: {self.provider}. Допустимо: gemini, openai_compatible.")

        if not self.model:
            errors.append(f"Не указана модель{route_name} (LLM_MODEL).")

        if self.provider == "gemini" and not self.api_key:
            errors.append(f"Для провайдера gemini{route_name} обязателен API-ключ (LLM_API_KEY).")

        if self.provider == "openai_compatible" and not self.base_url:
            errors.append(f"Для провайдера openai_compatible{route_name} обязателен базовый URL (LLM_BASE_URL).")

        if self.output_mode not in ("json_schema", "json_prompt"):
            errors.append(f"Недопустимый режим вывода{route_name}: {self.output_mode}. Допустимо: json_schema, json_prompt.")

        if self.timeout_seconds <= 0:
            errors.append(f"Таймаут{route_name} должен быть положительным числом.")

        if self.max_retries < 0:
            errors.append(f"Число повторов{route_name} не может быть отрицательным.")

        return errors


@dataclass(frozen=True)
class AppConfig:
    primary: ProviderConfig
    fallback: ProviderConfig | None
    fallback_enabled: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        primary = ProviderConfig(
            provider=os.getenv("LLM_PROVIDER", "gemini").strip(),  # type: ignore[arg-type]
            model=os.getenv("LLM_MODEL", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=os.getenv("LLM_BASE_URL", "").strip(),
            output_mode=os.getenv("LLM_OUTPUT_MODE", "json_schema").strip(),  # type: ignore[arg-type]
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

        fallback_enabled = os.getenv("LLM_FALLBACK_ENABLED", "false").strip().lower() in ("true", "1", "yes")
        fallback: ProviderConfig | None = None

        if fallback_enabled:
            fallback = ProviderConfig(
                provider=os.getenv("LLM_FALLBACK_PROVIDER", "openai_compatible").strip(),  # type: ignore[arg-type]
                model=os.getenv("LLM_FALLBACK_MODEL", "").strip(),
                api_key=os.getenv("LLM_FALLBACK_API_KEY", "").strip(),
                base_url=os.getenv("LLM_FALLBACK_BASE_URL", "").strip(),
                output_mode=os.getenv("LLM_FALLBACK_OUTPUT_MODE", "json_prompt").strip(),  # type: ignore[arg-type]
                timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            )

        return cls(primary=primary, fallback=fallback, fallback_enabled=fallback_enabled)

    def validate(self) -> list[str]:
        errors = self.primary.validate(prefix="Основной маршрут")
        if self.fallback_enabled and self.fallback:
            errors.extend(self.fallback.validate(prefix="Резервный маршрут (fallback)"))
        return errors
