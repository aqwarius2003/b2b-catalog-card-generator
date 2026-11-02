from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from config import OutputMode, ProviderConfig

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMAuthError(LLMError):
    pass


class LLMConfigError(LLMError):
    pass


class LLMTransientError(LLMError):
    pass


class LLMResponseCorrectionError(LLMError):
    """Ошибка, когда ответ модели после корректирующей генерации не удовлетворяет требованиям."""
    pass


class BaseLLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.request_count: int = 0

    @abstractmethod
    def _execute_request(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any] | None,
        mode: OutputMode,
    ) -> str:
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any] | None = None,
        mode: OutputMode | None = None,
    ) -> tuple[str, int]:
        effective_mode = mode or self.config.output_mode
        attempts = 0
        delay = 1.0

        for attempt in range(self.config.max_retries + 1):
            attempts += 1
            self.request_count += 1
            try:
                result = self._execute_request(prompt, system_prompt, schema, effective_mode)
                return result, attempts
            except (LLMAuthError, LLMConfigError, LLMResponseCorrectionError):
                raise
            except LLMTransientError as err:
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Временная ошибка API (%s/%s), попытка %d из %d: %s. Пауза %.1fc",
                        self.config.provider,
                        self.config.model,
                        attempt + 1,
                        self.config.max_retries + 1,
                        err,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error(
                        "Исчерпано число попыток (%d) для %s/%s: %s",
                        self.config.max_retries + 1,
                        self.config.provider,
                        self.config.model,
                        err,
                    )
                    raise
            except Exception as err:
                logger.error("Непредвиденная ошибка провайдера %s: %s", self.config.provider, err)
                raise

        raise LLMTransientError("Не удалось получить ответ сервиса после допустимого числа попыток.")


def _normalize_schema_for_gemini(schema: Any) -> Any:
    """Адаптирует схему под OpenAPI 3.0 требования Gemini API."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "additional_properties", "$schema", "title"):
            continue
        if k == "type" and isinstance(v, list):
            non_null = [t for t in v if t != "null"]
            if len(non_null) == 1:
                cleaned["type"] = non_null[0]
                cleaned["nullable"] = True
            elif non_null:
                cleaned["type"] = non_null[0]
            else:
                cleaned["type"] = "string"
                cleaned["nullable"] = True
        elif isinstance(v, dict):
            cleaned[k] = _normalize_schema_for_gemini(v)
        elif isinstance(v, list):
            cleaned[k] = [_normalize_schema_for_gemini(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned


def _normalize_schema_for_openai(schema: Any) -> Any:
    """Адаптирует схему под стандартную спецификацию JSON Schema для OpenAI."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("$schema", "title"):
            continue
        if k == "nullable":
            continue
        if isinstance(v, dict):
            cleaned[k] = _normalize_schema_for_openai(v)
        elif isinstance(v, list):
            cleaned[k] = [_normalize_schema_for_openai(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned


class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise LLMConfigError("Для провайдера gemini обязателен API-ключ (LLM_API_KEY).")
        try:
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(
                timeout=int(config.timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self.client = genai.Client(api_key=config.api_key, http_options=http_options)
        except Exception as err:
            raise LLMConfigError(f"Ошибка инициализации клиента Gemini: {err}") from err

    def _execute_request(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any] | None,
        mode: OutputMode,
    ) -> str:
        from google.genai import errors as gerrors
        from google.genai import types

        call_config: dict[str, Any] = {
            "system_instruction": system_prompt,
        }

        if mode == "json_schema" and schema:
            call_config["response_mime_type"] = "application/json"
            call_config["response_schema"] = _normalize_schema_for_gemini(schema)
        elif mode == "json_prompt":
            call_config["response_mime_type"] = "application/json"

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=types.GenerateContentConfig(**call_config),
            )
            if not response.text:
                raise LLMTransientError("Модель Gemini вернула пустой ответ.")
            return response.text
        except gerrors.APIError as err:
            code = getattr(err, "code", None)
            message = str(err)
            if code in (401, 403):
                raise LLMAuthError(f"Ошибка авторизации Gemini API (код {code}). Проверьте LLM_API_KEY.") from err
            if code == 404:
                raise LLMConfigError(f"Модель Gemini '{self.config.model}' не найдена (код 404).") from err
            if code in (429, 500, 502, 503, 504):
                raise LLMTransientError(f"Временная ошибка Gemini API (код {code}): {message}") from err
            raise LLMError(f"Ошибка Gemini API (код {code}): {message}") from err
        except Exception as err:
            err_str = str(err).lower()
            if "timeout" in err_str or "connection" in err_str or "temporarily" in err_str:
                raise LLMTransientError(f"Сетевой сбой при обращении к Gemini: {err}") from err
            raise


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not config.base_url:
            raise LLMConfigError("Для провайдера openai_compatible обязателен базовый URL (LLM_BASE_URL).")
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "no-key-required",
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        except Exception as err:
            raise LLMConfigError(f"Ошибка инициализации OpenAI-совместимого клиента: {err}") from err

    def _execute_request(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict[str, Any] | None,
        mode: OutputMode,
    ) -> str:
        import openai

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        if mode == "json_schema":
            normalized_schema = _normalize_schema_for_openai(schema) if schema else None
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "product_card_response",
                    "schema": normalized_schema,
                },
            }
        elif mode == "json_prompt":
            # В режиме json_prompt схема передается в промпте без response_format
            pass

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                raise LLMTransientError("OpenAI-совместимый сервер вернул пустой ответ.")
            return content
        except (openai.AuthenticationError, openai.PermissionDeniedError) as err:
            raise LLMAuthError(f"Ошибка авторизации ({err.status_code}): проверьте API-ключ.") from err
        except openai.NotFoundError as err:
            raise LLMConfigError(f"Модель '{self.config.model}' или URL '{self.config.base_url}' не найдены (404).") from err
        except openai.BadRequestError as err:
            err_str = str(err)
            if mode == "json_schema" and any(k in err_str.lower() for k in ("response_format", "json_schema", "schema", "unsupported")):
                raise LLMConfigError(
                    f"Сервер не поддерживает структурированный ответ (json_schema): {err}. "
                    f"Рекомендуется установить LLM_OUTPUT_MODE=json_prompt."
                ) from err
            raise LLMError(f"Некорректный запрос к серверу: {err}") from err
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as err:
            raise LLMTransientError(f"Временный сбой обращения к API: {err}") from err
        except Exception as err:
            err_str = str(err).lower()
            if "timeout" in err_str or "connection" in err_str or "refused" in err_str:
                raise LLMTransientError(f"Сетевой сбой при обращении к API: {err}") from err
            raise


def create_provider(config: ProviderConfig) -> BaseLLMProvider:
    if config.provider == "gemini":
        return GeminiProvider(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleProvider(config)
    raise LLMConfigError(f"Неподдерживаемый провайдер: {config.provider}. Допустимо: gemini, openai_compatible.")
