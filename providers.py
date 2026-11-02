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


class BaseLLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

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
            try:
                result = self._execute_request(prompt, system_prompt, schema, effective_mode)
                return result, attempts
            except (LLMAuthError, LLMConfigError):
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
                raise LLMError(f"Ошибка вызова сервиса {self.config.provider}: {err}") from err

        raise LLMTransientError("Не удалось получить ответ сервиса после допустимого числа попыток.")


class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        from google import genai
        self.client = genai.Client(api_key=config.api_key)

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
            call_config["response_schema"] = schema
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
        from openai import OpenAI
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "no-key-required",
            timeout=config.timeout_seconds,
            max_retries=0,
        )

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
            kwargs["response_format"] = {"type": "json_object"}
        elif mode == "json_prompt":
            kwargs["response_format"] = {"type": "json_object"}

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
            # Некоторые серверы не поддерживают response_format: fallback без него
            if "response_format" in str(err) and "response_format" in kwargs:
                kwargs.pop("response_format")
                try:
                    retry_resp = self.client.chat.completions.create(**kwargs)
                    text = retry_resp.choices[0].message.content
                    if not text:
                        raise LLMTransientError("Сервер вернул пустой ответ при повторе без response_format.")
                    return text
                except Exception as inner_err:
                    raise LLMError(f"Ошибка запроса: {inner_err}") from inner_err
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
    raise LLMConfigError(f"Неподдерживаемый провайдер: {config.provider}")
