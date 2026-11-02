from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from config import AppConfig, ProviderConfig
from models import (
    ExportMetadata,
    ExportPayload,
    LLMGeneratedCard,
    ProductCard,
    ValidationResult,
)
from prompts import (
    CARD_RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_corrective_prompt,
    build_user_prompt,
)
from providers import (
    BaseLLMProvider,
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMResponseCorrectionError,
    LLMTransientError,
    create_provider,
)

logger = logging.getLogger(__name__)


def _extract_json_text(raw_text: str) -> str:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _validate_card_content(card: LLMGeneratedCard) -> list[str]:
    violations: list[str] = []

    if card.insufficient_data_reason:
        return []

    title_len = len(card.seo_title)
    if not card.seo_title:
        violations.append("Поле seo_title отсутствует или пусто.")
    elif title_len > 70:
        violations.append(f"Длина seo_title составляет {title_len} символов (максимум 70).")

    desc_len = len(card.seo_description)
    if desc_len < 800 or desc_len > 1200:
        violations.append(
            f"Длина seo_description составляет {desc_len} символов с пробелами (требуется от 800 до 1200 символов)."
        )

    queries_count = len(card.search_queries)
    if queries_count < 3 or queries_count > 5:
        violations.append(f"Количество search_queries: {queries_count} (требуется от 3 до 5 запросов).")

    if not card.commercial_block:
        violations.append("Поле commercial_block отсутствует или пусто.")

    return violations


class GenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _call_and_validate(
        self,
        provider: BaseLLMProvider,
        config: ProviderConfig,
        cleaned_input: dict[str, Any],
    ) -> tuple[LLMGeneratedCard, int]:
        total_attempts = 0

        user_prompt = build_user_prompt(
            cleaned_input,
            include_schema_in_prompt=(config.output_mode == "json_prompt"),
        )

        raw_output, attempts = provider.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            schema=CARD_RESPONSE_SCHEMA,
            mode=config.output_mode,
        )
        total_attempts += attempts

        parsed_card, violations = self._try_parse(raw_output)

        if parsed_card.insufficient_data_reason:
            return parsed_card, total_attempts

        # Ровно одна корректирующая попытка при нарушении схемы или ограничений длины
        if violations:
            logger.warning(
                "Ответ %s/%s требует исправления (%s). Запуск корректирующей генерации.",
                config.provider,
                config.model,
                violations,
            )
            corrective_prompt = build_corrective_prompt(
                cleaned_input=cleaned_input,
                schema=CARD_RESPONSE_SCHEMA,
                previous_output=raw_output,
                violations=violations,
            )
            corr_output, corr_attempts = provider.generate(
                prompt=corrective_prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=CARD_RESPONSE_SCHEMA,
                mode=config.output_mode,
            )
            total_attempts += corr_attempts
            parsed_card, corr_violations = self._try_parse(corr_output)
            if corr_violations:
                raise LLMResponseCorrectionError(
                    f"Модель вернула результат, не соответствующий требованиям: {'; '.join(corr_violations)}"
                )

        return parsed_card, total_attempts

    def _try_parse(self, raw_text: str) -> tuple[LLMGeneratedCard, list[str]]:
        json_str = _extract_json_text(raw_text)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as err:
            return LLMGeneratedCard(
                seo_title="",
                seo_description="",
                search_queries=[],
                commercial_block="",
            ), [f"Ошибка синтаксиса JSON в ответе: {err.msg}"]

        if not isinstance(data, dict):
            return LLMGeneratedCard(
                seo_title="",
                seo_description="",
                search_queries=[],
                commercial_block="",
            ), ["Ответ должен быть объектом JSON."]

        try:
            card = LLMGeneratedCard.model_validate(data)
        except Exception as err:
            return LLMGeneratedCard(
                seo_title="",
                seo_description="",
                search_queries=[],
                commercial_block="",
            ), [f"Несоответствие схеме ответа: {err}"]

        violations = _validate_card_content(card)
        return card, violations

    def generate_card(self, validation_result: ValidationResult) -> tuple[ExportPayload | None, str | None]:
        start_time = time.perf_counter()

        config_errors = self.config.validate()
        if config_errors:
            return None, f"Ошибка конфигурации: {'; '.join(config_errors)}"

        if validation_result.is_blocked:
            return None, "Генерация заблокирована из-за ошибок во входных данных."

        active_config = self.config.primary
        try:
            active_provider = create_provider(active_config)
        except (LLMConfigError, LLMAuthError) as err:
            logger.error("Ошибка инициализации основного провайдера: %s", err)
            return None, str(err)
        except Exception as err:
            logger.error("Непредвиденная ошибка создания основного провайдера: %s", err)
            return None, f"Не удалось инициализировать основной провайдер: {err}"

        fallback_provider: BaseLLMProvider | None = None
        used_fallback = False

        logger.info("Старт генерации: провайдер=%s, модель=%s", active_config.provider, active_config.model)

        generated_card: LLMGeneratedCard | None = None

        try:
            generated_card, _ = self._call_and_validate(
                active_provider, active_config, validation_result.cleaned_input
            )
        except (LLMAuthError, LLMConfigError) as err:
            logger.error("Критическая ошибка конфигурации/авторизации основного маршрута: %s", err)
            return None, str(err)
        except (LLMTransientError, LLMResponseCorrectionError) as primary_err:
            logger.warning("Сбой основного маршрута (%s): %s", active_config.model, primary_err)

            # Проверка возможности переключения на fallback
            if self.config.fallback_enabled and self.config.fallback:
                logger.info(
                    "Переключение на резервный маршрут: %s/%s",
                    self.config.fallback.provider,
                    self.config.fallback.model,
                )
                used_fallback = True
                active_config = self.config.fallback
                try:
                    fallback_provider = create_provider(active_config)
                except (LLMConfigError, LLMAuthError) as fb_init_err:
                    logger.error("Ошибка инициализации резервного провайдера: %s", fb_init_err)
                    return None, f"Ошибка инициализации резервного провайдера: {fb_init_err}"
                except Exception as fb_init_err:
                    logger.error("Непредвиденная ошибка создания резервного провайдера: %s", fb_init_err)
                    return None, f"Не удалось инициализировать резервный провайдер: {fb_init_err}"

                try:
                    generated_card, _ = self._call_and_validate(
                        fallback_provider, active_config, validation_result.cleaned_input
                    )
                except Exception as fb_err:
                    logger.error("Сбой резервного маршрута: %s", fb_err)
                    return None, f"Основной и резервный маршруты завершились ошибкой: {fb_err}"
            else:
                return None, f"Не удалось получить ответ сервиса: {primary_err}"
        except Exception as unexpected_err:
            logger.error("Непредвиденная программная ошибка основного маршрута: %s", unexpected_err, exc_info=True)
            return None, f"Внутренняя программная ошибка: {unexpected_err}"

        duration = round(time.perf_counter() - start_time, 2)

        if not generated_card:
            return None, "Не удалось сформировать карточку товара."

        if generated_card.insufficient_data_reason:
            return None, f"Модель сообщила о недостаточности данных: {generated_card.insufficient_data_reason}"

        # Программная сборка карточки: specs_list берется строго из входных данных
        product_card = ProductCard(
            seo_title=generated_card.seo_title,
            seo_description=generated_card.seo_description,
            specs_list=validation_result.specs_list,
            search_queries=generated_card.search_queries,
            commercial_block=generated_card.commercial_block,
        )

        all_warnings = list(validation_result.warnings)
        if used_fallback:
            all_warnings.append(f"Использован резервный маршрут генерации: {active_config.provider} ({active_config.model}).")

        card_status = "needs_review" if all_warnings else "ready"

        primary_requests = active_provider.request_count
        fallback_requests = fallback_provider.request_count if (used_fallback and fallback_provider) else 0
        total_requests = primary_requests + fallback_requests
        retries_count = max(0, total_requests - 1)

        metadata = ExportMetadata(
            provider=active_config.provider,
            model=active_config.model,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            fallback_used=used_fallback,
            retries_count=retries_count,
            total_requests=total_requests,
            duration_seconds=duration,
        )

        payload = ExportPayload(
            status=card_status,
            card=product_card,
            warnings=all_warnings,
            metadata=metadata,
        )

        logger.info(
            "Генерация завершена за %.2fc. Статус=%s, fallback=%s, запросов=%d (повторов=%d)",
            duration,
            card_status,
            used_fallback,
            total_requests,
            retries_count,
        )
        return payload, None
