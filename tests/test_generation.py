from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from config import AppConfig, ProviderConfig
from generation import GenerationService
from models import ValidationResult, ValidationStatus, SpecItem
from prompts import CARD_RESPONSE_SCHEMA
from providers import (
    BaseLLMProvider,
    GeminiProvider,
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMTransientError,
    OpenAICompatibleProvider,
    _normalize_schema_for_gemini,
    _normalize_schema_for_openai,
)


def _make_sample_card_json(desc_len: int = 900) -> str:
    desc = "Прибор технический " + ("описание " * ((desc_len - 19) // 9))
    desc = desc.ljust(desc_len, ".")
    data = {
        "seo_title": "Манометр ТМ-510Р.00 0-1,6 МПа Физтех",
        "seo_description": desc,
        "search_queries": [
            "манометр ТМ 510 купить",
            "манометр технический 100 мм",
            "манометр физтех 1 6 мпа",
        ],
        "commercial_block": "Подходит для котельных и тепловых пунктов.",
        "insufficient_data_reason": None,
    }
    return json.dumps(data, ensure_ascii=False)


def _make_validation_result() -> ValidationResult:
    return ValidationResult(
        status=ValidationStatus.VALID,
        cleaned_input={
            "name": "Манометр ТМ-510",
            "manufacturer": "Физтех",
            "specs": {"Диапазон": "0–1,6 МПа"},
            "description": "Описание манометра",
        },
        specs_list=[SpecItem(name="Диапазон", value="0–1,6 МПа")],
        warnings=[],
        errors=[],
    )


class DummyProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig, responses: list[Any]) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: list[str] = []

    def _execute_request(self, prompt, system_prompt, schema, mode):
        self.call_count += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise LLMTransientError("Больше нет ответов.")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_generation_preserves_specs_list():
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="secret-key",
            base_url="",
            output_mode="json_schema",
            max_retries=1,
        ),
        fallback=None,
        fallback_enabled=False,
    )
    val_res = _make_validation_result()
    provider = DummyProvider(cfg.primary, [_make_sample_card_json(950)])

    service = GenerationService(cfg)
    with patch("generation.create_provider", return_value=provider):
        payload, err = service.generate_card(val_res)

    assert err is None
    assert payload is not None
    assert payload.card.specs_list[0].name == "Диапазон"
    assert payload.card.specs_list[0].value == "0–1,6 МПа"
    assert payload.status == "ready"
    assert payload.metadata.fallback_used is False
    assert payload.metadata.total_requests == 1
    assert payload.metadata.retries_count == 0


def test_generation_corrective_retry_on_invalid_length():
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="secret-key",
            base_url="",
            output_mode="json_schema",
            max_retries=1,
        ),
        fallback=None,
        fallback_enabled=False,
    )
    val_res = _make_validation_result()

    # 1-й ответ: слишком короткое описание (100 символов)
    bad_card = _make_sample_card_json(100)
    # 2-й ответ: корректное описание (850 символов)
    good_card = _make_sample_card_json(850)

    provider = DummyProvider(cfg.primary, [bad_card, good_card])
    service = GenerationService(cfg)

    with patch("generation.create_provider", return_value=provider):
        payload, err = service.generate_card(val_res)

    assert err is None
    assert payload is not None
    assert len(payload.card.seo_description) == 850
    assert provider.call_count == 2
    assert payload.metadata.total_requests == 2
    assert payload.metadata.retries_count == 1

    # Проверка, что во 2-м запросе (корректирующем) содержатся исходные данные, схема и пометка о недоверенном черновике
    corrective_prompt = provider.prompts[1]
    assert "Манометр ТМ-510" in corrective_prompt
    assert "0–1,6 МПа" in corrective_prompt
    assert "НЕДОВЕРЕННЫЙ ЧЕРНОВИК" in corrective_prompt
    assert "Требуемая схема ответа" in corrective_prompt


def test_generation_retries_transient_error():
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="secret-key",
            base_url="",
            output_mode="json_schema",
            max_retries=2,
        ),
        fallback=None,
        fallback_enabled=False,
    )
    val_res = _make_validation_result()

    provider = DummyProvider(
        cfg.primary,
        [
            LLMTransientError("503 Service Unavailable"),
            LLMTransientError("503 Service Unavailable"),
            _make_sample_card_json(900),
        ],
    )
    service = GenerationService(cfg)

    with patch("generation.create_provider", return_value=provider):
        with patch("time.sleep"):
            payload, err = service.generate_card(val_res)

    assert err is None
    assert payload is not None
    assert provider.call_count == 3
    assert payload.metadata.total_requests == 3
    assert payload.metadata.retries_count == 2


def test_generation_no_retry_on_auth_error():
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="invalid-key",
            base_url="",
            output_mode="json_schema",
            max_retries=2,
        ),
        fallback=None,
        fallback_enabled=False,
    )
    val_res = _make_validation_result()
    provider = DummyProvider(cfg.primary, [LLMAuthError("401 Неверный ключ")])
    service = GenerationService(cfg)

    with patch("generation.create_provider", return_value=provider):
        payload, err = service.generate_card(val_res)

    assert payload is None
    assert "Неверный ключ" in err
    assert provider.call_count == 1


def test_generation_fallback_route_activation():
    primary_cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="key",
        base_url="",
        output_mode="json_schema",
        max_retries=1,
    )
    fallback_cfg = ProviderConfig(
        provider="openai_compatible",
        model="local-model",
        api_key="",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
        max_retries=1,
    )
    cfg = AppConfig(primary=primary_cfg, fallback=fallback_cfg, fallback_enabled=True)
    val_res = _make_validation_result()

    primary_provider = DummyProvider(
        primary_cfg,
        [LLMTransientError("Таймаут основного"), LLMTransientError("Таймаут повторный")],
    )
    fallback_provider = DummyProvider(fallback_cfg, [_make_sample_card_json(900)])

    def fake_create_provider(p_cfg: ProviderConfig):
        if p_cfg.provider == "gemini":
            return primary_provider
        return fallback_provider

    service = GenerationService(cfg)
    with patch("generation.create_provider", side_effect=fake_create_provider):
        with patch("time.sleep"):
            payload, err = service.generate_card(val_res)

    assert err is None
    assert payload is not None
    assert payload.metadata.fallback_used is True
    assert payload.metadata.provider == "openai_compatible"
    assert payload.metadata.model == "local-model"
    assert any("резервный маршрут" in w for w in payload.warnings)


def test_generation_attempts_accounting_with_fallback():
    # Проверка сценария: основной маршрут исчерпал повторы (3 запроса), резервный ответил с 1-й попытки (1 запрос).
    # Итого: total_requests = 4, retries_count = 3. Запросы основного маршрута не теряются.
    primary_cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="key",
        base_url="",
        output_mode="json_schema",
        max_retries=2,
    )
    fallback_cfg = ProviderConfig(
        provider="openai_compatible",
        model="local-model",
        api_key="",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
        max_retries=1,
    )
    cfg = AppConfig(primary=primary_cfg, fallback=fallback_cfg, fallback_enabled=True)
    val_res = _make_validation_result()

    primary_provider = DummyProvider(
        primary_cfg,
        [
            LLMTransientError("Ошибка 1"),
            LLMTransientError("Ошибка 2"),
            LLMTransientError("Ошибка 3"),
        ],
    )
    fallback_provider = DummyProvider(fallback_cfg, [_make_sample_card_json(900)])

    def fake_create_provider(p_cfg: ProviderConfig):
        if p_cfg.provider == "gemini":
            return primary_provider
        return fallback_provider

    service = GenerationService(cfg)
    with patch("generation.create_provider", side_effect=fake_create_provider):
        with patch("time.sleep"):
            payload, err = service.generate_card(val_res)

    assert err is None
    assert payload is not None
    assert payload.metadata.fallback_used is True
    assert payload.metadata.total_requests == 4
    assert payload.metadata.retries_count == 3
    assert primary_provider.request_count == 3
    assert fallback_provider.request_count == 1


def test_generation_no_fallback_on_auth_error():
    primary_cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="key",
        base_url="",
        output_mode="json_schema",
        max_retries=1,
    )
    fallback_cfg = ProviderConfig(
        provider="openai_compatible",
        model="local-model",
        api_key="",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
        max_retries=1,
    )
    cfg = AppConfig(primary=primary_cfg, fallback=fallback_cfg, fallback_enabled=True)
    val_res = _make_validation_result()

    primary_provider = DummyProvider(primary_cfg, [LLMAuthError("401 Invalid Key")])
    fallback_provider = DummyProvider(fallback_cfg, [_make_sample_card_json(900)])

    def fake_create_provider(p_cfg: ProviderConfig):
        if p_cfg.provider == "gemini":
            return primary_provider
        return fallback_provider

    service = GenerationService(cfg)
    with patch("generation.create_provider", side_effect=fake_create_provider):
        payload, err = service.generate_card(val_res)

    assert payload is None
    assert "401 Invalid Key" in err
    assert fallback_provider.call_count == 0


def test_generation_no_fallback_on_insufficient_data():
    primary_cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="key",
        base_url="",
        output_mode="json_schema",
        max_retries=1,
    )
    fallback_cfg = ProviderConfig(
        provider="openai_compatible",
        model="local-model",
        api_key="",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
        max_retries=1,
    )
    cfg = AppConfig(primary=primary_cfg, fallback=fallback_cfg, fallback_enabled=True)
    val_res = _make_validation_result()

    resp = json.dumps(
        {
            "seo_title": "",
            "seo_description": "",
            "search_queries": [],
            "commercial_block": "",
            "insufficient_data_reason": "Недостаточно данных для подготовки описания без домыслов.",
        },
        ensure_ascii=False,
    )
    primary_provider = DummyProvider(primary_cfg, [resp])
    fallback_provider = DummyProvider(fallback_cfg, [_make_sample_card_json(900)])

    def fake_create_provider(p_cfg: ProviderConfig):
        if p_cfg.provider == "gemini":
            return primary_provider
        return fallback_provider

    service = GenerationService(cfg)
    with patch("generation.create_provider", side_effect=fake_create_provider):
        payload, err = service.generate_card(val_res)

    assert payload is None
    assert "недостаточности данных" in err
    assert fallback_provider.call_count == 0


def test_generation_no_fallback_on_unexpected_exception():
    primary_cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="key",
        base_url="",
        output_mode="json_schema",
        max_retries=1,
    )
    fallback_cfg = ProviderConfig(
        provider="openai_compatible",
        model="local-model",
        api_key="",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
        max_retries=1,
    )
    cfg = AppConfig(primary=primary_cfg, fallback=fallback_cfg, fallback_enabled=True)
    val_res = _make_validation_result()

    primary_provider = DummyProvider(primary_cfg, [TypeError("Непредвиденная программная ошибка")])
    fallback_provider = DummyProvider(fallback_cfg, [_make_sample_card_json(900)])

    def fake_create_provider(p_cfg: ProviderConfig):
        if p_cfg.provider == "gemini":
            return primary_provider
        return fallback_provider

    service = GenerationService(cfg)
    with patch("generation.create_provider", side_effect=fake_create_provider):
        payload, err = service.generate_card(val_res)

    assert payload is None
    assert "Внутренняя программная ошибка" in err
    assert fallback_provider.call_count == 0


def test_generation_create_provider_error_handled():
    # Ошибка инициализации провайдера не должна приводить к падению
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="",  # Пустой ключ вызовет ошибку
            base_url="",
            output_mode="json_schema",
        ),
        fallback=None,
        fallback_enabled=False,
    )
    val_res = _make_validation_result()
    service = GenerationService(cfg)
    payload, err = service.generate_card(val_res)

    assert payload is None
    assert "LLM_API_KEY" in err


def test_gemini_provider_timeout_and_retries_options():
    cfg = ProviderConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="test-api-key",
        base_url="",
        output_mode="json_schema",
        timeout_seconds=45,
        max_retries=2,
    )
    provider = GeminiProvider(cfg)
    http_options = provider.client._api_client._http_options
    # Таймаут должен быть переведен в миллисекунды: 45с * 1000 = 45000мс
    assert http_options.timeout == 45000
    # Внутренние повторы SDK должны быть отключены / установлены в 1, чтобы не умножать повторы приложения
    assert http_options.retry_options.attempts == 1


def test_openai_provider_json_schema_mode():
    cfg = ProviderConfig(
        provider="openai_compatible",
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        output_mode="json_schema",
    )
    provider = OpenAICompatibleProvider(cfg)

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"seo_title": "Тест"}'
    mock_resp.choices = [mock_choice]

    captured_kwargs = {}

    def fake_create(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return mock_resp

    with patch.object(provider.client.chat.completions, "create", side_effect=fake_create):
        res = provider._execute_request("Промпт", "Системный", CARD_RESPONSE_SCHEMA, "json_schema")
        assert res == '{"seo_title": "Тест"}'
        assert "response_format" in captured_kwargs
        assert captured_kwargs["response_format"]["type"] == "json_schema"
        assert captured_kwargs["response_format"]["json_schema"]["name"] == "product_card_response"


def test_openai_provider_json_prompt_mode():
    cfg = ProviderConfig(
        provider="openai_compatible",
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        output_mode="json_prompt",
    )
    provider = OpenAICompatibleProvider(cfg)

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"seo_title": "Тест"}'
    mock_resp.choices = [mock_choice]

    captured_kwargs = {}

    def fake_create(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return mock_resp

    with patch.object(provider.client.chat.completions, "create", side_effect=fake_create):
        res = provider._execute_request("Промпт", "Системный", CARD_RESPONSE_SCHEMA, "json_prompt")
        assert res == '{"seo_title": "Тест"}'
        # В режиме json_prompt response_format не должен передаваться
        assert "response_format" not in captured_kwargs


def test_openai_provider_schema_unsupported_recommends_json_prompt():
    import openai
    import httpx

    cfg = ProviderConfig(
        provider="openai_compatible",
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        output_mode="json_schema",
    )
    provider = OpenAICompatibleProvider(cfg)

    request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    bad_req_err = openai.BadRequestError(
        message="response_format type json_schema is not supported",
        response=httpx.Response(400, request=request),
        body={"error": "unsupported"},
    )

    with patch.object(provider.client.chat.completions, "create", side_effect=bad_req_err):
        with pytest.raises(LLMConfigError) as exc_info:
            provider._execute_request("Промпт", "Системный", CARD_RESPONSE_SCHEMA, "json_schema")

        assert "json_prompt" in str(exc_info.value)


def test_gemini_schema_compatibility():
    from google.genai import types

    # Нормализатор Gemini должен очистить универсальную схему от ['string', 'null'] и additionalProperties
    normalized = _normalize_schema_for_gemini(CARD_RESPONSE_SCHEMA)
    assert "additionalProperties" not in normalized
    assert "additional_properties" not in normalized

    # Проверяем, что нормализованная схема принимается google-genai без ошибок
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=normalized,
    )
    assert cfg.response_schema is not None
    assert cfg.response_schema["properties"]["insufficient_data_reason"]["nullable"] is True


def test_config_invalid_numbers_no_crash(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "not_a_number")
    monkeypatch.setenv("LLM_MAX_RETRIES", "")

    cfg = AppConfig.from_env()
    errors = cfg.validate()
    assert any("LLM_TIMEOUT_SECONDS" in e for e in errors)
    assert any("LLM_MAX_RETRIES" in e for e in errors)

    val_res = _make_validation_result()
    service = GenerationService(cfg)
    payload, err = service.generate_card(val_res)
    assert payload is None
    assert "LLM_TIMEOUT_SECONDS" in err
