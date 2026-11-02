from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from config import AppConfig, ProviderConfig
from generation import GenerationService
from models import ValidationResult, ValidationStatus, SpecItem
from providers import BaseLLMProvider, LLMAuthError, LLMTransientError


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

    def _execute_request(self, prompt, system_prompt, schema, mode):
        self.call_count += 1
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


def test_generation_insufficient_data_reason_blocks():
    cfg = AppConfig(
        primary=ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="key",
            base_url="",
            output_mode="json_schema",
            max_retries=1,
        ),
        fallback=None,
        fallback_enabled=False,
    )
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
    provider = DummyProvider(cfg.primary, [resp])
    service = GenerationService(cfg)

    with patch("generation.create_provider", return_value=provider):
        payload, err = service.generate_card(val_res)

    assert payload is None
    assert "недостаточности данных" in err
