from __future__ import annotations

import json
import pytest
from models import ValidationStatus
from validation import (
    parse_json_reject_duplicates,
    validate_manual_specs_table,
    validate_product_input,
)


def test_parse_json_reject_duplicates_valid():
    raw = '{"name": "Прибор 1", "manufacturer": "Завод", "specs": {"давление": "1 МПа"}}'
    data = parse_json_reject_duplicates(raw)
    assert data["name"] == "Прибор 1"
    assert data["manufacturer"] == "Завод"


def test_parse_json_reject_duplicates_detected():
    raw = '{"name": "Прибор 1", "name": "Дубликат", "manufacturer": "Завод"}'
    with pytest.raises(ValueError, match="дублирующийся ключ"):
        parse_json_reject_duplicates(raw)


def test_parse_json_syntax_error():
    raw = '{"name": "Прибор 1", "manufacturer": "Завод", '
    with pytest.raises(ValueError, match="Не удалось прочитать JSON"):
        parse_json_reject_duplicates(raw)


def test_validate_valid_input():
    data = {
        "name": "Манометр ТМ-510",
        "manufacturer": "Физтех",
        "specs": {"Диапазон": "0-1 МПа", "Класс": "1.5"},
        "description": "Технический манометр для измерения давления воды и пара.",
    }
    result = validate_product_input(data)
    assert result.status == ValidationStatus.VALID
    assert not result.is_blocked
    assert not result.requires_confirmation
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    assert len(result.specs_list) == 2


def test_validate_missing_required_fields():
    data = {"specs": {"Диапазон": "0-1 МПа"}, "description": "Описание"}
    result = validate_product_input(data)
    assert result.is_blocked
    assert any("name" in e for e in result.errors)
    assert any("manufacturer" in e for e in result.errors)


def test_validate_empty_string_name():
    data = {"name": "   ", "manufacturer": "Физтех", "description": "Текст"}
    result = validate_product_input(data)
    assert result.is_blocked
    assert any("name" in e for e in result.errors)


def test_validate_unknown_top_level_keys():
    data = {
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {},
        "description": "Описание",
        "unknown_extra": 123,
    }
    result = validate_product_input(data)
    assert result.is_blocked
    assert any("неизвестные поля" in e for e in result.errors)


def test_validate_invalid_spec_value_type():
    data = {
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {"давление": 10.5},
        "description": "Описание",
    }
    result = validate_product_input(data)
    assert result.is_blocked
    assert any("давление" in e and "недопустимый тип 'float'" in e for e in result.errors)


def test_validate_null_and_empty_specs():
    data = {
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {"Диапазон": None, "Материал": "  ", "Класс": "1.5"},
        "description": "Описание прибора",
    }
    result = validate_product_input(data)
    assert result.status == ValidationStatus.NEEDS_REVIEW
    assert result.requires_confirmation
    specs_dict = {item.name: item.value for item in result.specs_list}
    assert specs_dict["Диапазон"] == "данных нет"
    assert specs_dict["Материал"] == "данных нет"
    assert specs_dict["Класс"] == "1.5"


def test_validate_insufficient_data():
    data = {
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {"Диапазон": None},
        "description": "",
    }
    result = validate_product_input(data)
    assert result.is_blocked
    assert any("Недостаточно данных" in e for e in result.errors)


def test_validate_missing_spec_key_not_added_and_not_error():
    # Если характеристика не передана, она не должна добавляться в список и не считается ошибкой
    data = {
        "name": "Манометр ТМ-510",
        "manufacturer": "Физтех",
        "specs": {"шкала": "0–1,6 МПа"},
        "description": "Описание манометра",
    }
    result = validate_product_input(data)
    assert result.status == ValidationStatus.VALID
    assert len(result.specs_list) == 1
    assert result.specs_list[0].name == "шкала"
    assert result.specs_list[0].value == "0–1,6 МПа"
    assert len(result.warnings) == 0


def test_validate_duplicate_keys_after_strip_in_json():
    raw_json = json.dumps({
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {
            "диапазон": "0–10",
            " диапазон ": "0–20",
        },
        "description": "Описание",
    }, ensure_ascii=False)
    result = validate_product_input(raw_json)
    assert result.status == ValidationStatus.ERROR
    assert any("дублирующийся ключ" in e.lower() for e in result.errors)


def test_validate_duplicate_keys_after_strip_in_dict():
    data = {
        "name": "Манометр",
        "manufacturer": "Физтех",
        "specs": {
            "диапазон": "0–10",
            " диапазон ": "0–20",
        },
        "description": "Описание",
    }
    result = validate_product_input(data)
    assert result.status == ValidationStatus.ERROR
    assert any("дублирование характеристики после удаления пробелов" in e for e in result.errors)


def test_validate_manual_specs_table():
    rows = [
        {"Характеристика": "  ", "Значение": "   "},  # Пустая строка -> игнорируется
        {"Характеристика": "Диапазон", "Значение": "0-10"},
        {"Характеристика": "Класс", "Значение": ""},  # Пустое значение -> разрешено, будет None
    ]
    specs, errors = validate_manual_specs_table(rows)
    assert len(errors) == 0
    assert specs == {"Диапазон": "0-10", "Класс": None}

    # Значение без названия
    rows_val_without_name = [
        {"Характеристика": "", "Значение": "0-10"},
    ]
    _, errs1 = validate_manual_specs_table(rows_val_without_name)
    assert len(errs1) == 1
    assert "название характеристики не заполнено" in errs1[0]

    # Дублирование названий строк
    rows_duplicate = [
        {"Характеристика": "Диапазон", "Значение": "0-10"},
        {"Характеристика": " Диапазон ", "Значение": "0-20"},
    ]
    _, errs2 = validate_manual_specs_table(rows_duplicate)
    assert len(errs2) == 1
    assert "дублирование характеристики" in errs2[0]
