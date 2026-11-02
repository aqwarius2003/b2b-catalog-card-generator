from __future__ import annotations

import json
from typing import Any
from models import SpecItem, ValidationResult, ValidationStatus


def parse_json_reject_duplicates(raw_text: str) -> dict[str, Any]:
    def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        seen_stripped: set[str] = set()
        for key, val in pairs:
            if key in result:
                raise ValueError(f"Обнаружен дублирующийся ключ в JSON: '{key}'")
            clean_key = key.strip()
            if clean_key in seen_stripped:
                raise ValueError(f"Обнаружен дублирующийся ключ после удаления пробелов в JSON: '{clean_key}'")
            seen_stripped.add(clean_key)
            result[key] = val
        return result

    try:
        data = json.loads(raw_text, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as err:
        raise ValueError(f"Не удалось прочитать JSON. Проверьте синтаксис: {err.msg} (строка {err.lineno}, колонка {err.colno})") from err

    if not isinstance(data, dict):
        raise ValueError("JSON верхнего уровня должен быть объектом (словарем).")

    return data


def validate_manual_specs_table(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """
    Проверяет строки ручной таблицы характеристик до преобразования в словарь.
    - Пустую строку (без названия и значения) игнорирует.
    - Строку со значением, но без названия, отмечает как ошибку.
    - Повторяющиеся названия строк отклоняет с понятной ошибкой.
    """
    specs_dict: dict[str, Any] = {}
    errors: list[str] = []
    seen_names: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        k = row.get("Характеристика")
        v = row.get("Значение")

        k_str = "" if k is None else str(k).strip()
        v_str = "" if v is None else str(v).strip()

        # Пустая строка без названия и значения игнорируется
        if not k_str and not v_str:
            continue

        # Значение указано, но название пустое
        if not k_str and v_str:
            errors.append(f"Строка {idx}: указано значение '{v_str}', но название характеристики не заполнено.")
            continue

        # Проверка дубликатов названий
        if k_str in seen_names:
            errors.append(f"Строка {idx}: обнаружено дублирование характеристики '{k_str}'.")
            continue

        seen_names.add(k_str)
        specs_dict[k_str] = v_str if v_str else None

    return specs_dict, errors


def validate_product_input(raw_data: dict[str, Any] | str) -> ValidationResult:
    warnings: list[str] = []
    errors: list[str] = []

    # 1. Парсинг строки JSON при необходимости
    if isinstance(raw_data, str):
        try:
            data = parse_json_reject_duplicates(raw_data)
        except ValueError as err:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                errors=[str(err)],
            )
    elif isinstance(raw_data, dict):
        data = dict(raw_data)
    else:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            errors=["Входные данные должны быть JSON-строкой или словарем."],
        )

    # 2. Проверка неизвестных полей верхнего уровня
    allowed_keys = {"name", "manufacturer", "specs", "description"}
    unknown_keys = set(data.keys()) - allowed_keys
    if unknown_keys:
        errors.append(f"Обнаружены неизвестные поля верхнего уровня: {', '.join(sorted(unknown_keys))}")

    # 3. Проверка обязательных полей name и manufacturer
    name_raw = data.get("name")
    if name_raw is None or not isinstance(name_raw, str) or not name_raw.strip():
        errors.append("Заполните обязательное поле 'name' (непустая строка).")
        clean_name = ""
    else:
        clean_name = name_raw.strip()

    mfr_raw = data.get("manufacturer")
    if mfr_raw is None or not isinstance(mfr_raw, str) or not mfr_raw.strip():
        errors.append("Заполните обязательное поле 'manufacturer' (непустая строка).")
        clean_mfr = ""
    else:
        clean_mfr = mfr_raw.strip()

    # 4. Проверка поля description
    if "description" not in data or data.get("description") is None:
        warnings.append("Поле 'description' отсутствует или null; трактуется как пустая строка.")
        clean_description = ""
    elif not isinstance(data["description"], str):
        errors.append("Поле 'description' должно быть строкой.")
        clean_description = ""
    else:
        clean_description = data["description"].strip()
        if not clean_description:
            warnings.append("Исходное описание пустое.")

    # 5. Проверка поля specs
    clean_specs: dict[str, str] = {}
    specs_raw = data.get("specs")

    if "specs" not in data or specs_raw is None:
        warnings.append("Поле 'specs' отсутствует или null; трактуется как пустой словарь.")
    elif not isinstance(specs_raw, dict):
        errors.append("Поле 'specs' должно содержать объект характеристик.")
    else:
        seen_clean_keys: set[str] = set()
        for k, v in specs_raw.items():
            if not isinstance(k, str) or not k.strip():
                errors.append(f"Название характеристики должно быть непустой строкой, получено: {k!r}")
                continue
            clean_k = k.strip()
            if clean_k in seen_clean_keys:
                errors.append(f"Обнаружено дублирование характеристики после удаления пробелов: '{clean_k}'.")
                continue
            seen_clean_keys.add(clean_k)

            if v is None:
                clean_specs[clean_k] = "данных нет"
                warnings.append(f"Характеристика '{clean_k}' имеет значение null и отмечена как «данных нет».")
            elif isinstance(v, str):
                v_clean = v.strip()
                if not v_clean:
                    clean_specs[clean_k] = "данных нет"
                    warnings.append(f"Характеристика '{clean_k}' пуста и отмечена как «данных нет».")
                else:
                    clean_specs[clean_k] = v_clean
            else:
                errors.append(
                    f"Характеристика '{clean_k}' имеет недопустимый тип '{type(v).__name__}'. "
                    f"Значения должны быть строками либо null."
                )

    if errors:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            cleaned_input={
                "name": clean_name,
                "manufacturer": clean_mfr,
                "specs": clean_specs,
                "description": clean_description,
            },
            warnings=warnings,
            errors=errors,
        )

    # 6. Проверка минимума для генерации
    has_known_spec = any(val != "данных нет" for val in clean_specs.values())
    has_description = bool(clean_description)
    if not has_known_spec and not has_description:
        errors.append("Недостаточно данных для генерации: добавьте непустое описание или хотя бы одну характеристику со значением.")
        return ValidationResult(
            status=ValidationStatus.ERROR,
            cleaned_input={
                "name": clean_name,
                "manufacturer": clean_mfr,
                "specs": clean_specs,
                "description": clean_description,
            },
            warnings=warnings,
            errors=errors,
        )

    # 7. Формирование структурированного списка характеристик программно из входа
    specs_list: list[SpecItem] = [SpecItem(name=k, value=v) for k, v in clean_specs.items()]

    cleaned_input = {
        "name": clean_name,
        "manufacturer": clean_mfr,
        "specs": clean_specs,
        "description": clean_description,
    }

    status = ValidationStatus.NEEDS_REVIEW if warnings else ValidationStatus.VALID

    return ValidationResult(
        status=status,
        cleaned_input=cleaned_input,
        specs_list=specs_list,
        warnings=warnings,
        errors=errors,
    )
