from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from models import ProductInput, SpecItem, ValidationResult, ValidationStatus

DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "category_profiles.json"


def parse_json_reject_duplicates(raw_text: str) -> dict[str, Any]:
    def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, val in pairs:
            if key in result:
                raise ValueError(f"Обнаружен дублирующийся ключ в JSON: '{key}'")
            result[key] = val
        return result

    try:
        data = json.loads(raw_text, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as err:
        raise ValueError(f"Не удалось прочитать JSON. Проверьте синтаксис: {err.msg} (строка {err.lineno}, колонка {err.colno})") from err

    if not isinstance(data, dict):
        raise ValueError("JSON верхнего уровня должен быть объектом (словарем).")

    return data


def load_category_profiles(path: Path | str | None = None) -> dict[str, Any]:
    file_path = Path(path) if path else DEFAULT_PROFILES_PATH
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_product_input(
    raw_data: dict[str, Any] | str,
    category_key: str | None = None,
    profiles_path: Path | str | None = None,
) -> ValidationResult:
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
        for k, v in specs_raw.items():
            if not isinstance(k, str) or not k.strip():
                errors.append(f"Название характеристики должно быть непустой строкой, получено: {k!r}")
                continue
            clean_k = k.strip()

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

    # 7. Формирование списка характеристик и сверка с профилем категории
    profiles = load_category_profiles(profiles_path)
    specs_list: list[SpecItem] = []

    if category_key and category_key in profiles:
        cat_info = profiles[category_key]
        expected_items = cat_info.get("expected_specs", [])

        # Индексируем переданные характеристики по нормализованному ключу для быстрого поиска
        provided_normalized = {k.lower(): (k, v) for k, v in clean_specs.items()}
        matched_input_keys: set[str] = set()

        for expected in expected_items:
            canonical_name = expected["canonical_name"]
            aliases = [canonical_name.lower()] + [a.lower() for a in expected.get("aliases", [])]

            found_key: str | None = None
            found_val: str | None = None

            for alias in aliases:
                if alias in provided_normalized:
                    orig_k, val = provided_normalized[alias]
                    found_key = orig_k
                    found_val = val
                    matched_input_keys.add(orig_k)
                    break

            if found_key is not None and found_val is not None:
                specs_list.append(SpecItem(name=canonical_name, value=found_val))
                if found_val == "данных нет":
                    warnings.append(f"В категории '{cat_info.get('title')}' характеристика '{canonical_name}' не имеет значения.")
            else:
                specs_list.append(SpecItem(name=canonical_name, value="данных нет"))
                warnings.append(f"В категории '{cat_info.get('title')}' отсутствует ожидаемая характеристика: '{canonical_name}'.")

        # Добавляем все остальные переданные характеристики, не вошедшие в профиль
        for k, v in clean_specs.items():
            if k not in matched_input_keys:
                specs_list.append(SpecItem(name=k, value=v))
    else:
        # Без категории — сохраняем все переданные характеристики
        for k, v in clean_specs.items():
            specs_list.append(SpecItem(name=k, value=v))

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
