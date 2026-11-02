from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import streamlit as st

from config import AppConfig
from formatting import format_markdown_export
from generation import GenerationService
from models import ExportPayload, ValidationResult, ValidationStatus
from validation import validate_manual_specs_table, validate_product_input

st.set_page_config(
    page_title="AI-помощник товарных карточек B2B",
    page_icon="⚙️",
    layout="wide",
)

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def init_session_state() -> None:
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = ""
    if "last_generated_hash" not in st.session_state:
        st.session_state.last_generated_hash = ""
    if "card_payload" not in st.session_state:
        st.session_state.card_payload = None
    if "generation_error" not in st.session_state:
        st.session_state.generation_error = None
    if "specs_version" not in st.session_state:
        st.session_state.specs_version = 0
    if "manual_specs" not in st.session_state:
        st.session_state.manual_specs = [
            {"Характеристика": "Диапазон измерений", "Значение": "0–1,6 МПа"},
            {"Характеристика": "Класс точности", "Значение": "1,5"},
        ]
    if "w_name" not in st.session_state:
        st.session_state.w_name = ""
    if "w_mfr" not in st.session_state:
        st.session_state.w_mfr = ""
    if "w_desc" not in st.session_state:
        st.session_state.w_desc = ""
    if "w_json_area" not in st.session_state:
        st.session_state.w_json_area = ""
    if "w_allow_incomplete" not in st.session_state:
        st.session_state.w_allow_incomplete = False


def load_example_data(data: dict[str, Any]) -> None:
    st.session_state.w_name = data.get("name", "")
    st.session_state.w_mfr = data.get("manufacturer", "")
    st.session_state.w_desc = data.get("description", "")
    st.session_state.w_json_area = json.dumps(data, ensure_ascii=False, indent=2)

    specs = data.get("specs") or {}
    st.session_state.manual_specs = [
        {"Характеристика": str(k), "Значение": "" if v is None else str(v)}
        for k, v in specs.items()
    ]
    st.session_state.specs_version += 1
    st.session_state.w_allow_incomplete = False
    st.session_state.card_payload = None
    st.session_state.generation_error = None
    st.session_state.last_generated_hash = ""
    st.session_state.input_hash = ""


def get_example_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not EXAMPLES_DIR.exists():
        return items

    for file_path in sorted(EXAMPLES_DIR.glob("*.json")):
        if file_path.name.endswith(".output.json"):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "name" in data:
                items.append({
                    "file_path": file_path,
                    "filename": file_path.name,
                    "name": str(data.get("name", "")).strip(),
                    "data": data,
                })
        except Exception:
            continue
    return items


def format_button_label(filename: str, name: str, max_chars_per_line: int = 34) -> str:
    import textwrap

    short_fn = filename if len(filename) <= 28 else filename[:25] + "..."
    clean_name = name.strip() or "Без названия"
    lines = textwrap.wrap(clean_name, width=max_chars_per_line)
    if len(lines) == 1:
        title_lines = lines[0]
    elif len(lines) == 2:
        title_lines = f"{lines[0]}\n{lines[1]}"
    else:
        second_line = lines[1]
        if len(second_line) > max_chars_per_line - 3:
            second_line = second_line[:max_chars_per_line - 3] + "..."
        else:
            second_line = second_line + "..."
        title_lines = f"{lines[0]}\n{second_line}"

    return f"📄 {short_fn}\n{title_lines}"


def clear_form() -> None:
    st.session_state.w_name = ""
    st.session_state.w_mfr = ""
    st.session_state.w_desc = ""
    st.session_state.w_json_area = ""
    st.session_state.manual_specs = [
        {"Характеристика": "", "Значение": ""},
    ]
    st.session_state.specs_version += 1
    st.session_state.w_allow_incomplete = False
    st.session_state.card_payload = None
    st.session_state.generation_error = None
    st.session_state.last_generated_hash = ""
    st.session_state.input_hash = ""


def main() -> None:
    init_session_state()

    st.header("⚙️ AI-помощник товарных карточек B2B")

    config = AppConfig.from_env()
    config_errors = config.validate()

    # Боковая панель
    with st.sidebar:
        st.warning(
            "📌 **Черновик — проверьте перед публикацией.**\n\n"
            "Результат генерации требует обязательной верификации контент-менеджером."
        )

        if st.button("🗑️ Очистить форму", width="stretch"):
            clear_form()
            st.rerun()

        st.divider()

        st.subheader("Тестовые примеры")

        st.html("""
        <style>
        [data-testid="stSidebar"] div.stButton > button {
            font-size: 0.8rem !important;
            line-height: 1.3 !important;
            padding: 0.4rem 0.6rem !important;
            text-align: left !important;
            white-space: pre-wrap !important;
            justify-content: flex-start !important;
        }
        </style>
        """)

        example_items = get_example_items()
        if not example_items:
            st.caption("В папке examples нет JSON-файлов.")
        else:
            for item in example_items:
                btn_label = format_button_label(item["filename"], item["name"])
                if st.button(
                    btn_label,
                    key=f"btn_ex_{item['filename']}",
                    help=item["name"],
                    width="stretch",
                ):
                    load_example_data(item["data"])
                    st.rerun()

        st.divider()

        st.caption("⚙️ **Конфигурация**")
        st.caption(
            f"**Основной маршрут:** `{config.primary.provider}`  \n"
            f"**Модель:** `{config.primary.model or 'не указана'}`  \n"
            f"**Режим вывода:** `{config.primary.output_mode}`  \n"
            f"**Fallback маршрут:** `{config.fallback.provider if (config.fallback_enabled and config.fallback) else 'отключен'}`"
        )

        if config_errors:
            st.error("Ошибка конфигурации .env:\n" + "\n".join(f"- {e}" for e in config_errors))

    # Выбор способа ввода данных
    input_mode = st.segmented_control(
        "Способ ввода данных:",
        options=["📝 Ручной ввод", "📄 Ввод через JSON"],
        default="📝 Ручной ввод",
        key="w_input_mode",
    )

    manual_table_errors: list[str] = []

    if input_mode == "📄 Ввод через JSON":
        uploaded_file = st.file_uploader("Загрузить файл .json", type=["json"])
        if uploaded_file is not None:
            try:
                uploaded_content = uploaded_file.read().decode("utf-8")
                st.session_state.w_json_area = uploaded_content
            except Exception as err:
                st.error(f"Ошибка чтения файла: {err}")

        json_val = st.text_area(
            "Вставьте JSON-структуру товара:",
            height=240,
            key="w_json_area",
        )
        raw_payload_to_validate = json_val
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            name_val = st.text_input("Название прибора*", key="w_name")
        with col_m2:
            mfr_val = st.text_input("Производитель*", key="w_mfr")

        st.caption("Характеристики (редактируйте таблицу или добавляйте новые строки):")
        edited_specs = st.data_editor(
            st.session_state.manual_specs,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Характеристика": st.column_config.TextColumn(required=True),
                "Значение": st.column_config.TextColumn(),
            },
            key=f"w_specs_editor_{st.session_state.specs_version}",
        )

        desc_val = st.text_area(
            "Исходное описание или фрагмент паспорта:",
            height=120,
            key="w_desc",
        )

        specs_dict, manual_table_errors = validate_manual_specs_table(edited_specs)

        manual_data = {
            "name": name_val,
            "manufacturer": mfr_val,
            "specs": specs_dict,
            "description": desc_val,
        }
        raw_payload_to_validate = manual_data

    # Выполняем предварительную валидацию
    val_result: ValidationResult = validate_product_input(raw_payload_to_validate)

    # Объединяем ошибки валидации ручной таблицы со списком ошибок валидации
    if manual_table_errors:
        val_result = ValidationResult(
            status=ValidationStatus.ERROR,
            cleaned_input=val_result.cleaned_input,
            specs_list=val_result.specs_list,
            warnings=val_result.warnings,
            errors=manual_table_errors + val_result.errors,
        )

    # Отслеживаем изменение входных данных для сброса подтверждения
    curr_hash = f"{raw_payload_to_validate}_{input_mode}"
    if curr_hash != st.session_state.input_hash:
        st.session_state.input_hash = curr_hash
        st.session_state.w_allow_incomplete = False

    st.subheader("Проверка готовности данных")

    # Отображение ошибок
    if val_result.errors:
        st.error("⛔ **Ошибки во входных данных (генерация заблокирована):**\n" + "\n".join(f"- {e}" for e in val_result.errors))

    # Отображение предупреждений
    if val_result.warnings:
        st.warning("⚠️ **Предупреждения о неполных данных:**\n" + "\n".join(f"- {w}" for w in val_result.warnings))

    if val_result.status == ValidationStatus.VALID:
        st.success("✅ Все предварительные проверки пройдены.")

    # Чекбокс подтверждения при неполных данных
    can_generate = False
    if val_result.is_blocked:
        can_generate = False
    elif val_result.requires_confirmation:
        allow_inc = st.checkbox(
            "Продолжить с неполными данными. Неполные характеристики будут отмечены как «данных нет».",
            key="w_allow_incomplete",
        )
        can_generate = allow_inc
    else:
        can_generate = True

    # Кнопки запуска генерации и сброса
    gen_btn_disabled = not can_generate or bool(config_errors)
    btn_text = "Сгенерировать черновик карточки"

    col_btn1, col_btn2 = st.columns([3, 1], vertical_alignment="center")
    with col_btn1:
        if st.button(btn_text, disabled=gen_btn_disabled, type="primary", width="stretch"):
            with st.spinner("Генерация карточки товара через LLM..."):
                service = GenerationService(config)
                payload, err = service.generate_card(val_result)
                st.session_state.card_payload = payload
                st.session_state.generation_error = err
                st.session_state.last_generated_hash = curr_hash
    with col_btn2:
        if st.button("Сбросить ввод", width="stretch", help="Очистить поля формы"):
            clear_form()
            st.rerun()

    # Вывод сообщения об ошибке генерации, если была
    if st.session_state.generation_error:
        st.error(f"❌ Ошибка генерации: {st.session_state.generation_error}")

    # Предупреждение о неактуальности результата, если вход был изменен
    if st.session_state.card_payload and st.session_state.last_generated_hash != curr_hash:
        st.warning("⚠️ Входные данные были изменены после последней генерации. Нажмите «Сгенерировать черновик карточки» для обновления.")

    # Отображение результата генерации
    if st.session_state.card_payload:
        card_payload: ExportPayload = st.session_state.card_payload
        card = card_payload.card
        meta = card_payload.metadata

        st.divider()
        st.header("Результат генерации")

        col_st1, col_st2, col_st3 = st.columns(3)
        col_st1.metric("Статус карточки", card_payload.status.upper())
        desc_len = len(card.seo_description)
        is_desc_ok = 800 <= desc_len <= 1200
        col_st2.metric(
            "Символов в SEO-описании",
            desc_len,
            delta="В норме (800–1200)" if is_desc_ok else "Вне нормы (800–1200)",
            delta_color="normal" if is_desc_ok else "inverse",
        )
        col_st3.metric("Время ответа", f"{meta.duration_seconds} c")

        if meta.fallback_used:
            st.warning("⚠️ Использован резервный маршрут (fallback).")

        # 1. SEO Title
        st.subheader("1. SEO-заголовок (нередактируемый блок с копированием)")
        st.code(card.seo_title, language=None)
        st.caption(f"Длина: {len(card.seo_title)} из 70 символов.")

        # 2. SEO Description
        st.subheader("2. SEO-описание (нередактируемый блок с копированием)")
        if is_desc_ok:
            st.markdown(
                f"**Объем:** **{desc_len}** знаков с пробелами — "
                f":green[**✅ В пределах нормы (800–1200 символов)**]"
            )
        else:
            st.markdown(
                f"**Объем:** **{desc_len}** знаков с пробелами — "
                f":red[**⚠️ Отклонение от нормы (требуется от 800 до 1200 символов)**]"
            )
        st.code(card.seo_description, language=None)

        # 3. Specs List
        st.subheader("3. Характеристики (сформированы программно из входа)")
        specs_table_data = [{"Характеристика": item.name, "Значение": item.value} for item in card.specs_list]
        st.table(specs_table_data)

        # 4. Search Queries
        st.subheader("4. Поисковые B2B-запросы")
        for q in card.search_queries:
            st.markdown(f"- `{q}`")

        # 5. Commercial Block
        st.subheader("5. Commercial-блок")
        st.info(card.commercial_block)

        # Экспорт в JSON и Markdown и переход к следующему запросу
        st.subheader("Экспорт")
        col_exp1, col_exp2, col_exp3 = st.columns([2, 2, 2])

        export_json_str = card_payload.model_dump_json(indent=2)
        export_md_str = format_markdown_export(card_payload)

        with col_exp1:
            st.download_button(
                label="📥 Скачать JSON",
                data=export_json_str,
                file_name="product_card.json",
                mime="application/json",
                width="stretch",
            )

        with col_exp2:
            st.download_button(
                label="📥 Скачать Markdown",
                data=export_md_str,
                file_name="product_card.md",
                mime="text/markdown",
                width="stretch",
            )

        with col_exp3:
            if st.button("➕ Новая карточка товара", width="stretch", type="secondary", help="Очистить форму и перейти к следующему прибору"):
                clear_form()
                st.rerun()


if __name__ == "__main__":
    main()
