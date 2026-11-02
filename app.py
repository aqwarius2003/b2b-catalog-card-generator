from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import streamlit as st

from config import AppConfig
from generation import GenerationService
from models import ExportPayload, SpecItem, ValidationResult, ValidationStatus
from validation import load_category_profiles, validate_product_input

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
    if "manual_specs" not in st.session_state:
        st.session_state.manual_specs = [
            {"Характеристика": "Диапазон измерений", "Значение": "0–1,6 МПа"},
            {"Характеристика": "Класс точности", "Значение": "1,5"},
        ]
    if "input_name" not in st.session_state:
        st.session_state.input_name = ""
    if "input_mfr" not in st.session_state:
        st.session_state.input_mfr = ""
    if "input_desc" not in st.session_state:
        st.session_state.input_desc = ""
    if "json_input_text" not in st.session_state:
        st.session_state.json_input_text = ""
    if "allow_incomplete" not in st.session_state:
        st.session_state.allow_incomplete = False


def load_example(example_name: str) -> None:
    path = EXAMPLES_DIR / f"{example_name}.input.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    st.session_state.input_name = data.get("name", "")
    st.session_state.input_mfr = data.get("manufacturer", "")
    st.session_state.input_desc = data.get("description", "")
    st.session_state.json_input_text = json.dumps(data, ensure_ascii=False, indent=2)

    specs = data.get("specs") or {}
    st.session_state.manual_specs = [
        {"Характеристика": str(k), "Значение": "" if v is None else str(v)}
        for k, v in specs.items()
    ]
    st.session_state.allow_incomplete = False
    st.session_state.card_payload = None
    st.session_state.generation_error = None


def format_markdown_export(payload: ExportPayload) -> str:
    card = payload.card
    meta = payload.metadata
    lines: list[str] = [
        f"# Черновик товарной карточки: {card.seo_title}",
        "",
        "> **Черновик — проверьте перед публикацией**",
        "",
        f"**Статус проверки:** `{payload.status}`  ",
        f"**Дата генерации:** `{meta.generation_timestamp}`  ",
        f"**Модель:** `{meta.provider} / {meta.model}` (повторов: {meta.retries_count}, время: {meta.duration_seconds}c)  ",
    ]

    if meta.fallback_used:
        lines.append("> ⚠️ **Внимание:** Использован резервный маршрут (fallback).")

    lines.extend([
        "",
        "## 1. SEO-заголовок",
        card.seo_title,
        "",
        f"## 2. SEO-описание ({len(card.seo_description)} знаков с пробелами)",
        card.seo_description,
        "",
        "## 3. Характеристики",
        "| Характеристика | Значение |",
        "|---|---|",
    ])

    for spec in card.specs_list:
        lines.append(f"| {spec.name} | {spec.value} |")

    lines.extend([
        "",
        "## 4. Поисковые B2B-запросы",
    ])
    for q in card.search_queries:
        lines.append(f"- {q}")

    lines.extend([
        "",
        "## 5. Commercial-блок",
        card.commercial_block,
    ])

    if payload.warnings:
        lines.extend(["", "## Предупреждения"])
        for w in payload.warnings:
            lines.append(f"- ⚠️ {w}")

    return "\n".join(lines)


def main() -> None:
    init_session_state()

    st.title("⚙️ AI-помощник товарных карточек B2B")
    st.info("📌 **Черновик — проверьте перед публикацией.** Результат генерации требует обязательной верификации контент-менеджером.")

    config = AppConfig.from_env()
    config_errors = config.validate()

    profiles = load_category_profiles()
    cat_options = {"none": "Без категории / Другое"}
    for k, v in profiles.items():
        cat_options[k] = v.get("title", k)

    # Боковая панель
    with st.sidebar:
        st.header("Конфигурация")
        st.markdown(f"**Основной маршрут:** `{config.primary.provider}`")
        st.markdown(f"**Модель:** `{config.primary.model or 'не указана'}`")
        st.markdown(f"**Режим вывода:** `{config.primary.output_mode}`")

        if config.fallback_enabled and config.fallback:
            st.markdown(f"**Fallback маршрут:** `{config.fallback.provider} ({config.fallback.model})`")
        else:
            st.caption("Fallback маршрут: отключен")

        if config_errors:
            st.error("Ошибка конфигурации .env:\n" + "\n".join(f"- {e}" for e in config_errors))

        st.divider()
        st.subheader("Тестовые примеры")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Товар 1 (Манометр)", use_container_width=True):
                load_example("product_1")
                st.rerun()
        with col_ex2:
            if st.button("Товар 2 (Датчик)", use_container_width=True):
                load_example("product_2")
                st.rerun()

    # Выбор категории
    selected_cat_key = st.selectbox(
        "Категория оборудования для проверки полноты характеристик:",
        options=list(cat_options.keys()),
        format_func=lambda x: cat_options[x],
    )
    active_cat = None if selected_cat_key == "none" else selected_cat_key

    # Ввод данных: Ручной или JSON
    tab_manual, tab_json = st.tabs(["📝 Ручной ввод", "📄 Ввод через JSON"])
    current_input_data: dict[str, Any] = {}

    with tab_manual:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            name_val = st.text_input("Название прибора*", value=st.session_state.input_name, key="w_name")
        with col_m2:
            mfr_val = st.text_input("Производитель*", value=st.session_state.input_mfr, key="w_mfr")

        st.caption("Характеристики (редактируйте таблицу или добавляйте новые строки):")
        edited_specs = st.data_editor(
            st.session_state.manual_specs,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Характеристика": st.column_config.TextColumn(required=True),
                "Значение": st.column_config.TextColumn(),
            },
            key="w_specs_editor",
        )

        desc_val = st.text_area(
            "Исходное описание или фрагмент паспорта:",
            value=st.session_state.input_desc,
            height=120,
            key="w_desc",
        )

        specs_dict: dict[str, Any] = {}
        for row in edited_specs:
            k = row.get("Характеристика")
            v = row.get("Значение")
            if k and str(k).strip():
                specs_dict[str(k).strip()] = None if v is None or not str(v).strip() else str(v).strip()

        manual_data = {
            "name": name_val,
            "manufacturer": mfr_val,
            "specs": specs_dict,
            "description": desc_val,
        }

    with tab_json:
        uploaded_file = st.file_uploader("Загрузить файл .json", type=["json"])
        if uploaded_file is not None:
            try:
                uploaded_content = uploaded_file.read().decode("utf-8")
                st.session_state.json_input_text = uploaded_content
            except Exception as err:
                st.error(f"Ошибка чтения файла: {err}")

        json_val = st.text_area(
            "Вставьте JSON-структуру товара:",
            value=st.session_state.json_input_text,
            height=240,
            key="w_json_area",
        )

    # Определяем активный источник ввода
    # Если на вкладке JSON не пусто и отличается от ручного, используем его, иначе ручной
    use_json_tab = bool(json_val.strip()) and (tab_json == st.session_state.get("_active_tab"))
    # Для однозначности в Streamlit смотрим на вкладку или ручной ввод
    raw_payload_to_validate: Any = json_val if json_val.strip() else manual_data

    # Выполняем предварительную валидацию
    val_result: ValidationResult = validate_product_input(
        raw_payload_to_validate,
        category_key=active_cat,
    )

    # Отслеживаем изменение входных данных для сброса подтверждения
    curr_hash = f"{raw_payload_to_validate}_{active_cat}"
    if curr_hash != st.session_state.input_hash:
        st.session_state.input_hash = curr_hash
        st.session_state.allow_incomplete = False

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
            "Продолжить с неполными данными. Отсутствующие характеристики будут отмечены как «данных нет».",
            value=st.session_state.allow_incomplete,
            key="w_allow_incomplete",
        )
        st.session_state.allow_incomplete = allow_inc
        can_generate = allow_inc
    else:
        can_generate = True

    # Кнопка явного запуска генерации
    gen_btn_disabled = not can_generate or bool(config_errors)
    btn_text = "Сгенерировать черновик карточки"

    if st.button(btn_text, disabled=gen_btn_disabled, type="primary"):
        with st.spinner("Генерация карточки товара через LLM..."):
            service = GenerationService(config)
            payload, err = service.generate_card(val_result)
            st.session_state.card_payload = payload
            st.session_state.generation_error = err
            st.session_state.last_generated_hash = curr_hash

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
        col_st2.metric("Символов в SEO-описании", len(card.seo_description))
        col_st3.metric("Время ответа", f"{meta.duration_seconds} c")

        if meta.fallback_used:
            st.warning("⚠️ Использован резервный маршрут (fallback).")

        # 1. SEO Title
        st.subheader("1. SEO-заголовок")
        st.text_input("SEO-заголовок", value=card.seo_title, label_visibility="collapsed", disabled=True)
        st.caption(f"Длина: {len(card.seo_title)} из 70 символов.")

        # 2. SEO Description
        st.subheader("2. SEO-описание")
        desc_len = len(card.seo_description)
        is_desc_ok = 800 <= desc_len <= 1200
        desc_badge = "✅ В пределах нормы (800–1200)" if is_desc_ok else "⚠️ Отклонение от нормы (800–1200)"
        st.markdown(f"**Объем:** `{desc_len}` знаков с пробелами — {desc_badge}")
        st.text_area("SEO-описание", value=card.seo_description, height=180, label_visibility="collapsed", disabled=True)

        # 3. Specs List
        st.subheader("3. Характеристики (сформированы программно)")
        specs_table_data = [{"Характеристика": item.name, "Значение": item.value} for item in card.specs_list]
        st.table(specs_table_data)

        # 4. Search Queries
        st.subheader("4. Поисковые B2B-запросы")
        for q in card.search_queries:
            st.markdown(f"- `{q}`")

        # 5. Commercial Block
        st.subheader("5. Commercial-блок")
        st.info(card.commercial_block)

        # Экспорт в JSON и Markdown
        st.subheader("Экспорт")
        col_exp1, col_exp2 = st.columns(2)

        export_json_str = card_payload.model_dump_json(indent=2)
        export_md_str = format_markdown_export(card_payload)

        with col_exp1:
            st.download_button(
                label="📥 Скачать результат в JSON",
                data=export_json_str,
                file_name="product_card.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_exp2:
            st.download_button(
                label="📥 Скачать результат в Markdown",
                data=export_md_str,
                file_name="product_card.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
