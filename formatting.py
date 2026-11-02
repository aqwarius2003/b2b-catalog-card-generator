"""
Модуль общих утилит форматирования для B2B товарных карточек.

Используется совместно веб-интерфейсом (app.py) и консольным интерфейсом (cli.py),
не завися от Streamlit.
"""
from __future__ import annotations

from models import ExportPayload


def format_markdown_export(payload: ExportPayload) -> str:
    """Форматирует сгенерированную карточку товара в читаемый Markdown документ."""
    card = payload.card
    meta = payload.metadata
    lines: list[str] = [
        f"# Черновик товарной карточки: {card.seo_title}",
        "",
        "> **Черновик — проверьте перед публикацией**",
        "",
        f"**Статус проверки:** `{payload.status}`  ",
        f"**Дата генерации:** `{meta.generation_timestamp}`  ",
        f"**Модель:** `{meta.provider} / {meta.model}` (запросов: {meta.total_requests}, повторов: {meta.retries_count}, время: {meta.duration_seconds}c)  ",
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


def format_summary_export(payload: ExportPayload) -> str:
    """Форматирует краткую текстовую сводку о сгенерированной карточке."""
    card = payload.card
    meta = payload.metadata
    return (
        f"Статус: {payload.status}\n"
        f"Модель: {meta.provider} / {meta.model} (время: {meta.duration_seconds}с, повторов: {meta.retries_count})\n"
        f"Заголовок ({len(card.seo_title)} симв.): {card.seo_title}\n"
        f"Длина SEO-описания: {len(card.seo_description)} симв.\n"
        f"Поисковых запросов: {len(card.search_queries)}\n"
        f"Характеристик: {len(card.specs_list)}\n"
        f"Предупреждений: {len(payload.warnings)}"
    )

