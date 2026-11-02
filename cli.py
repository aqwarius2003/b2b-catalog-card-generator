"""
CLI интерфейс для генерации B2B товарных карточек SoyuzPribor.

Поддерживает аргументы командной строки:
- Входной JSON-файл или строка
- Выходной файл (JSON / Markdown)
- Проверка без обращения к модели (--check-only)
- Явное согласие на продолжение с предупреждениями (--force / -y)
Настройки LLM загружаются из .env или переменных окружения.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Гарантируем вывод в UTF-8 на Windows при перенаправлении/пайпах
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import AppConfig
from formatting import format_markdown_export, format_summary_export
from generation import GenerationService
from models import ExportPayload
from validation import validate_product_input


# Константы кодов завершения процесса
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_WARNINGS_UNCONFIRMED = 4
EXIT_CONFIG_ERROR = 5
EXIT_GENERATION_ERROR = 6
EXIT_EMPTY_RESULT = 7
EXIT_OUTPUT_ERROR = 8


def setup_cli_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Генерация B2B-карточек товаров КИПиА через LLM (CLI режим)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Входные данные
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input",
        dest="input_file",
        type=str,
        help="Путь к входному JSON-файлу с данными прибора (например, examples/demo_1.input.json)",
    )
    input_group.add_argument(
        "--json-str",
        dest="json_str",
        type=str,
        help="Входная строка в формате JSON (например, '{\"name\": \"...\", \"manufacturer\": \"...\"}')",
    )

    # Валидация и подтверждение
    parser.add_argument(
        "-y", "--force", "--proceed-incomplete",
        dest="force_incomplete",
        action="store_true",
        help="Продолжить несмотря на предупреждения о неполных данных",
    )

    # Вывод
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        default=None,
        help="Путь для сохранения результата (.json или .md). Если не указан, выводится в stdout",
    )
    parser.add_argument(
        "-f", "--format",
        dest="output_format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Формат вывода в stdout/файл: json (полный payload), markdown (товарная карточка) или summary (краткая сводка). По умолчанию: json",
    )

    # Служебные флаги
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод логов в stderr",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только выполнить валидацию данных без обращения к модели",
    )

    return parser


def read_input_payload(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """
    Безопасно читает входной JSON из файла или аргумента строки.
    Возвращает (content, error_message).
    """
    if args.json_str is not None:
        return args.json_str, None

    input_path = Path(args.input_file)
    if not input_path.exists():
        return None, f"Входной файл '{args.input_file}' не найден."
    if input_path.is_dir():
        return None, f"Указанный путь '{args.input_file}' является директорией, а не файлом."

    try:
        content = input_path.read_text(encoding="utf-8")
        return content, None
    except PermissionError:
        return None, f"Нет прав доступа для чтения файла '{args.input_file}'."
    except UnicodeDecodeError as err:
        return None, f"Ошибка кодировки при чтении файла '{args.input_file}': {err} (ожидается UTF-8)."
    except OSError as err:
        return None, f"Не удалось прочитать файл '{args.input_file}': {err.strerror or err}"


def write_output_payload(output_path_str: str, text: str) -> str | None:
    """
    Безопасно сохраняет результат в файл.
    Возвращает None при успехе или текст ошибки.
    """
    out_path = Path(output_path_str)
    if out_path.exists() and out_path.is_dir():
        return f"Указанный путь для сохранения '{output_path_str}' является существующей директорией."

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return None
    except PermissionError:
        return f"Нет прав доступа для записи в файл '{output_path_str}'."
    except OSError as err:
        return f"Не удалось сохранить результат в '{output_path_str}': {err.strerror or err}"


def main() -> int:
    parser = create_cli_parser()
    args = parser.parse_args()

    setup_cli_logging(args.verbose)

    # 1. Загрузка входных данных с аккуратной обработкой ошибок ФС
    raw_input, read_err = read_input_payload(args)
    if read_err:
        sys.stderr.write(f"Ошибка [INPUT_ERROR]: {read_err}\n")
        return EXIT_INPUT_ERROR

    # 2. Валидация входных данных
    validation = validate_product_input(raw_input)

    if validation.is_blocked:
        sys.stderr.write("Ошибки валидации входных данных [VALIDATION_ERROR]:\n")
        for err in validation.errors:
            sys.stderr.write(f"  ❌ {err}\n")
        return EXIT_VALIDATION_ERROR

    if validation.warnings:
        sys.stderr.write("Предупреждения валидации [VALIDATION_WARNING]:\n")
        for warn in validation.warnings:
            sys.stderr.write(f"  ⚠️  {warn}\n")

        if not args.force_incomplete and not args.check_only:
            sys.stderr.write(
                "\nОбнаружены предупреждения о неполных данных.\n"
                "Для продолжения генерации используйте ключ --force (или -y): продолжить несмотря на предупреждения.\n"
            )
            return EXIT_WARNINGS_UNCONFIRMED

    if args.check_only:
        sys.stdout.write("Валидация успешно пройдена [STATUS_OK].\n")
        return EXIT_SUCCESS

    # 3. Подготовка конфигурации LLM
    config = AppConfig.from_env()
    config_errors = config.validate()
    if config_errors:
        sys.stderr.write("Ошибка конфигурации LLM [CONFIG_ERROR]:\n")
        for ce in config_errors:
            sys.stderr.write(f"  ❌ {ce}\n")
        return EXIT_CONFIG_ERROR

    # 4. Генерация
    service = GenerationService(config)
    payload, gen_error = service.generate_card(validation)

    if gen_error:
        sys.stderr.write(f"Ошибка генерации LLM [GENERATION_ERROR]: {gen_error}\n")
        return EXIT_GENERATION_ERROR

    if not payload:
        sys.stderr.write("Ошибка [EMPTY_RESULT]: Не удалось получить результат генерации.\n")
        return EXIT_EMPTY_RESULT

    # 5. Вывод результата
    if args.output_format == "markdown":
        output_text = format_markdown_export(payload)
    elif args.output_format == "summary":
        output_text = format_summary_export(payload)
    else:
        output_text = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)

    if args.output_file:
        write_err = write_output_payload(args.output_file, output_text)
        if write_err:
            sys.stderr.write(f"Ошибка сохранения файла [OUTPUT_ERROR]: {write_err}\n")
            return EXIT_OUTPUT_ERROR
        sys.stderr.write(f"Результат сохранен в {args.output_file}\n")
    else:
        sys.stdout.write(output_text + "\n")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
