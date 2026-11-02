from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_OUTPUT_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
    EXIT_WARNINGS_UNCONFIRMED,
    create_cli_parser,
    main,
)


def test_cli_parser_defaults():
    parser = create_cli_parser()
    args = parser.parse_args(["-i", "examples/demo_1.input.json"])
    assert args.input_file == "examples/demo_1.input.json"
    assert args.output_format == "json"
    assert not args.force_incomplete
    assert not args.check_only


@pytest.mark.parametrize("setting", ["LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES"])
@pytest.mark.parametrize("value", ["abc", ""])
def test_cli_rejects_invalid_config(monkeypatch, capsys, setting, value):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv(setting, value)
    argv = ["cli.py", "-i", "examples/product_1.input.json"]
    def fail_if_called(*args, **kwargs):
        pytest.fail("Invalid configuration must not start generation")

    monkeypatch.setattr("cli.GenerationService", fail_if_called)
    monkeypatch.setattr("sys.argv", argv)
    assert main() == EXIT_CONFIG_ERROR
    assert setting in capsys.readouterr().err


def test_cli_check_only_valid(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", "examples/product_1.input.json", "--check-only"])
    code = main()
    assert code == EXIT_SUCCESS


def test_cli_blocked_input(monkeypatch, tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"name": ""}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", str(bad_file), "--check-only"])
    code = main()
    assert code == EXIT_VALIDATION_ERROR


def test_cli_warnings_without_force(monkeypatch):
    # warning_incomplete_data has empty description and empty specs -> triggers warnings
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", "examples/warning_incomplete_data.input.json"])
    code = main()
    assert code == EXIT_WARNINGS_UNCONFIRMED


def test_cli_nonexistent_file(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", "nonexistent_file_12345.json"])
    code = main()
    assert code == EXIT_INPUT_ERROR


def test_cli_directory_as_input(monkeypatch, tmp_path):
    folder = tmp_path / "somedir"
    folder.mkdir()
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", str(folder)])
    code = main()
    assert code == EXIT_INPUT_ERROR


def test_cli_directory_as_output(monkeypatch, tmp_path):
    folder = tmp_path / "outdir"
    folder.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "-i", "examples/product_1.input.json", "-o", str(folder), "--check-only"]
    )
    code = main()
    assert code == EXIT_SUCCESS


def test_cli_no_category_enforcement_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "-i", "examples/product_1.input.json", "--check-only"])
    code = main()
    assert code == EXIT_SUCCESS
