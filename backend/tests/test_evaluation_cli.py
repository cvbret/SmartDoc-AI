import json
import sys
from unittest.mock import AsyncMock

import pytest

from scripts import run_rag_evaluation as cli


@pytest.mark.parametrize("complete,expected_code", [(2, 0), (1, 1)])
def test_cli_writes_explicit_output_and_returns_completion_status(monkeypatch, tmp_path, complete, expected_code):
    output = tmp_path / "result.json"
    report = {"metadata": {"configured_sample_count": 2, "completed_sample_count": complete},
              "samples": [{"metrics": {"failed": None, "real_zero": 0.0}}], "summary": {}}
    run = AsyncMock(return_value=report)
    monkeypatch.setattr(cli, "run_evaluation", run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "--mode", "baseline", "--limit", "2",
                                      "--sample", "EVAL-010", "EVAL-019", "--output", str(output)])
    assert cli.main() == expected_code
    run.assert_awaited_once_with("baseline", 2, ["EVAL-010", "EVAL-019"])
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_cli_default_runs_without_limit_and_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluation"])
    run = AsyncMock(return_value={"metadata": {"configured_sample_count": 20, "completed_sample_count": 20},
                                 "samples": [], "summary": {}})
    monkeypatch.setattr(cli, "run_evaluation", run)
    assert cli.main() == 0
    run.assert_awaited_once_with("reranker", None, None)
    assert list(tmp_path.iterdir()) == []


def test_cli_setup_failure_is_sanitized_and_returns_two(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluation"])
    monkeypatch.setattr(cli, "run_evaluation", AsyncMock(side_effect=RuntimeError("dummy-api-key")))
    assert cli.main() == 2
    error = capsys.readouterr().err
    assert "dummy-api-key" not in error
    assert json.loads(error)["stage"] == "run_setup_or_output"


@pytest.mark.parametrize("args", [["--limit", "0"], ["--limit", "-1"], ["--mode", "unknown"]])
def test_cli_rejects_bad_arguments_without_running(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["evaluation", *args])
    run = AsyncMock()
    monkeypatch.setattr(cli, "run_evaluation", run)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    run.assert_not_awaited()
