"""Tests for the stlm CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stlm import cli


def test_cli_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["stlm"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0


def test_cli_train_passes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("model: {}", encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake_train(path: Path) -> None:
        seen["config"] = path

    monkeypatch.setattr("sys.argv", ["stlm", "train", "--config", str(config)])
    monkeypatch.setattr("stlm.train.train", fake_train)
    cli.main()
    assert seen["config"] == config


def test_cli_generate_prints_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"unused")
    seen: dict[str, Any] = {}

    def fake_load(path: Path, device: str = "cpu") -> tuple[object, object]:
        seen["checkpoint"] = path
        seen["device"] = device
        return object(), object()

    def fake_generate(
        model: object,
        tok: object,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
    ) -> str:
        seen["prompt"] = prompt
        seen["max_tokens"] = max_tokens
        seen["temperature"] = temperature
        seen["top_k"] = top_k
        return "hello from generate"

    monkeypatch.setattr(
        "sys.argv",
        [
            "stlm",
            "generate",
            "--checkpoint",
            str(ckpt),
            "--prompt",
            "Once upon",
            "--max-tokens",
            "12",
            "--temperature",
            "0.5",
            "--top-k",
            "7",
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr("stlm.generate.load_checkpoint", fake_load)
    monkeypatch.setattr("stlm.generate.generate", fake_generate)
    cli.main()

    captured = capsys.readouterr()
    assert "hello from generate" in captured.out
    assert seen["checkpoint"] == ckpt
    assert seen["device"] == "cpu"
    assert seen["prompt"] == "Once upon"
    assert seen["max_tokens"] == 12
    assert seen["temperature"] == 0.5
    assert seen["top_k"] == 7


def test_cli_generate_requires_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["stlm", "generate"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0
