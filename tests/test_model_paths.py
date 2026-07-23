from pathlib import Path

import pytest

from llm_length_prediction.runtime.model_paths import (
    DEFAULT_MODEL_ID,
    resolve_model_source,
)


def test_model_source_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    assert resolve_model_source(local_path=tmp_path / "missing") == DEFAULT_MODEL_ID
    assert resolve_model_source("explicit/model", local_path=tmp_path) == "explicit/model"

    local_model = tmp_path / "Qwen2.5-7B-Instruct"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_source(local_path=local_model) == str(local_model)

    monkeypatch.setenv("MODEL_PATH", str(local_model))
    assert resolve_model_source(local_path=tmp_path / "other") == str(local_model)


def test_invalid_model_path_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    with pytest.raises(FileNotFoundError, match="MODEL_PATH"):
        resolve_model_source()
