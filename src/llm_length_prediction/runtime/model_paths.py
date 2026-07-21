from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LOCAL_MODEL_PATH = Path("models/Qwen2.5-7B-Instruct")
MODEL_PATH_ENV = "MODEL_PATH"


def is_local_model_directory(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file()


def resolve_model_source(
    requested: str | None = None,
    *,
    local_path: str | Path = DEFAULT_LOCAL_MODEL_PATH,
) -> str:
    if requested:
        return requested

    environment_path = os.environ.get(MODEL_PATH_ENV)
    if environment_path:
        if not is_local_model_directory(environment_path):
            raise FileNotFoundError(
                f"{MODEL_PATH_ENV} does not point to a model directory: {environment_path}"
            )
        return environment_path

    if is_local_model_directory(local_path):
        return str(Path(local_path))
    return DEFAULT_MODEL_ID
