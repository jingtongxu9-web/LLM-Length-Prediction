"""Small, explicit checkpoint helpers for Hybrid v3."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("checkpoint creation requires PyTorch") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("checkpoint loading requires PyTorch") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported Hybrid v3 checkpoint")
    return payload
