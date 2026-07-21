"""Validate an isolated server before the frozen ALPS v1 experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from llm_length_prediction.runtime.model_paths import resolve_model_source

EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    experiment: dict[str, Any] = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "experiment_id": experiment["experiment_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    prompt_path = Path(experiment["inputs"]["prompt_manifest"])
    prompt_hash = _sha256(prompt_path)
    report["prompt_manifest_sha256"] = prompt_hash
    if prompt_hash != experiment["inputs"]["prompt_manifest_sha256"]:
        failures.append("prompt manifest SHA-256 does not match the frozen experiment")

    model_source = Path(resolve_model_source())
    report["model_source"] = str(model_source)
    if not model_source.is_dir():
        failures.append("frozen model is not available locally; set MODEL_PATH or download it")
    else:
        required = ("config.json", "tokenizer_config.json")
        for filename in required:
            if not (model_source / filename).is_file():
                failures.append(f"model directory is missing {filename}")
        if not list(model_source.glob("*.safetensors")):
            failures.append("model directory contains no safetensors weights")
        marker = model_source / ".frozen_revision"
        if not marker.is_file():
            failures.append("model directory is missing .frozen_revision")
        elif marker.read_text(encoding="utf-8").strip() != experiment["model"]["revision"]:
            failures.append(".frozen_revision does not match the experiment SHA")

    free_bytes = shutil.disk_usage(Path.cwd()).free
    report["free_disk_gib"] = round(free_bytes / 1024**3, 2)
    if free_bytes < 40 * 1024**3:
        warnings.append("less than 40 GiB free disk remains")

    try:
        import torch
        import transformers
    except ImportError as error:
        failures.append(f"missing server dependency: {error.name}")
    else:
        report["torch_version"] = torch.__version__
        report["transformers_version"] = transformers.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            failures.append("CUDA GPU is not available")
        else:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            if not torch.cuda.is_bf16_supported():
                failures.append("GPU does not support BF16")
            if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
                warnings.append("GPU has less than 24 GiB memory; the frozen BF16 run may OOM")

    for directory in (
        Path(experiment["outputs"]["trace_root"]),
        Path(experiment["outputs"]["run_root"]),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            failures.append(f"output directory is not writable: {directory}")

    report["warnings"] = warnings
    report["failures"] = failures
    report["ready"] = not failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
