"""Validate GPU, model revision, disk, and contracts before the Bayesian pilot."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

from llm_length_prediction.bayesian_pilot import load_bayesian_pilot
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_pilot_v1.json")


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for component in value.split("+")[0].split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pilot, _, records = load_bayesian_pilot(args.config)
    model_source = resolve_model_source(args.model)
    failures = []
    warnings = []
    report: dict[str, object] = {
        "pilot_id": pilot["pilot_id"],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "selected_prompt_count": len(records),
        "expected_rollout_count": pilot["generation"]["expected_rollout_count"],
        "model_source": model_source,
        "model_revision": pilot["model"]["revision"],
        "final_holdout_accessed": False,
    }
    model_path = Path(model_source)
    if model_path.is_dir():
        marker = model_path / ".frozen_revision"
        if not (model_path / "config.json").is_file():
            failures.append("local model snapshot is missing config.json")
        if not marker.is_file():
            failures.append("local model snapshot is missing .frozen_revision")
        elif marker.read_text(encoding="utf-8").strip() != pilot["model"]["revision"]:
            failures.append("local model revision marker does not match the pilot")
        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        if not weight_files:
            failures.append("local model snapshot contains no weight files")
        report["local_model_weight_file_count"] = len(weight_files)
        report["local_model_weight_bytes"] = sum(path.stat().st_size for path in weight_files)
    else:
        warnings.append("model will be resolved from a Hub ID; a local frozen snapshot is safer")

    trace_root = Path(pilot["outputs"]["trace_root"])
    run_root = Path(pilot["outputs"]["run_root"])
    trace_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(trace_root).free
    hidden_size = int(pilot["model"]["hidden_size"])
    max_tokens = int(pilot["generation"]["max_new_tokens"])
    stride = int(pilot["trace"]["stride"])
    maximum_saved_points = 1 + max_tokens // stride + 1
    per_trace_hidden_bytes = (maximum_saved_points + 3) * hidden_size * 4
    raw_signal_bytes = max_tokens * (4 + 4 + 4)
    estimated_pilot_bytes = (
        per_trace_hidden_bytes + raw_signal_bytes
    ) * pilot["generation"]["expected_rollout_count"]
    report.update(
        {
            "free_disk_gib": round(free_bytes / 1024**3, 2),
            "estimated_uncompressed_pilot_gib": round(
                estimated_pilot_bytes / 1024**3,
                3,
            ),
            "maximum_saved_points_per_trace": maximum_saved_points,
        }
    )
    if free_bytes < max(5 * 1024**3, estimated_pilot_bytes * 4):
        failures.append("free disk is below the pilot safety budget")

    try:
        import torch
        import transformers
    except ImportError as error:
        failures.append(f"missing server dependency: {error.name}")
    else:
        report.update(
            {
                "torch_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__),
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if _version_tuple(str(torch.__version__)) < (2, 6):
            failures.append("PyTorch 2.6 or newer is required")
        if _version_tuple(str(transformers.__version__)) < (4, 48):
            failures.append("Transformers 4.48 or newer is required")
        if not torch.cuda.is_available():
            failures.append("CUDA GPU is required for the real Qwen pilot")
        else:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            report.update(
                {
                    "gpu_name": torch.cuda.get_device_name(index),
                    "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
            if not torch.cuda.is_bf16_supported():
                failures.append("GPU does not support BF16")
            if properties.total_memory < 24 * 1024**3:
                failures.append("GPU has less than 24 GiB memory")
            if capability[0] >= 10 and _version_tuple(str(torch.version.cuda)) < (12, 8):
                failures.append("Blackwell GPU requires a CUDA 12.8 or newer PyTorch build")

    for directory in (trace_root, run_root):
        probe = directory / ".bayesian_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            failures.append(f"output directory is not writable: {directory}")
    report["warnings"] = warnings
    report["failures"] = failures
    report["ready"] = not failures
    output = args.output or run_root / "environment" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Bayesian pilot preflight report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
