"""Validate a machine before the frozen ALPS v1 experiment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from llm_length_prediction.experiment import (
    ExperimentContractError,
    file_sha256,
    load_experiment,
    load_frozen_prompts,
)
from llm_length_prediction.models.plp import plp_head_parameter_count
from llm_length_prediction.plp_experiment import (
    load_plp_base_experiment,
    load_plp_config,
    validate_plp_config,
)
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _version_tuple(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def _validate_model_snapshot(
    model_source: Path,
    experiment: dict[str, Any],
    plp_config: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    if not model_source.is_dir():
        return ["frozen model is not available locally; set MODEL_PATH or download it"]

    required = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    for filename in required:
        if not (model_source / filename).is_file():
            failures.append(f"model directory is missing {filename}")

    config_path = model_source / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("model config.json is not valid JSON")
        else:
            if config.get("model_type") != "qwen2":
                failures.append("model config is not Qwen2/Qwen2.5")
            layer_count = int(config.get("num_hidden_layers", 0))
            if int(experiment["model"]["feature_layer"]) >= layer_count:
                failures.append("frozen feature layer is outside the model layer range")
            if plp_config is not None and int(config.get("hidden_size", 0)) != int(
                plp_config["representation"]["hidden_size"]
            ):
                failures.append("model hidden_size does not match the PLP representation")

    tokenizer_path = model_source / "tokenizer_config.json"
    if tokenizer_path.is_file():
        try:
            tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("tokenizer_config.json is not valid JSON")
        else:
            if not tokenizer_config.get("chat_template"):
                failures.append("tokenizer_config.json has no chat_template")

    index_path = model_source / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(weight_index["weight_map"].values())
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            failures.append("model.safetensors.index.json is invalid")
        else:
            missing_shards = sorted(
                shard for shard in shard_names if not (model_source / shard).is_file()
            )
            if missing_shards:
                failures.append(f"model snapshot is missing {len(missing_shards)} weight shards")

    marker = model_source / ".frozen_revision"
    if not marker.is_file():
        failures.append("model directory is missing .frozen_revision")
    elif marker.read_text(encoding="utf-8").strip() != experiment["model"]["revision"]:
        failures.append(".frozen_revision does not match the experiment SHA")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model", help="Optional local model path override")
    parser.add_argument(
        "--plp-config",
        type=Path,
        help="Also validate a PLP method config and report its worst-case storage budget",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path; defaults to <run_root>/environment/preflight.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_experiment(args.experiment)
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "experiment_id": experiment["experiment_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    plp_config: dict[str, Any] | None = None
    plp_minimum_disk_bytes = 0
    plp_recommended_disk_bytes = 0
    if args.plp_config is not None:
        try:
            plp_config = load_plp_config(args.plp_config)
            plp_experiment, _ = load_plp_base_experiment(plp_config)
            validate_plp_config(plp_config, plp_experiment)
            if plp_experiment["experiment_id"] != experiment["experiment_id"]:
                failures.append("PLP base experiment does not match --experiment")
            hidden_size = int(plp_config["representation"]["hidden_size"])
            input_dim = int(plp_config["representation"]["input_dim"])
            stride = int(plp_config["trace"]["stride"])
            max_new_tokens = int(experiment["generation"]["max_new_tokens"])
            points_per_trace = (
                1
                + max_new_tokens // stride
                + int(max_new_tokens % stride != 0)
            )
            rollout_count = int(experiment["inputs"]["rollout_count"])
            trace_bytes = rollout_count * (
                hidden_size * 4
                + points_per_trace * (hidden_size * 4 + 3 * 4)
                + max_new_tokens * 4
            )
            parameter_count = plp_head_parameter_count(
                input_dim, int(plp_config["prediction_head"]["num_bins"])
            )
            checkpoint_bytes = parameter_count * 4
            plp_minimum_disk_bytes = trace_bytes + checkpoint_bytes + 1024**3
            plp_recommended_disk_bytes = int(
                trace_bytes * 1.25 + checkpoint_bytes + 5 * 1024**3
            )
            report["plp"] = {
                "method_id": plp_config["method_id"],
                "hidden_size": hidden_size,
                "input_dim": input_dim,
                "worst_case_points_per_trace": points_per_trace,
                "worst_case_trace_storage_gib": round(trace_bytes / 1024**3, 3),
                "trainable_parameter_count": parameter_count,
                "estimated_checkpoint_mib": round(checkpoint_bytes / 1024**2, 2),
                "minimum_free_disk_gib": round(plp_minimum_disk_bytes / 1024**3, 2),
                "recommended_free_disk_gib": round(
                    plp_recommended_disk_bytes / 1024**3, 2
                ),
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            failures.append(f"invalid PLP config: {error}")

    if _version_tuple(platform.python_version()) < (3, 10):
        failures.append("Python 3.10 or newer is required")
    try:
        load_frozen_prompts(experiment)
    except ExperimentContractError as error:
        failures.append(str(error))
    prompt_path = Path(experiment["inputs"]["prompt_manifest"])
    if prompt_path.is_file():
        report["prompt_manifest_sha256"] = file_sha256(prompt_path)

    try:
        model_source = Path(resolve_model_source(args.model))
    except FileNotFoundError as error:
        report["model_source"] = args.model or "MODEL_PATH"
        failures.append(str(error))
    else:
        report["model_source"] = str(model_source)
        failures.extend(_validate_model_snapshot(model_source, experiment, plp_config))

    outputs = plp_config["outputs"] if plp_config is not None else experiment["outputs"]
    trace_root = Path(outputs["trace_root"])
    run_root = Path(outputs["run_root"])
    trace_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(trace_root).free
    report["free_disk_gib"] = round(free_bytes / 1024**3, 2)
    if free_bytes < 40 * 1024**3:
        warnings.append("less than 40 GiB free disk remains")
    if plp_minimum_disk_bytes and free_bytes < plp_minimum_disk_bytes:
        failures.append("free disk is below the PLP worst-case minimum budget")
    elif plp_recommended_disk_bytes and free_bytes < plp_recommended_disk_bytes:
        warnings.append("free disk is below the recommended PLP safety budget")

    try:
        import torch
        import transformers
    except ImportError as error:
        failures.append(f"missing runtime dependency: {error.name}")
    else:
        report["torch_version"] = torch.__version__
        report["transformers_version"] = transformers.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if _version_tuple(torch.__version__) < (2, 6):
            failures.append("PyTorch 2.6 or newer is required")
        if _version_tuple(transformers.__version__) < (4, 48):
            failures.append("Transformers 4.48 or newer is required")
        if not torch.cuda.is_available():
            failures.append("CUDA GPU is not available")
        else:
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            report.update(
                {
                    "gpu_name": torch.cuda.get_device_name(device_index),
                    "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
            if not torch.cuda.is_bf16_supported():
                failures.append("GPU does not support BF16")
            if properties.total_memory < 24 * 1024**3:
                warnings.append("GPU has less than 24 GiB memory; the BF16 run may OOM")
            if capability[0] >= 10 and _version_tuple(torch.version.cuda) < (12, 8):
                failures.append(
                    "Blackwell-class GPU requires a PyTorch build with CUDA 12.8 or newer"
                )

    for directory in (trace_root, run_root):
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
    output = args.output or run_root / "environment" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"preflight report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
