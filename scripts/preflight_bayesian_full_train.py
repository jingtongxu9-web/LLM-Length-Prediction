"""Validate contracts, capacity, GPU, and model before full-Train collection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

from llm_length_prediction.bayesian_full_train import (
    BayesianFullTrainJob,
    bayesian_full_train_jobs,
    load_bayesian_full_train,
    validate_bayesian_full_train_trace,
)
from llm_length_prediction.data.bayesian_trace import (
    bayesian_trace_path,
    read_bayesian_trace,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_full_train_v1.json")
MINIMUM_GPU_MEMORY_GB = 24


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for component in value.split("+")[0].split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _has_minimum_gpu_memory(total_memory_bytes: int) -> bool:
    """Apply the nominal decimal-GB boundary used for advertised GPU memory."""
    return total_memory_bytes >= MINIMUM_GPU_MEMORY_GB * 1000**3


def _trace_path(root: Path, job: BayesianFullTrainJob) -> Path:
    return bayesian_trace_path(
        root,
        split="train",
        prompt_id=job.record["prompt_id"],
        temperature=job.temperature,
        seed=job.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    collection, _, records = load_bayesian_full_train(args.config)
    jobs = bayesian_full_train_jobs(collection, records)
    config_sha256 = file_sha256(args.config)
    model_source = resolve_model_source(args.model)
    failures = []
    warnings = []
    report: dict[str, object] = {
        "collection_id": collection["collection_id"],
        "collection_config_sha256": config_sha256,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "selected_prompt_count": len(records),
        "expected_rollout_count": len(jobs),
        "model_source": model_source,
        "model_revision": collection["model"]["revision"],
        "stage3_pilot_status": "pass",
        "final_holdout_accessed": False,
    }

    model_path = Path(model_source)
    if model_path.is_dir():
        marker = model_path / ".frozen_revision"
        if not (model_path / "config.json").is_file():
            failures.append("local model snapshot is missing config.json")
        if not marker.is_file():
            failures.append("local model snapshot is missing .frozen_revision")
        elif marker.read_text(encoding="utf-8").strip() != collection["model"][
            "revision"
        ]:
            failures.append("local model revision marker does not match full-Train")
        weight_files = list(model_path.glob("*.safetensors")) + list(
            model_path.glob("*.bin")
        )
        if not weight_files:
            failures.append("local model snapshot contains no weight files")
        report["local_model_weight_file_count"] = len(weight_files)
        report["local_model_weight_bytes"] = sum(
            path.stat().st_size for path in weight_files
        )
    else:
        failures.append("full-Train collection requires a local frozen model snapshot")

    trace_root = Path(collection["outputs"]["trace_root"])
    run_root = Path(collection["outputs"]["run_root"])
    trace_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(trace_root)
    expected_paths = {_trace_path(trace_root, job) for job in jobs}
    valid_existing_trace_count = 0
    invalid_existing_traces = []
    for job in jobs:
        path = _trace_path(trace_root, job)
        if not path.is_file():
            continue
        try:
            trace = read_bayesian_trace(path)
            validate_bayesian_full_train_trace(
                trace,
                job=job,
                collection=collection,
                collection_config_sha256=config_sha256,
            )
            valid_existing_trace_count += 1
        except (OSError, KeyError, TypeError, ValueError) as error:
            invalid_existing_traces.append(f"{path}: {error}")
    unexpected_paths = sorted(set(trace_root.rglob("*.npz")).difference(expected_paths))
    invalid_existing_traces.extend(
        f"{path}: unexpected full-Train trace path" for path in unexpected_paths
    )
    existing_trace_file_count = len(set(trace_root.rglob("*.npz")))
    remaining_trace_count = len(jobs) - valid_existing_trace_count
    budget = collection["budget"]
    maximum_per_trace = budget["maximum_uncompressed_trace_bytes"] / len(jobs)
    dynamic_required_bytes = max(
        5 * 1024**3,
        remaining_trace_count
        * maximum_per_trace
        * budget["disk_safety_multiplier"],
    )
    if valid_existing_trace_count == 0:
        dynamic_required_bytes = max(
            dynamic_required_bytes,
            budget["required_free_disk_gib_at_empty_start"] * 1024**3,
        )
    report.update(
        {
            "disk_total_gib": round(disk.total / 1024**3, 2),
            "free_disk_gib": round(disk.free / 1024**3, 2),
            "required_free_disk_gib": round(dynamic_required_bytes / 1024**3, 2),
            "existing_trace_file_count": existing_trace_file_count,
            "valid_existing_trace_count": valid_existing_trace_count,
            "invalid_existing_trace_count": len(invalid_existing_traces),
            "remaining_trace_count": remaining_trace_count,
            "projected_gpu_hours": budget["projected_gpu_hours"],
            "budgeted_gpu_hours": budget["budgeted_gpu_hours"],
            "projected_compressed_trace_gib": budget[
                "projected_compressed_trace_gib"
            ],
            "maximum_uncompressed_trace_gib": budget[
                "maximum_uncompressed_trace_gib"
            ],
            "recommended_data_disk_gib_including_model": budget[
                "recommended_data_disk_gib_including_model"
            ],
        }
    )
    if disk.free < dynamic_required_bytes:
        failures.append("free disk is below the full-Train safety budget")
    if invalid_existing_traces:
        failures.append("existing full-Train trace files failed contract validation")
        report["invalid_existing_traces"] = invalid_existing_traces[:10]

    omp_threads = os.environ.get("OMP_NUM_THREADS")
    report["omp_num_threads"] = omp_threads
    if omp_threads is not None and (
        not omp_threads.isdigit() or int(omp_threads) <= 0
    ):
        failures.append("OMP_NUM_THREADS must be a positive integer")

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
            failures.append("CUDA GPU is required for full-Train collection")
        else:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            report.update(
                {
                    "gpu_name": torch.cuda.get_device_name(index),
                    "gpu_memory_gb": round(properties.total_memory / 1000**3, 2),
                    "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "stage3_peak_reserved_gib": round(
                        budget["basis_peak_cuda_reserved_bytes"] / 1024**3,
                        2,
                    ),
                }
            )
            if not torch.cuda.is_bf16_supported():
                failures.append("GPU does not support BF16")
            if not _has_minimum_gpu_memory(properties.total_memory):
                failures.append("GPU has less than nominal 24 GB memory")
            if budget["basis_peak_cuda_reserved_bytes"] >= properties.total_memory:
                failures.append("stage-three peak memory does not fit this GPU")
            if capability[0] >= 10 and _version_tuple(str(torch.version.cuda)) < (12, 8):
                failures.append("Blackwell GPU requires a CUDA 12.8 or newer PyTorch build")

    for directory in (trace_root, run_root):
        probe = directory / ".bayesian_full_train_write_probe"
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
    print(f"Bayesian full-Train preflight report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
