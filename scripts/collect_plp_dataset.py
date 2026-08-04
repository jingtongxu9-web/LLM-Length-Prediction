"""Re-run frozen prompts and collect the hidden-state inputs required by PLP v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from llm_length_prediction.data.plp import plp_trace_path, read_plp_trace, write_plp_trace
from llm_length_prediction.experiment import file_sha256, rollout_jobs
from llm_length_prediction.instrumentation.plp import HuggingFacePLPCollector
from llm_length_prediction.plp_experiment import (
    load_plp_base_experiment,
    load_plp_config,
    validate_plp_config,
    validate_plp_trace,
)
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_CONFIG = Path("configs/experiments/plp_v2_manifest.json")


def validate_local_model_revision(model_source: str, expected_revision: str) -> None:
    """Reject an unmarked or mismatched local snapshot before expensive generation."""

    model_path = Path(model_source)
    if not model_path.is_dir():
        return
    marker = model_path / ".frozen_revision"
    if not marker.is_file():
        raise ValueError(f"local model directory is missing {marker.name}: {model_path}")
    actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != expected_revision:
        raise ValueError(
            "local model revision mismatch: "
            f"expected {expected_revision!r}, got {actual_revision!r}"
        )


def _atomic_write(path: Path, trace: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_plp_trace(temporary, trace)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_existing(
    path: Path,
    record: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        trace = read_plp_trace(path)
        validate_plp_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
    except (OSError, ValueError, TypeError):
        return False
    return True


def _write_index(
    trace_root: Path,
    run_root: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    counts = {"train": 0, "test": 0}
    stop_reasons: Counter[str] = Counter()
    total_points = 0
    total_bytes = 0
    max_trace_bytes = 0
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    for record, seed in rollout_jobs(records):
        path = plp_trace_path(trace_root, record, seed)
        if not path.is_file():
            continue
        trace = read_plp_trace(path)
        validate_plp_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        counts[record["split"]] += 1
        stop_reasons[trace.stop_reason] += 1
        trace_bytes = path.stat().st_size
        point_count = len(trace.steps)
        total_points += point_count
        total_bytes += trace_bytes
        max_trace_bytes = max(max_trace_bytes, trace_bytes)
        peak_allocated_bytes = max(
            peak_allocated_bytes,
            int(trace.metadata.get("cuda_peak_allocated_bytes") or 0),
        )
        peak_reserved_bytes = max(
            peak_reserved_bytes,
            int(trace.metadata.get("cuda_peak_reserved_bytes") or 0),
        )
        rows.append(
            {
                "split": record["split"],
                "prompt_id": trace.prompt_id,
                "seed": trace.seed,
                "prompt_tokens": trace.prompt_tokens,
                "output_tokens": trace.output_tokens,
                "generated_token_ids_sha256": hashlib.sha256(
                    trace.generated_token_ids.tobytes()
                ).hexdigest(),
                "plp_points": point_count,
                "stop_reason": trace.stop_reason,
                "trace_bytes": trace_bytes,
                "trace_path": str(path),
                "trace_sha256": file_sha256(path),
            }
        )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "collection_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "completed": len(rows),
        "by_split": counts,
        "by_stop_reason": dict(sorted(stop_reasons.items())),
        "censored_max_new_tokens": stop_reasons["max_new_tokens"],
        "total_plp_points": total_points,
        "total_trace_bytes": total_bytes,
        "total_trace_gib": total_bytes / (1024**3),
        "max_trace_bytes": max_trace_bytes,
        "max_cuda_peak_allocated_bytes": peak_allocated_bytes,
        "max_cuda_peak_reserved_bytes": peak_reserved_bytes,
    }
    (run_root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Optional local model path override")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train"])
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if "test" in args.splits and not args.confirm_final_test:
        raise SystemExit("refusing PLP Test collection without --confirm-final-test")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    config = load_plp_config(args.config)
    experiment, records = load_plp_base_experiment(config)
    validate_plp_config(config, experiment)
    generation = experiment["generation"]
    model = experiment["model"]
    trace_root = Path(config["outputs"]["trace_root"])
    run_root = Path(config["outputs"]["run_root"])
    selected = set(args.splits)
    jobs = [job for job in rollout_jobs(records) if job[0]["split"] in selected]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    pending = []
    resumed = 0
    for record, seed in jobs:
        path = plp_trace_path(trace_root, record, seed)
        if _valid_existing(path, record, seed, config, experiment):
            resumed += 1
            continue
        pending.append((record, seed))

    completed = 0
    if pending:
        model_source = resolve_model_source(args.model)
        validate_local_model_revision(model_source, model["revision"])
        collector = HuggingFacePLPCollector(
            model_source,
            revision=model["revision"],
            dtype=model["dtype"],
            max_new_tokens=generation["max_new_tokens"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            trace_stride=config["trace"]["stride"],
            pooling_temperature=config["representation"]["prompt_pooling_temperature"],
        )
        actual_hidden_size = int(collector.model.config.hidden_size)
        expected_hidden_size = int(config["representation"]["hidden_size"])
        if actual_hidden_size != expected_hidden_size:
            raise ValueError(
                f"model hidden_size={actual_hidden_size}, expected {expected_hidden_size}"
            )
    for record, seed in pending:
        path = plp_trace_path(trace_root, record, seed)
        collector.seed = seed
        trace = collector.collect_trace(
            record["prompt"], prompt_id=record["prompt_id"], task=record["task_type"]
        )
        trace.metadata.update(
            {
                "method_id": config["method_id"],
                "base_experiment_id": experiment["experiment_id"],
                "prompt_family_id": record["prompt_family_id"],
                "intended_length": record["intended_length"],
                "split": record["split"],
                "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
            }
        )
        validate_plp_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        _atomic_write(path, trace)
        completed += 1
        print(f"completed PLP {record['prompt_id']} seed={seed}: {path}")

    summary = _write_index(trace_root, run_root, records, config, experiment)
    print(
        f"new={completed} resumed={resumed} indexed={summary['completed']} "
        f"by_split={summary['by_split']} total_gib={summary['total_trace_gib']:.3f}"
    )


if __name__ == "__main__":
    main()
