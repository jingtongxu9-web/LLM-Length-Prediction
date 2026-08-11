"""Collect the frozen nine-rollout Bayesian Sequential unified-trace pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from llm_length_prediction.bayesian_pilot import (
    BayesianPilotJob,
    bayesian_pilot_jobs,
    load_bayesian_pilot,
    validate_bayesian_pilot_trace,
)
from llm_length_prediction.data.bayesian_trace import (
    BayesianTraceV1,
    bayesian_trace_path,
    read_bayesian_trace,
    write_bayesian_trace,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.instrumentation.bayesian import HuggingFaceBayesianCollector
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_pilot_v1.json")


def _validate_local_revision(model_source: str, expected: str) -> None:
    path = Path(model_source)
    if not path.is_dir():
        return
    marker = path / ".frozen_revision"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("local model is missing the matching .frozen_revision marker")


def _atomic_trace(path: Path, trace: BayesianTraceV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_bayesian_trace(temporary, trace)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _path(root: Path, job: BayesianPilotJob) -> Path:
    return bayesian_trace_path(
        root,
        split=job.record["split"],
        prompt_id=job.record["prompt_id"],
        temperature=job.temperature,
        seed=job.seed,
    )


def _valid_existing(
    path: Path,
    *,
    job: BayesianPilotJob,
    pilot: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        trace = read_bayesian_trace(path)
        validate_bayesian_pilot_trace(trace, job=job, pilot=pilot)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _write_pilot_reports(
    *,
    trace_root: Path,
    run_root: Path,
    jobs: list[BayesianPilotJob],
    pilot: dict[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    rows = []
    stop_reasons: Counter[str] = Counter()
    task_lengths: Counter[str] = Counter()
    total_bytes = 0
    total_duration_ms = 0.0
    peak_allocated = 0
    peak_reserved = 0
    for job in jobs:
        path = _path(trace_root, job)
        if not path.is_file():
            continue
        trace = read_bayesian_trace(path)
        validate_bayesian_pilot_trace(trace, job=job, pilot=pilot)
        stop_reasons[trace.stop_reason] += 1
        task_lengths[f"{trace.task}:{trace.intended_length}"] += 1
        total_bytes += path.stat().st_size
        total_duration_ms += trace.duration_ms
        peak_allocated = max(
            peak_allocated,
            int(trace.metadata.get("cuda_peak_allocated_bytes", 0)),
        )
        peak_reserved = max(
            peak_reserved,
            int(trace.metadata.get("cuda_peak_reserved_bytes", 0)),
        )
        rows.append(
            {
                "prompt_id": trace.prompt_id,
                "prompt_family_id": trace.prompt_family_id,
                "task": trace.task,
                "intended_length": trace.intended_length,
                "temperature": trace.temperature,
                "seed": trace.seed,
                "observed_tokens": trace.observed_tokens,
                "saved_point_count": len(trace.saved_steps),
                "stop_reason": trace.stop_reason,
                "duration_ms": trace.duration_ms,
                "trace_bytes": path.stat().st_size,
                "generated_token_ids_sha256": hashlib.sha256(
                    trace.generated_token_ids.tobytes()
                ).hexdigest(),
                "trace_sha256": file_sha256(path),
                "trace_path": str(path),
            }
        )
    expected = int(pilot["acceptance"]["expected_trace_count"])
    missing = len(jobs) - len(rows)
    censoring_rate = stop_reasons["max_new_tokens"] / len(rows) if rows else 0.0
    acceptance = pilot["acceptance"]
    failures = []
    warnings = []
    if require_complete and len(rows) != expected:
        failures.append(f"expected {expected} valid traces, found {len(rows)}")
    if require_complete and missing > acceptance["maximum_missing_trace_count"]:
        failures.append(f"missing trace count {missing} exceeds the frozen limit")
    if require_complete and len(task_lengths) != acceptance["required_task_length_cells"]:
        failures.append("the complete task-by-length pilot grid is not present")
    if censoring_rate >= acceptance["abort_censoring_rate"]:
        failures.append("pilot censoring rate reaches the frozen abort threshold")
    elif censoring_rate >= acceptance["warning_censoring_rate"]:
        warnings.append("pilot censoring rate reaches the frozen warning threshold")
    summary = {
        "pilot_id": pilot["pilot_id"],
        "status": "pass" if not failures and len(rows) == expected else "incomplete_or_failed",
        "expected_trace_count": expected,
        "valid_trace_count": len(rows),
        "missing_selected_job_count": missing,
        "by_stop_reason": dict(sorted(stop_reasons.items())),
        "by_task_length": dict(sorted(task_lengths.items())),
        "censoring_rate": censoring_rate,
        "total_observed_tokens": sum(row["observed_tokens"] for row in rows),
        "total_duration_ms": total_duration_ms,
        "total_trace_bytes": total_bytes,
        "total_trace_gib": total_bytes / 1024**3,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "warnings": warnings,
        "failures": failures,
        "real_qwen_pilot_complete": not failures and len(rows) == expected,
        "final_holdout_accessed": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "collection_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_root / "pilot_acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("Bayesian pilot acceptance failed: " + "; ".join(failures))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    pilot, _, records = load_bayesian_pilot(args.config)
    all_jobs = bayesian_pilot_jobs(pilot, records)
    selected_jobs = all_jobs if args.limit is None else all_jobs[: args.limit]
    trace_root = Path(pilot["outputs"]["trace_root"])
    run_root = Path(pilot["outputs"]["run_root"])
    pending = []
    resumed = 0
    for job in selected_jobs:
        path = _path(trace_root, job)
        if _valid_existing(path, job=job, pilot=pilot):
            resumed += 1
        else:
            pending.append(job)

    collector = None
    if pending:
        model_source = resolve_model_source(args.model)
        _validate_local_revision(model_source, pilot["model"]["revision"])
        collector = HuggingFaceBayesianCollector(
            model_source,
            revision=pilot["model"]["revision"],
            dtype=pilot["model"]["dtype"],
            device=args.device,
            max_new_tokens=pilot["generation"]["max_new_tokens"],
            temperature=pending[0].temperature,
            top_p=pilot["generation"]["top_p"],
            seed=pending[0].seed,
            trace_stride=pilot["trace"]["stride"],
            pooling_temperature=pilot["prompt_representation"]["pooling_temperature"],
            prior_layer=pilot["model"]["prior_layer_zero_based"],
            entropy_chunk_tokens=pilot["prompt_representation"]["entropy_chunk_tokens"],
            reported_model_name=pilot["model"]["id"],
        )
    completed = 0
    for job in pending:
        assert collector is not None
        collector.temperature = job.temperature
        collector.seed = job.seed
        record = job.record
        trace = collector.collect_trace(
            record["prompt"],
            prompt_id=record["prompt_id"],
            prompt_family_id=record["prompt_family_id"],
            task=record["task_type"],
            intended_length=record["intended_length"],
            split=record["split"],
        )
        trace.metadata.update(
            {
                "pilot_id": pilot["pilot_id"],
                "scientific_contract_sha256": pilot["scientific_contract"]["sha256"],
                "source_prompt_manifest_sha256": pilot["source_prompts"]["sha256"],
            }
        )
        validate_bayesian_pilot_trace(trace, job=job, pilot=pilot)
        path = _path(trace_root, job)
        _atomic_trace(path, trace)
        completed += 1
        print(
            f"completed Bayesian pilot {record['prompt_id']} "
            f"temperature={job.temperature} seed={job.seed}: {path}"
        )
    summary = _write_pilot_reports(
        trace_root=trace_root,
        run_root=run_root,
        jobs=selected_jobs,
        pilot=pilot,
        require_complete=args.limit is None,
    )
    print(
        f"new={completed} resumed={resumed} valid={summary['valid_trace_count']} "
        f"expected={summary['expected_trace_count']} status={summary['status']}"
    )


if __name__ == "__main__":
    main()
