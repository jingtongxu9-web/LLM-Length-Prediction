"""Collect the frozen 324-rollout final holdout exactly once after Stage-8B unlocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from llm_length_prediction.data.bayesian_trace import (
    BayesianTraceV1,
    bayesian_trace_path,
    read_bayesian_trace,
    write_bayesian_trace,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.instrumentation.bayesian import HuggingFaceBayesianCollector
from llm_length_prediction.runtime.model_paths import resolve_model_source
from llm_length_prediction.stage8_freeze import final_holdout_gate_report
from llm_length_prediction.stage8_holdout import (
    FinalHoldoutJob,
    build_final_holdout_summary,
    final_holdout_jobs,
    load_final_holdout_contract,
    validate_final_holdout_trace,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _trace_path(root: Path, job: FinalHoldoutJob) -> Path:
    return bayesian_trace_path(
        root,
        split="final_holdout",
        prompt_id=str(job.record["prompt_id"]),
        temperature=job.temperature,
        seed=job.seed,
    )


def _row(path: Path, job: FinalHoldoutJob, trace: BayesianTraceV1) -> dict[str, Any]:
    return {
        "job_rank": job.rank,
        "prompt_id": trace.prompt_id,
        "prompt_family_id": trace.prompt_family_id,
        "task": trace.task,
        "intended_length": trace.intended_length,
        "split": trace.split,
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


def _scan(
    *,
    root: Path,
    jobs: list[FinalHoldoutJob],
    config: dict[str, Any],
    lock_sha256: str,
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[FinalHoldoutJob], list[str]]:
    rows = []
    pending = []
    invalid = []
    expected_paths = {_trace_path(root, job) for job in jobs}
    for job in jobs:
        path = _trace_path(root, job)
        if not path.is_file():
            pending.append(job)
            continue
        try:
            trace = read_bayesian_trace(path)
            validate_final_holdout_trace(
                trace,
                job=job,
                config=config,
                lock_sha256=lock_sha256,
                manifest_sha256=manifest_sha256,
            )
            rows.append(_row(path, job, trace))
        except (OSError, KeyError, TypeError, ValueError) as error:
            invalid.append(f"{path}: {error}")
    unexpected = sorted(set(root.rglob("*.npz")).difference(expected_paths))
    invalid.extend(f"{path}: unexpected final holdout trace" for path in unexpected)
    return rows, pending, invalid


def _write_reports(
    *,
    run_root: Path,
    config: dict[str, Any],
    lock_sha256: str,
    manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows.sort(key=lambda row: int(row["job_rank"]))
    summary = build_final_holdout_summary(
        config,
        lock_sha256=lock_sha256,
        manifest_sha256=manifest_sha256,
        rows=rows,
    )
    _atomic_text(
        run_root / "collection_index.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    _atomic_text(
        run_root / "collection_report.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-only", action="store_true")
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--max-new-jobs", type=int)
    limit.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.max_new_jobs is not None and args.max_new_jobs <= 0:
        raise SystemExit("--max-new-jobs must be positive")
    gate = final_holdout_gate_report(args.config)
    if not gate["ready"]:
        raise SystemExit("final holdout gate is blocked: " + "; ".join(gate["failures"]))
    config, lock, records = load_final_holdout_contract(args.config)
    lock_path = Path(config["holdout_gate"]["benchmark_lock"])
    lock_sha256 = file_sha256(lock_path)
    manifest_sha256 = str(lock["final_holdout_manifest_sha256"])
    jobs = final_holdout_jobs(config, records)
    root = Path(config["outputs"]["final_trace_root"])
    run_root = Path(config["outputs"]["collection_run_root"])
    rows, pending, invalid = _scan(
        root=root,
        jobs=jobs,
        config=config,
        lock_sha256=lock_sha256,
        manifest_sha256=manifest_sha256,
    )
    if invalid:
        raise RuntimeError(
            "existing final traces failed validation; no files were overwritten:\n"
            + "\n".join(invalid[:10])
        )
    if args.report_only:
        report = _write_reports(
            run_root=run_root,
            config=config,
            lock_sha256=lock_sha256,
            manifest_sha256=manifest_sha256,
            rows=rows,
        )
        print(
            f"valid={report['valid_trace_count']} "
            f"missing={report['missing_trace_count']} status={report['status']}"
        )
        return
    selected = pending if args.all else pending[: args.max_new_jobs or 25]
    full_train = json.loads(
        Path("configs/experiments/bayesian_sequential_full_train_v1.json").read_text(
            encoding="utf-8"
        )
    )
    model_source = resolve_model_source(args.model)
    revision = str(full_train["model"]["revision"])
    marker = Path(model_source) / ".frozen_revision"
    if not Path(model_source).is_dir() or not marker.is_file():
        raise ValueError("final collection requires the local frozen Qwen snapshot")
    if marker.read_text(encoding="utf-8").strip() != revision:
        raise ValueError("local Qwen revision changed")
    collector = None
    if selected:
        collector = HuggingFaceBayesianCollector(
            model_source,
            revision=revision,
            dtype=str(full_train["model"]["dtype"]),
            device=args.device,
            max_new_tokens=int(config["final_holdout_plan"]["max_new_tokens"]),
            temperature=selected[0].temperature,
            top_p=float(config["final_holdout_plan"]["top_p"]),
            seed=selected[0].seed,
            trace_stride=int(full_train["trace"]["stride"]),
            pooling_temperature=float(full_train["prompt_representation"]["pooling_temperature"]),
            prior_layer=int(full_train["model"]["prior_layer_zero_based"]),
            entropy_chunk_tokens=int(full_train["prompt_representation"]["entropy_chunk_tokens"]),
            reported_model_name=str(full_train["model"]["id"]),
        )
    for job in selected:
        assert collector is not None
        collector.temperature = job.temperature
        collector.seed = job.seed
        record = job.record
        trace = collector.collect_trace(
            str(record["prompt"]),
            prompt_id=str(record["prompt_id"]),
            prompt_family_id=str(record["prompt_family_id"]),
            task=str(record["task_type"]),
            intended_length=str(record["intended_length"]),
            split="final_holdout",
        )
        trace.metadata.update(
            {
                "collection_id": "bayesian-sequential-v1-final-holdout",
                "collection_job_rank": job.rank,
                "stage8a_config_sha256": config["_config_sha256"],
                "stage8b_lock_sha256": lock_sha256,
                "final_holdout_manifest_sha256": manifest_sha256,
                "final_holdout_accessed": True,
            }
        )
        validate_final_holdout_trace(
            trace,
            job=job,
            config=config,
            lock_sha256=lock_sha256,
            manifest_sha256=manifest_sha256,
        )
        path = _trace_path(root, job)
        _atomic_trace(path, trace)
        restored = read_bayesian_trace(path)
        validate_final_holdout_trace(
            restored,
            job=job,
            config=config,
            lock_sha256=lock_sha256,
            manifest_sha256=manifest_sha256,
        )
        rows.append(_row(path, job, restored))
        print(
            f"completed final rank={job.rank} prompt={record['prompt_id']} "
            f"temperature={job.temperature} seed={job.seed}: {path}",
            flush=True,
        )
    report = _write_reports(
        run_root=run_root,
        config=config,
        lock_sha256=lock_sha256,
        manifest_sha256=manifest_sha256,
        rows=rows,
    )
    print(
        f"valid={report['valid_trace_count']} missing={report['missing_trace_count']} "
        f"status={report['status']}"
    )
    if report["failures"]:
        raise RuntimeError("final collection acceptance failed: " + "; ".join(report["failures"]))


if __name__ == "__main__":
    main()
