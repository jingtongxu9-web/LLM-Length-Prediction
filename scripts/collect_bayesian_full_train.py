"""Collect the frozen 1,620-rollout Bayesian Sequential full-Train trace set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from llm_length_prediction.bayesian_full_train import (
    BayesianFullTrainJob,
    bayesian_full_train_jobs,
    build_bayesian_full_train_summary,
    load_bayesian_full_train,
    validate_bayesian_full_train_trace,
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

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_full_train_v1.json")


def _validate_local_revision(model_source: str, expected: str) -> None:
    path = Path(model_source)
    if not path.is_dir():
        raise ValueError("full-Train collection requires a local frozen model snapshot")
    marker = path / ".frozen_revision"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("local model is missing the matching .frozen_revision marker")


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


def _path(root: Path, job: BayesianFullTrainJob) -> Path:
    return bayesian_trace_path(
        root,
        split="train",
        prompt_id=job.record["prompt_id"],
        temperature=job.temperature,
        seed=job.seed,
    )


def _row(path: Path, job: BayesianFullTrainJob, trace: BayesianTraceV1) -> dict[str, Any]:
    return {
        "job_rank": job.rank,
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
        "cuda_peak_allocated_bytes": int(
            trace.metadata["cuda_peak_allocated_bytes"]
        ),
        "cuda_peak_reserved_bytes": int(trace.metadata["cuda_peak_reserved_bytes"]),
        "generated_token_ids_sha256": hashlib.sha256(
            trace.generated_token_ids.tobytes()
        ).hexdigest(),
        "trace_sha256": file_sha256(path),
        "trace_path": str(path),
    }


def _scan_existing(
    *,
    trace_root: Path,
    jobs: list[BayesianFullTrainJob],
    collection: dict[str, Any],
    collection_config_sha256: str,
) -> tuple[list[dict[str, Any]], list[BayesianFullTrainJob], list[str]]:
    rows = []
    pending = []
    invalid = []
    expected_paths = {_path(trace_root, job) for job in jobs}
    for job in jobs:
        path = _path(trace_root, job)
        if not path.is_file():
            pending.append(job)
            continue
        try:
            trace = read_bayesian_trace(path)
            validate_bayesian_full_train_trace(
                trace,
                job=job,
                collection=collection,
                collection_config_sha256=collection_config_sha256,
            )
            rows.append(_row(path, job, trace))
        except (OSError, KeyError, TypeError, ValueError) as error:
            invalid.append(f"{path}: {error}")
    unexpected_paths = sorted(set(trace_root.rglob("*.npz")).difference(expected_paths))
    invalid.extend(f"{path}: unexpected full-Train trace path" for path in unexpected_paths)
    return rows, pending, invalid


def _write_reports(
    *,
    run_root: Path,
    collection: dict[str, Any],
    collection_config_sha256: str,
    rows: list[dict[str, Any]],
    new_trace_count: int,
    resumed_trace_count: int,
) -> dict[str, Any]:
    rows.sort(key=lambda row: int(row["job_rank"]))
    summary = build_bayesian_full_train_summary(
        collection,
        collection_config_sha256=collection_config_sha256,
        rows=rows,
        new_trace_count=new_trace_count,
        resumed_trace_count=resumed_trace_count,
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
    if args.report_only and (args.max_new_jobs is not None or args.all):
        raise SystemExit("--report-only cannot be combined with collection limits")

    collection, _, records = load_bayesian_full_train(args.config)
    collection_config_sha256 = file_sha256(args.config)
    jobs = bayesian_full_train_jobs(collection, records)
    trace_root = Path(collection["outputs"]["trace_root"])
    run_root = Path(collection["outputs"]["run_root"])
    rows, pending, invalid = _scan_existing(
        trace_root=trace_root,
        jobs=jobs,
        collection=collection,
        collection_config_sha256=collection_config_sha256,
    )
    if invalid:
        details = "\n".join(invalid[:10])
        remainder = f"\n... and {len(invalid) - 10} more" if len(invalid) > 10 else ""
        raise RuntimeError(
            "existing full-Train traces failed validation; no files were overwritten:\n"
            + details
            + remainder
        )
    resumed = len(rows)
    if args.report_only:
        summary = _write_reports(
            run_root=run_root,
            collection=collection,
            collection_config_sha256=collection_config_sha256,
            rows=rows,
            new_trace_count=0,
            resumed_trace_count=resumed,
        )
        print(
            f"valid={summary['valid_trace_count']} "
            f"missing={summary['missing_trace_count']} status={summary['status']}"
        )
        return

    if args.all:
        selected_jobs = pending
    else:
        maximum = args.max_new_jobs or collection["resumability"][
            "maximum_new_jobs_default"
        ]
        selected_jobs = pending[: int(maximum)]
    collector = None
    if selected_jobs:
        model_source = resolve_model_source(args.model)
        _validate_local_revision(model_source, collection["model"]["revision"])
        collector = HuggingFaceBayesianCollector(
            model_source,
            revision=collection["model"]["revision"],
            dtype=collection["model"]["dtype"],
            device=args.device,
            max_new_tokens=collection["generation"]["max_new_tokens"],
            temperature=selected_jobs[0].temperature,
            top_p=collection["generation"]["top_p"],
            seed=selected_jobs[0].seed,
            trace_stride=collection["trace"]["stride"],
            pooling_temperature=collection["prompt_representation"][
                "pooling_temperature"
            ],
            prior_layer=collection["model"]["prior_layer_zero_based"],
            entropy_chunk_tokens=collection["prompt_representation"][
                "entropy_chunk_tokens"
            ],
            reported_model_name=collection["model"]["id"],
        )
    completed = 0
    for job in selected_jobs:
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
            split="train",
        )
        trace.metadata.update(
            {
                "collection_job_rank": job.rank,
                "collection_id": collection["collection_id"],
                "collection_stage": collection["stage"],
                "collection_config_sha256": collection_config_sha256,
                "scientific_contract_sha256": collection["scientific_contract"][
                    "sha256"
                ],
                "source_prompt_manifest_sha256": collection["source_prompts"][
                    "sha256"
                ],
                "stage3_pilot_summary_sha256": collection["stage3_pilot_gate"][
                    "sha256"
                ],
                "final_holdout_accessed": False,
            }
        )
        validate_bayesian_full_train_trace(
            trace,
            job=job,
            collection=collection,
            collection_config_sha256=collection_config_sha256,
        )
        path = _path(trace_root, job)
        _atomic_trace(path, trace)
        restored = read_bayesian_trace(path)
        validate_bayesian_full_train_trace(
            restored,
            job=job,
            collection=collection,
            collection_config_sha256=collection_config_sha256,
        )
        rows.append(_row(path, job, restored))
        completed += 1
        print(
            f"completed full-Train rank={job.rank} prompt={record['prompt_id']} "
            f"temperature={job.temperature} seed={job.seed}: {path}"
        )
    summary = _write_reports(
        run_root=run_root,
        collection=collection,
        collection_config_sha256=collection_config_sha256,
        rows=rows,
        new_trace_count=completed,
        resumed_trace_count=resumed,
    )
    print(
        f"new={completed} resumed={resumed} valid={summary['valid_trace_count']} "
        f"missing={summary['missing_trace_count']} status={summary['status']}"
    )
    if summary["failures"]:
        raise RuntimeError(
            "Bayesian full-Train acceptance failed: "
            + "; ".join(summary["failures"])
        )


if __name__ == "__main__":
    main()
