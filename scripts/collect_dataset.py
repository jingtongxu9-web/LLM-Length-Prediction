"""Collect the frozen ALPS v1 Qwen dataset with resumable per-rollout files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from llm_length_prediction.data.io import read_trace_jsonl, write_trace_jsonl
from llm_length_prediction.experiment import (
    file_sha256,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)
from llm_length_prediction.instrumentation.huggingface import HuggingFaceSignalCollector
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _atomic_write_trace(path: Path, trace: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_trace_jsonl(temporary, trace)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_valid_existing_trace(
    path: Path,
    record: dict[str, Any],
    seed: int,
    experiment: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _write_indexes(
    trace_root: Path,
    run_root: Path,
    records: list[dict[str, Any]],
    experiment: dict[str, Any],
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts = {"train": 0, "test": 0}
    for record, seed in rollout_jobs(records):
        path = trace_path(trace_root, record, seed)
        if not path.exists():
            continue
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
        split = record["split"]
        counts[split] = counts.get(split, 0) + 1
        rows.append(
            {
                "split": split,
                "prompt_id": trace.prompt_id,
                "seed": trace.seed,
                "prompt_tokens": trace.prompt_tokens,
                "output_tokens": trace.output_tokens,
                "stop_reason": trace.stop_reason,
                "duration_ms": trace.duration_ms,
                "trace_path": str(path),
                "trace_sha256": file_sha256(path),
                "model_revision": trace.model_revision,
                "tokenizer_revision": trace.tokenizer_revision,
            }
        )
    run_root.mkdir(parents=True, exist_ok=True)
    index = run_root / "collection_index.jsonl"
    index.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {"completed": len(rows), "by_split": counts}
    (run_root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model", default=None, help="Optional local model path override")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train"])
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required whenever the immutable final test split is collected",
    )
    parser.add_argument("--limit", type=int, help="Pilot limit after prompt/seed expansion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if "test" in args.splits and not args.confirm_final_test:
        raise SystemExit("refusing to collect final test split without --confirm-final-test")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    expected_hash = experiment["inputs"]["prompt_manifest_sha256"]

    model = experiment["model"]
    generation = experiment["generation"]
    collector = HuggingFaceSignalCollector(
        resolve_model_source(args.model),
        revision=model["revision"],
        dtype=model["dtype"],
        candidate_layers=[model["feature_layer"]],
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        trace_stride=generation["trace_stride"],
        entropy_window=generation["entropy_window"],
    )

    trace_root = Path(experiment["outputs"]["trace_root"])
    run_root = Path(experiment["outputs"]["run_root"])
    selected_splits = set(args.splits)
    jobs = [job for job in rollout_jobs(records) if job[0]["split"] in selected_splits]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    completed = 0
    skipped = 0
    for record, seed in jobs:
        output = trace_path(trace_root, record, seed)
        if _is_valid_existing_trace(output, record, seed, experiment):
            skipped += 1
            continue
        collector.seed = seed
        trace = collector.collect_trace(
            record["prompt"], prompt_id=record["prompt_id"], task=record["task_type"]
        )
        trace.metadata.update(
            {
                "experiment_id": experiment["experiment_id"],
                "prompt_family_id": record["prompt_family_id"],
                "intended_length": record["intended_length"],
                "split": record["split"],
                "prompt_manifest_sha256": expected_hash,
            }
        )
        _atomic_write_trace(output, trace)
        completed += 1
        print(f"completed {record['prompt_id']} seed={seed}: {output}")

    counts = _write_indexes(trace_root, run_root, records, experiment)
    print(f"new={completed} resumed={skipped} indexed={sum(counts.values())} by_split={counts}")


if __name__ == "__main__":
    main()
