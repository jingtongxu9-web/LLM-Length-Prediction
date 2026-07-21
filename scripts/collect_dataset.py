"""Collect the frozen ALPS v1 Qwen dataset with resumable per-rollout files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from llm_length_prediction.data.io import read_trace_jsonl, write_trace_jsonl
from llm_length_prediction.instrumentation.huggingface import HuggingFaceSignalCollector
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prompts(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _trace_path(root: Path, record: dict[str, Any], seed: int) -> Path:
    return root / record["split"] / record["prompt_id"] / f"seed_{seed}.jsonl"


def _is_valid_existing_trace(path: Path, record: dict[str, Any], seed: int) -> bool:
    if not path.exists():
        return False
    try:
        trace = read_trace_jsonl(path)[0]
    except (OSError, ValueError, TypeError):
        return False
    return trace.prompt_id == record["prompt_id"] and trace.seed == seed


def _write_indexes(trace_root: Path, run_root: Path) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts = {"train": 0, "test": 0}
    for path in sorted(trace_root.glob("*/*/seed_*.jsonl")):
        trace = read_trace_jsonl(path)[0]
        split = path.relative_to(trace_root).parts[0]
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
                "trace_sha256": _sha256(path),
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

    experiment = _read_json(args.experiment)
    prompt_path = Path(experiment["inputs"]["prompt_manifest"])
    expected_hash = experiment["inputs"]["prompt_manifest_sha256"]
    actual_hash = _sha256(prompt_path)
    if actual_hash != expected_hash:
        raise SystemExit(
            f"prompt manifest hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

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
    jobs = [
        (record, seed)
        for record in _read_prompts(prompt_path)
        if record["split"] in args.splits
        for seed in record["generation_seeds"]
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    completed = 0
    skipped = 0
    for record, seed in jobs:
        output = _trace_path(trace_root, record, seed)
        if _is_valid_existing_trace(output, record, seed):
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

    counts = _write_indexes(trace_root, run_root)
    print(f"new={completed} resumed={skipped} indexed={sum(counts.values())} by_split={counts}")


if __name__ == "__main__":
    main()
