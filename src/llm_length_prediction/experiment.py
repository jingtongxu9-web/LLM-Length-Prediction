from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_length_prediction.data.schema import GenerationTrace


class ExperimentContractError(ValueError):
    """Raised when a frozen experiment input or trace violates its contract."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment(path: str | Path) -> dict[str, Any]:
    experiment = json.loads(Path(path).read_text(encoding="utf-8"))
    if experiment.get("schema_version") != 1:
        raise ExperimentContractError("unsupported experiment schema_version")
    return experiment


def load_frozen_prompts(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = experiment["inputs"]
    path = Path(inputs["prompt_manifest"])
    if not path.is_file():
        raise ExperimentContractError(f"prompt manifest does not exist: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != inputs["prompt_manifest_sha256"]:
        raise ExperimentContractError(
            "prompt manifest SHA-256 mismatch: "
            f"expected {inputs['prompt_manifest_sha256']}, got {actual_hash}"
        )

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ExperimentContractError(f"invalid prompt JSON on line {line_number}") from error

    expected_prompt_count = int(inputs["prompt_count"])
    if len(records) != expected_prompt_count:
        raise ExperimentContractError(
            f"expected {expected_prompt_count} prompts, found {len(records)}"
        )
    prompt_ids = [record.get("prompt_id") for record in records]
    if any(not prompt_id for prompt_id in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ExperimentContractError("prompt_id values must be non-empty and unique")

    frozen_seeds = tuple(int(seed) for seed in experiment["generation"]["seeds"])
    split_rollouts: Counter[str] = Counter()
    for record in records:
        if record.get("split") not in inputs["splits"]:
            raise ExperimentContractError(f"unknown split for prompt {record.get('prompt_id')}")
        if tuple(record.get("generation_seeds", ())) != frozen_seeds:
            raise ExperimentContractError(
                f"prompt {record['prompt_id']} does not use frozen seeds {frozen_seeds}"
            )
        if not record.get("prompt"):
            raise ExperimentContractError(f"prompt {record['prompt_id']} is empty")
        split_rollouts[record["split"]] += len(frozen_seeds)

    expected_splits = {key: int(value) for key, value in inputs["splits"].items()}
    if dict(split_rollouts) != expected_splits:
        raise ExperimentContractError(
            f"rollout split counts mismatch: expected {expected_splits}, got {dict(split_rollouts)}"
        )
    if sum(split_rollouts.values()) != int(inputs["rollout_count"]):
        raise ExperimentContractError("total rollout count does not match the frozen manifest")
    return records


def rollout_jobs(
    records: list[dict[str, Any]], *, split: str | None = None
) -> Iterator[tuple[dict[str, Any], int]]:
    for record in records:
        if split is not None and record["split"] != split:
            continue
        for seed in record["generation_seeds"]:
            yield record, int(seed)


def trace_path(trace_root: str | Path, record: dict[str, Any], seed: int) -> Path:
    return Path(trace_root) / record["split"] / record["prompt_id"] / f"seed_{seed}.jsonl"


def validate_frozen_trace(
    trace: GenerationTrace,
    *,
    record: dict[str, Any],
    seed: int,
    experiment: dict[str, Any],
) -> None:
    trace.validate()
    model = experiment["model"]
    generation = experiment["generation"]
    errors: list[str] = []

    expected_values = {
        "prompt_id": (trace.prompt_id, record["prompt_id"]),
        "task": (trace.task, record["task_type"]),
        "seed": (trace.seed, seed),
        "temperature": (trace.temperature, generation["temperature"]),
        "model_revision": (trace.model_revision, model["revision"]),
        "tokenizer_revision": (trace.tokenizer_revision, model["tokenizer_revision"]),
    }
    for name, (actual, expected) in expected_values.items():
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")

    expected_metadata = {
        "experiment_id": experiment["experiment_id"],
        "prompt_family_id": record["prompt_family_id"],
        "intended_length": record["intended_length"],
        "split": record["split"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "top_p": generation["top_p"],
        "max_new_tokens": generation["max_new_tokens"],
        "trace_stride": generation["trace_stride"],
        "entropy_window": generation["entropy_window"],
        "chat_template": generation["chat_template"],
        "layer_indexing": model["layer_indexing"],
        "output_length_includes_eos": generation["output_length_includes_eos"],
    }
    for name, expected in expected_metadata.items():
        actual = trace.metadata.get(name)
        if actual != expected:
            errors.append(f"metadata.{name}={actual!r}, expected {expected!r}")

    feature_layer = int(model["feature_layer"])
    if feature_layer not in trace.prefill_hidden_states:
        errors.append(f"missing prefill hidden state for Layer {feature_layer}")
    if model["dtype"] not in str(trace.metadata.get("dtype")):
        errors.append(f"metadata.dtype={trace.metadata.get('dtype')!r}, expected {model['dtype']}")
    if trace.output_tokens <= 0 or trace.output_tokens > int(generation["max_new_tokens"]):
        errors.append("output_tokens is outside the frozen generation range")
    if not trace.points or trace.points[-1].step != trace.output_tokens:
        errors.append("trace points do not include the final generated token")
    if trace.stop_reason not in {"eos", "max_new_tokens"}:
        errors.append(f"unsupported stop_reason={trace.stop_reason!r}")

    if errors:
        raise ExperimentContractError("trace contract mismatch: " + "; ".join(errors))
