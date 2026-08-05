"""Contracts and trace loading for the isolated Hybrid v3 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

from llm_length_prediction.data.hybrid import (
    HybridV3Trace,
    hybrid_trace_path,
    read_hybrid_trace,
)
from llm_length_prediction.experiment import (
    file_sha256,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
)
from llm_length_prediction.models.hybrid import HybridSample, build_hybrid_samples

LoadedHybridTrace: TypeAlias = tuple[dict[str, Any], int, Path, HybridV3Trace]


def load_hybrid_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("method_id") != "alps-plp-hybrid-v3":
        raise ValueError("unsupported Hybrid v3 config")
    return config


def load_hybrid_experiment(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment = load_experiment(config["base_experiment"])
    if experiment["experiment_id"] != config["base_experiment_id"]:
        raise ValueError("Hybrid config and base experiment IDs differ")
    return experiment, load_frozen_prompts(experiment)


def validate_hybrid_config(config: dict[str, Any], experiment: dict[str, Any]) -> None:
    errors: list[str] = []
    if config.get("preserves_frozen_v2") is not True:
        errors.append("preserves_frozen_v2 must be true")
    trace = config.get("trace", {})
    if trace.get("schema_name") != "hybrid-v3-unified-trace" or trace.get("schema_version") != 1:
        errors.append("Hybrid trace schema must be isolated version 1")
    if trace.get("stride") != experiment["generation"]["trace_stride"]:
        errors.append("trace stride differs from the base experiment")
    representation = config.get("representation", {})
    expected = {
        "prior_layer": 14,
        "prior_hidden_size": 3584,
        "prompt_layer": "final_transformer_layer",
        "decode_layer": "final_transformer_layer",
        "hidden_size": 3584,
        "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
        "prompt_pooling_temperature": 1.0,
    }
    for name, value in expected.items():
        if representation.get(name) != value:
            errors.append(f"representation.{name} must be {value!r}")
    if config.get("stacking", {}).get("alps_target") != "log1p_output_tokens":
        errors.append("ALPS stacking target must be log1p_output_tokens")
    head = config.get("progressive_head", {})
    if head.get("terminal_zero_bin") is not True or head.get("num_bins") != 20:
        errors.append("v3 progressive head requires 20 bins and a terminal zero bin")
    training = config.get("training", {})
    frozen_training = {
        "seed": 42,
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "hyperparameter_selection": "none_after_protocol_freeze",
    }
    for name, value in frozen_training.items():
        if training.get(name) != value:
            errors.append(f"training.{name} must be {value!r}")
    if errors:
        raise ValueError("Hybrid v3 contract mismatch: " + "; ".join(errors))


def validate_hybrid_trace(
    trace: HybridV3Trace,
    *,
    record: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    trace.validate()
    errors: list[str] = []
    generation = experiment["generation"]
    model = experiment["model"]
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
    metadata_expected = {
        "experiment_id": experiment["experiment_id"],
        "prompt_family_id": record["prompt_family_id"],
        "intended_length": record["intended_length"],
        "split": record["split"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "trace_schema": config["trace"]["schema_name"],
        "trace_schema_version": config["trace"]["schema_version"],
        "trace_stride": config["trace"]["stride"],
        "entropy_window": generation["entropy_window"],
        "prior_feature_layer": config["representation"]["prior_layer"],
        "prior_layer_indexing": model["layer_indexing"],
        "hidden_layer": "final_transformer_layer",
        "hidden_size": config["representation"]["hidden_size"],
        "prompt_pooling": config["representation"]["prompt_pooling"],
        "prompt_pooling_temperature": config["representation"]["prompt_pooling_temperature"],
        "top_p": generation["top_p"],
        "max_new_tokens": generation["max_new_tokens"],
        "chat_template": generation["chat_template"],
        "output_length_includes_eos": True,
    }
    for name, expected in metadata_expected.items():
        if trace.metadata.get(name) != expected:
            errors.append(f"metadata.{name} does not match the frozen contract")
    hidden_size = int(config["representation"]["hidden_size"])
    if trace.prior_feature.shape != (hidden_size,):
        errors.append("prior feature has the wrong shape")
    if trace.prompt_feature.shape != (hidden_size,):
        errors.append("prompt feature has the wrong shape")
    if trace.decode_hidden_states.shape[1:] != (hidden_size,):
        errors.append("decode features have the wrong hidden size")
    expected_steps = sorted(
        {
            1,
            trace.output_tokens,
            *range(
                int(config["trace"]["stride"]),
                trace.output_tokens + 1,
                int(config["trace"]["stride"]),
            ),
        }
    )
    if trace.steps.tolist() != expected_steps:
        errors.append("saved points do not follow first/stride/terminal schedule")
    if model["dtype"] not in str(trace.metadata.get("dtype")):
        errors.append("trace inference dtype does not match")
    if errors:
        raise ValueError("Hybrid trace contract mismatch: " + "; ".join(errors))


def load_complete_hybrid_split(
    config: dict[str, Any],
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    split: str,
    trace_root: Path | None = None,
) -> list[LoadedHybridTrace]:
    root = trace_root or Path(config["outputs"]["trace_root"])
    jobs = list(rollout_jobs(records, split=split))
    missing = [
        hybrid_trace_path(root, record, seed)
        for record, seed in jobs
        if not hybrid_trace_path(root, record, seed).is_file()
    ]
    if missing:
        raise ValueError(
            f"{split} Hybrid collection missing {len(missing)} of {len(jobs)}; "
            f"first missing: {missing[0]}"
        )
    loaded: list[LoadedHybridTrace] = []
    for record, seed in jobs:
        path = hybrid_trace_path(root, record, seed)
        trace = read_hybrid_trace(path)
        validate_hybrid_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        loaded.append((record, seed, path, trace))
    return loaded


def partition_censored(
    loaded: list[LoadedHybridTrace],
) -> tuple[list[LoadedHybridTrace], int]:
    effective = [item for item in loaded if item[3].stop_reason != "max_new_tokens"]
    return effective, len(loaded) - len(effective)


def enforce_censoring_policy(
    *, loaded_count: int, censored_count: int, warning_rate: float, abort_rate: float
) -> dict[str, Any]:
    if loaded_count <= 0 or not 0 <= censored_count <= loaded_count:
        raise ValueError("invalid censoring counts")
    if not 0 <= warning_rate < abort_rate <= 1:
        raise ValueError("invalid censoring thresholds")
    rate = censored_count / loaded_count
    status = "abort" if rate >= abort_rate else "warning" if rate >= warning_rate else "pass"
    report = {
        "loaded_trace_count": loaded_count,
        "censored_trace_count": censored_count,
        "censoring_rate": rate,
        "warning_rate": warning_rate,
        "abort_rate": abort_rate,
        "status": status,
    }
    if status == "abort":
        raise RuntimeError(
            f"right-censoring rate {rate:.3%} reaches abort threshold {abort_rate:.3%}"
        )
    return report


def hybrid_dataset_digest(loaded: list[LoadedHybridTrace]) -> str:
    canonical = "\n".join(
        f"{trace.prompt_id}\t{seed}\t{file_sha256(path)}" for _, seed, path, trace in loaded
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def hybrid_samples(loaded: list[LoadedHybridTrace]) -> list[HybridSample]:
    samples = []
    for record, _, _, trace in loaded:
        samples.extend(
            build_hybrid_samples(
                trace,
                prompt_family_id=record["prompt_family_id"],
                intended_length=record["intended_length"],
            )
        )
    return samples
