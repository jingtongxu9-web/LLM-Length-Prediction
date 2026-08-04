from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

from llm_length_prediction.data.plp import (
    PLP_TRACE_SCHEMA_VERSION,
    PLPHiddenStateTrace,
    plp_trace_path,
    read_plp_trace,
)
from llm_length_prediction.experiment import (
    file_sha256,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
)

LoadedPLPTrace: TypeAlias = tuple[dict[str, Any], int, Path, PLPHiddenStateTrace]


def load_plp_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported PLP config schema_version")
    return config


def load_plp_base_experiment(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment = load_experiment(config["base_experiment"])
    if experiment["experiment_id"] != config["base_experiment_id"]:
        raise ValueError("PLP base_experiment_id does not match the frozen experiment")
    return experiment, load_frozen_prompts(experiment)


def validate_plp_config(config: dict[str, Any], experiment: dict[str, Any]) -> None:
    errors = []
    if config.get("method_id") != "hidden-state-plp-v2":
        errors.append("method_id must be hidden-state-plp-v2")
    if config.get("method_role") != "paper_aligned_plp_only_nonexact":
        errors.append("method_role must be paper_aligned_plp_only_nonexact")
    scope = config.get("scope", {})
    expected_scope = {
        "uses_existing_rollouts": False,
        "requires_generation_rerun": True,
        "uses_alps_prior": False,
        "uses_prompt_hidden_state": True,
        "uses_decode_hidden_state": True,
        "plp_only": True,
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            errors.append(f"scope.{name} must be {expected!r}")
    representation = config.get("representation", {})
    expected_representation = {
        "hidden_layer": "final_transformer_layer",
        "hidden_size": 3584,
        "input_dim": 7168,
        "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
        "prompt_pooling_temperature": 1.0,
        "dynamic_aggregation": "concat_prompt_pool_with_current_causal_hidden_state",
        "storage_dtype": "float32",
        "training_dtype": "float32",
    }
    for name, expected in expected_representation.items():
        if representation.get(name) != expected:
            errors.append(f"representation.{name} must be {expected!r}")
    trace = config.get("trace", {})
    if int(trace.get("stride", -1)) != int(experiment["generation"]["trace_stride"]):
        errors.append("PLP trace stride must match the frozen update frequency")
    expected_trace = {
        "schema_version": PLP_TRACE_SCHEMA_VERSION,
        "include_first_point": True,
        "include_terminal_point": True,
        "sequence_balanced_loss": True,
        "exclude_max_new_tokens_traces": True,
    }
    for name, expected in expected_trace.items():
        if trace.get(name) != expected:
            errors.append(f"trace.{name} must be {expected!r}")
    target = config.get("target", {})
    if target.get("name") != "remaining_tokens" or target.get("transform") != "none":
        errors.append("target must be untransformed remaining_tokens")
    training = config.get("training", {})
    expected_training = {
        "optimizer": "adamw",
        "seed": 42,
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "device": "auto",
        "hyperparameter_selection": "none",
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected:
            errors.append(f"training.{name} must be {expected!r}")
    head = config.get("prediction_head", {})
    expected_head = {
        "type": "soft_label_length_bins",
        "num_bins": 20,
        "target_range_percentiles": [1.0, 99.0],
        "architecture": "Linear(2d,d)-LayerNorm-ReLU-Dropout-Linear(d,20)",
        "dropout": 0.1,
        "lambda_ce": 0.95,
        "lambda_mse": 0.05,
    }
    for name, expected in expected_head.items():
        if head.get(name) != expected:
            errors.append(f"prediction_head.{name} must be {expected!r}")
    outputs = config.get("outputs", {})
    for name in ("trace_root", "run_root", "checkpoint", "training_report"):
        if not outputs.get(name):
            errors.append(f"outputs.{name} must be non-empty")
    if errors:
        raise ValueError("PLP v2 contract mismatch: " + "; ".join(errors))


def validate_plp_trace(
    trace: PLPHiddenStateTrace,
    *,
    record: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    trace.validate()
    errors = []
    generation = experiment["generation"]
    model = experiment["model"]
    expected = {
        "prompt_id": (trace.prompt_id, record["prompt_id"]),
        "task": (trace.task, record["task_type"]),
        "seed": (trace.seed, seed),
        "temperature": (trace.temperature, generation["temperature"]),
        "model_revision": (trace.model_revision, model["revision"]),
        "tokenizer_revision": (trace.tokenizer_revision, model["tokenizer_revision"]),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            errors.append(f"{name}={actual!r}, expected {wanted!r}")
    metadata_expected = {
        "method_id": config["method_id"],
        "base_experiment_id": experiment["experiment_id"],
        "prompt_family_id": record["prompt_family_id"],
        "intended_length": record["intended_length"],
        "split": record["split"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "top_p": generation["top_p"],
        "max_new_tokens": generation["max_new_tokens"],
        "trace_stride": config["trace"]["stride"],
        "trace_schema_version": config["trace"]["schema_version"],
        "chat_template": generation["chat_template"],
        "hidden_layer": "final_transformer_layer",
        "hidden_size": config["representation"]["hidden_size"],
        "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
        "prompt_pooling_temperature": config["representation"][
            "prompt_pooling_temperature"
        ],
        "dynamic_aggregation": "concat_prompt_pool_with_current_causal_hidden_state",
        "storage_dtype": config["representation"]["storage_dtype"],
        "output_length_includes_eos": generation["output_length_includes_eos"],
    }
    for name, wanted in metadata_expected.items():
        if trace.metadata.get(name) != wanted:
            errors.append(f"metadata.{name}={trace.metadata.get(name)!r}, expected {wanted!r}")
    if trace.output_tokens > int(generation["max_new_tokens"]):
        errors.append("output_tokens exceeds max_new_tokens")
    if model["dtype"] not in str(trace.metadata.get("dtype")):
        errors.append(f"metadata.dtype={trace.metadata.get('dtype')!r}, expected {model['dtype']}")
    expected_hidden_size = int(config["representation"]["hidden_size"])
    if trace.prompt_feature.shape != (expected_hidden_size,):
        errors.append(
            f"prompt_feature has shape {trace.prompt_feature.shape}, "
            f"expected ({expected_hidden_size},)"
        )
    if trace.decode_hidden_states.shape[1:] != (expected_hidden_size,):
        errors.append(
            "decode_hidden_states hidden dimension does not match the frozen representation"
        )
    if int(trace.steps[0]) != 1:
        errors.append("PLP trace must include the first generated token")
    if int(trace.steps[-1]) != trace.output_tokens:
        errors.append("PLP trace must include the final generated token")
    stride = int(config["trace"]["stride"])
    expected_steps = sorted(
        {1, trace.output_tokens, *range(stride, trace.output_tokens + 1, stride)}
    )
    actual_steps = [int(step) for step in trace.steps]
    if actual_steps != expected_steps:
        errors.append(
            "PLP steps do not match the frozen first/stride/final schedule: "
            f"expected {expected_steps[:5]}...{expected_steps[-3:]}, "
            f"got {actual_steps[:5]}...{actual_steps[-3:]}"
        )
    if errors:
        raise ValueError("PLP trace contract mismatch: " + "; ".join(errors))


def load_complete_plp_split(
    config: dict[str, Any],
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    split: str,
    trace_root: Path | None = None,
) -> list[LoadedPLPTrace]:
    root = trace_root or Path(config["outputs"]["trace_root"])
    jobs = list(rollout_jobs(records, split=split))
    missing = [
        plp_trace_path(root, record, seed)
        for record, seed in jobs
        if not plp_trace_path(root, record, seed).is_file()
    ]
    if missing:
        raise ValueError(
            f"{split} PLP collection is incomplete: missing {len(missing)} of {len(jobs)}; "
            f"first missing path: {missing[0]}"
        )
    loaded = []
    for record, seed in jobs:
        path = plp_trace_path(root, record, seed)
        trace = read_plp_trace(path)
        validate_plp_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        loaded.append((record, seed, path, trace))
    return loaded


def partition_censored_plp_traces(
    loaded: list[LoadedPLPTrace], *, exclude_censored: bool
) -> tuple[list[LoadedPLPTrace], int]:
    """Return effective traces and the number excluded as right-censored."""

    if not exclude_censored:
        return list(loaded), 0
    effective = [item for item in loaded if item[3].stop_reason != "max_new_tokens"]
    return effective, len(loaded) - len(effective)


def plp_dataset_digest(
    loaded: list[LoadedPLPTrace],
) -> str:
    canonical = "\n".join(
        f"{trace.prompt_id}\t{seed}\t{file_sha256(path)}"
        for _, seed, path, trace in loaded
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
