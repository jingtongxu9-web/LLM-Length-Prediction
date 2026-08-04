"""Train hidden-state PLP v2 on newly collected Train traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from llm_length_prediction.evaluation.plp import (
    hidden_state_plp_group_breakdown,
    hidden_state_plp_metrics,
    hidden_state_plp_progress_breakdown,
)
from llm_length_prediction.models.plp import (
    HiddenStatePLPSample,
    build_hidden_state_plp_samples,
    fit_hidden_state_plp,
    predict_hidden_state_plp,
)
from llm_length_prediction.plp_experiment import (
    load_complete_plp_split,
    load_plp_base_experiment,
    load_plp_config,
    partition_censored_plp_traces,
    plp_dataset_digest,
    validate_plp_config,
)

DEFAULT_CONFIG = Path("configs/experiments/plp_v2_manifest.json")


def atomic_torch_save(payload: object, path: Path, torch_module: object) -> None:
    """Write a checkpoint atomically so an interrupted save cannot replace a valid file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch_module.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_predictions(
    path: Path, samples: list[HiddenStatePLPSample], predictions: list[float]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "prompt_family_id",
                "task",
                "intended_length",
                "seed",
                "step",
                "decode_progress",
                "actual_output_tokens",
                "actual_remaining_tokens",
                "predicted_remaining_tokens",
                "predicted_total_tokens",
            ],
        )
        writer.writeheader()
        for sample, prediction in zip(samples, predictions, strict=True):
            writer.writerow(
                {
                    "prompt_id": sample.prompt_id,
                    "prompt_family_id": sample.prompt_family_id,
                    "task": sample.task,
                    "intended_length": sample.intended_length,
                    "seed": sample.seed,
                    "step": sample.step,
                    "decode_progress": sample.step / sample.output_tokens,
                    "actual_output_tokens": sample.output_tokens,
                    "actual_remaining_tokens": sample.remaining_tokens,
                    "predicted_remaining_tokens": prediction,
                    "predicted_total_tokens": sample.step + prediction,
                }
            )


def write_evaluation(
    path: Path,
    *,
    split: str,
    samples: list[HiddenStatePLPSample],
    predictions: list[float],
    trace_accounting: dict[str, int] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "split": split,
                "prediction_target": "remaining_tokens",
                "primary_weighting": "sequence_balanced",
                "trace_accounting": trace_accounting or {},
                "overall": hidden_state_plp_metrics(samples, predictions),
                "by_decode_progress": hidden_state_plp_progress_breakdown(
                    samples, predictions
                ),
                "by_task": hidden_state_plp_group_breakdown(
                    samples, predictions, group_by=("task",)
                ),
                "by_intended_length": hidden_state_plp_group_breakdown(
                    samples, predictions, group_by=("intended_length",)
                ),
                "by_task_and_intended_length": hidden_state_plp_group_breakdown(
                    samples,
                    predictions,
                    group_by=("task", "intended_length"),
                ),
                "by_seed": hidden_state_plp_group_breakdown(
                    samples, predictions, group_by=("seed",)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_predictions(path.with_suffix(".csv"), samples, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_plp_config(args.config)
    experiment, records = load_plp_base_experiment(config)
    validate_plp_config(config, experiment)
    loaded = load_complete_plp_split(
        config, experiment, records, split="train", trace_root=args.trace_root
    )
    loaded_trace_count = len(loaded)
    effective_loaded, excluded_censored_trace_count = partition_censored_plp_traces(
        loaded,
        exclude_censored=config["trace"]["exclude_max_new_tokens_traces"],
    )
    training_dataset_sha256 = plp_dataset_digest(effective_loaded)
    samples = [
        sample
        for record, _, _, trace in effective_loaded
        for sample in build_hidden_state_plp_samples(
            trace,
            prompt_family_id=record["prompt_family_id"],
            intended_length=record["intended_length"],
            exclude_censored=False,
        )
    ]
    if not samples:
        raise SystemExit("Train PLP traces contain no uncensored samples")
    effective_training_trace_count = len(effective_loaded)
    del loaded, effective_loaded

    head_config = config["prediction_head"]
    training = config["training"]
    head, report = fit_hidden_state_plp(
        samples,
        num_bins=head_config["num_bins"],
        target_percentiles=tuple(head_config["target_range_percentiles"]),
        lambda_ce=head_config["lambda_ce"],
        dropout=head_config["dropout"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        seed=training["seed"],
        device=training["device"],
    )
    output_dir = args.output_dir or Path(config["outputs"]["run_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    checkpoint_metadata = {
        "schema_version": 1,
        "method_id": config["method_id"],
        "base_experiment_id": experiment["experiment_id"],
        "model_revision": experiment["model"]["revision"],
        "tokenizer_revision": experiment["model"]["tokenizer_revision"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "method_config_sha256": config_sha,
        "training_dataset_sha256": training_dataset_sha256,
        "fit_split": "train",
        "loaded_trace_count": loaded_trace_count,
        "effective_training_trace_count": effective_training_trace_count,
        "excluded_censored_trace_count": excluded_censored_trace_count,
        **{key: report[key] for key in ("input_dim", "num_bins", "target_range")},
        "dropout": head_config["dropout"],
        "trainable_parameter_count": report["trainable_parameter_count"],
    }
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PLP checkpoint writing requires PyTorch") from error
    state_dict = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    checkpoint = output_dir / config["outputs"]["checkpoint"]
    atomic_torch_save(
        {"metadata": checkpoint_metadata, "state_dict": state_dict}, checkpoint, torch
    )
    (output_dir / "method_config.json").write_bytes(args.config.read_bytes())
    report.update(
        {
            "method_id": config["method_id"],
            "base_experiment_id": experiment["experiment_id"],
            "loaded_trace_count": loaded_trace_count,
            "effective_training_trace_count": effective_training_trace_count,
            "excluded_censored_trace_count": excluded_censored_trace_count,
            "training_dataset_sha256": checkpoint_metadata["training_dataset_sha256"],
            "sequence_balanced_loss": True,
        }
    )
    (output_dir / config["outputs"]["training_report"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    predictions, _ = predict_hidden_state_plp(
        head,
        samples,
        batch_size=training["batch_size"],
        device=report["device"],
    )
    write_evaluation(
        output_dir / "train_evaluation.json",
        split="train",
        samples=samples,
        predictions=[float(value) for value in predictions],
        trace_accounting={
            "loaded_trace_count": loaded_trace_count,
            "effective_trace_count": effective_training_trace_count,
            "excluded_censored_trace_count": excluded_censored_trace_count,
        },
    )
    print(
        f"fitted Hidden-State PLP v2 on {effective_training_trace_count} effective "
        f"Train traces ({excluded_censored_trace_count} censored excluded) / "
        f"{len(samples)} points: {checkpoint}"
    )


if __name__ == "__main__":
    main()
