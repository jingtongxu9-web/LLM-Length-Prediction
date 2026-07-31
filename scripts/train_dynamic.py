"""Train frozen Dynamic-Signal MLP v1 from existing Train traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from llm_length_prediction.comparison import (
    load_base_experiment_for_method,
    load_complete_split_traces,
    load_method_config,
    trace_dataset_digest,
    validate_project_plp_contract,
)
from llm_length_prediction.evaluation.progressive import (
    progress_breakdown,
    progressive_metrics,
)
from llm_length_prediction.models.dynamic import (
    ProgressiveSample,
    build_progressive_samples,
    fit_plp_mlp,
)

DEFAULT_CONFIG = Path("configs/experiments/plp_v1_manifest.json")


def _write_predictions(
    path: Path,
    samples: list[ProgressiveSample],
    predicted_mu: list[float],
    predicted_remaining: list[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "prompt_family_id",
                "seed",
                "step",
                "decode_progress",
                "actual_output_tokens",
                "actual_remaining_tokens",
                "predicted_log1p_remaining_mu",
                "predicted_mean_remaining_tokens",
                "predicted_mean_total_tokens",
            ],
        )
        writer.writeheader()
        for sample, mu, remaining in zip(
            samples, predicted_mu, predicted_remaining, strict=True
        ):
            writer.writerow(
                {
                    "prompt_id": sample.prompt_id,
                    "prompt_family_id": sample.prompt_family_id,
                    "seed": sample.seed,
                    "step": sample.step,
                    "decode_progress": sample.step / sample.output_tokens,
                    "actual_output_tokens": sample.output_tokens,
                    "actual_remaining_tokens": sample.remaining_tokens,
                    "predicted_log1p_remaining_mu": mu,
                    "predicted_mean_remaining_tokens": remaining,
                    "predicted_mean_total_tokens": sample.step + remaining,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_method_config(args.config)
    _, experiment, records = load_base_experiment_for_method(config)
    validate_project_plp_contract(config, experiment)
    loaded = load_complete_split_traces(
        experiment,
        records,
        split="train",
        trace_root=args.trace_root,
    )
    samples = [
        sample
        for record, _, _, trace in loaded
        for sample in build_progressive_samples(
            trace,
            prompt_family_id=record["prompt_family_id"],
        )
    ]
    if not samples:
        raise SystemExit("Train traces contain no non-terminal dynamic samples")

    model_config = config["model"]
    training = config["training"]
    model, training_report = fit_plp_mlp(
        [sample.features for sample in samples],
        [sample.remaining_tokens for sample in samples],
        [sample.sequence_weight for sample in samples],
        hidden_sizes=model_config["hidden_sizes"],
        dropout=float(model_config["dropout"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        seed=int(training["seed"]),
        device=str(training["device"]),
    )
    predicted_mu = [
        float(value)
        for value in model.predict_mu_many([sample.features for sample in samples])
    ]
    predicted_remaining = [
        float(value)
        for value in model.predict_remaining_many(
            [sample.features for sample in samples]
        )
    ]

    output_dir = args.output_dir or Path(config["outputs"]["run_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model.to_dict()
    model_payload.update(
        {
            "method_id": config["method_id"],
            "experiment_id": experiment["experiment_id"],
            "model_revision": experiment["model"]["revision"],
            "tokenizer_revision": experiment["model"]["tokenizer_revision"],
            "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
            "fit_split": "train",
            "training_trace_count": len(loaded),
            "training_sample_count": len(samples),
            "training_dataset_sha256": trace_dataset_digest(loaded),
            "method_config_path": str(args.config),
            "method_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        }
    )
    model_path = output_dir / config["outputs"]["model"]
    model_path.write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        **training_report,
        "method_id": config["method_id"],
        "experiment_id": experiment["experiment_id"],
        "fit_split": "train",
        "sequence_balanced_loss": True,
    }
    (output_dir / config["outputs"]["training_report"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evaluation = {
        "split": "train",
        "prediction_target": "remaining_tokens",
        "overall": progressive_metrics(
            samples,
            predicted_remaining,
            predicted_mu,
            model.residual_variance,
        ),
        "by_decode_progress": progress_breakdown(
            samples,
            predicted_remaining,
            predicted_mu,
            model.residual_variance,
        ),
    }
    (output_dir / "train_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        output_dir / "train_evaluation.csv",
        samples,
        predicted_mu,
        predicted_remaining,
    )
    print(
        f"fitted Dynamic-Signal MLP v1 on {len(loaded)} Train traces / "
        f"{len(samples)} non-terminal points: {model_path}"
    )


if __name__ == "__main__":
    main()
