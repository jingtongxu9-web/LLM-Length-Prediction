"""Evaluate hidden-state PLP v2 on newly collected traces."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from train_plp import write_evaluation

from llm_length_prediction.models.plp import (
    build_hidden_state_plp_samples,
    build_plp_head,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("refusing PLP Test evaluation without --confirm-final-test")

    config = load_plp_config(args.config)
    experiment, records = load_plp_base_experiment(config)
    validate_plp_config(config, experiment)
    checkpoint_path = args.checkpoint or (
        Path(config["outputs"]["run_root"]) / config["outputs"]["checkpoint"]
    )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PLP evaluation requires PyTorch") from error
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    expected = {
        "schema_version": 1,
        "method_id": config["method_id"],
        "base_experiment_id": experiment["experiment_id"],
        "model_revision": experiment["model"]["revision"],
        "tokenizer_revision": experiment["model"]["tokenizer_revision"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "method_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "fit_split": "train",
        "input_dim": config["representation"]["input_dim"],
        "num_bins": config["prediction_head"]["num_bins"],
        "dropout": config["prediction_head"]["dropout"],
    }
    mismatches = [
        f"{name}: expected {wanted!r}, got {metadata.get(name)!r}"
        for name, wanted in expected.items()
        if metadata.get(name) != wanted
    ]
    if mismatches:
        raise SystemExit("PLP checkpoint contract mismatch: " + "; ".join(mismatches))

    device_setting = config["training"]["device"]
    device = "cuda" if device_setting == "auto" and torch.cuda.is_available() else device_setting
    if device == "auto":
        device = "cpu"
    head = build_plp_head(
        metadata["input_dim"],
        num_bins=metadata["num_bins"],
        target_range=tuple(metadata["target_range"]),
        dropout=metadata["dropout"],
    )
    head.load_state_dict(payload["state_dict"])
    actual_parameter_count = sum(
        parameter.numel() for parameter in head.parameters() if parameter.requires_grad
    )
    if actual_parameter_count != metadata.get("trainable_parameter_count"):
        raise SystemExit(
            "PLP checkpoint parameter-count mismatch: "
            f"expected {metadata.get('trainable_parameter_count')!r}, "
            f"constructed {actual_parameter_count}"
        )
    head.to(device)

    loaded = load_complete_plp_split(
        config, experiment, records, split=args.split, trace_root=args.trace_root
    )
    loaded_trace_count = len(loaded)
    effective_loaded, excluded_censored_trace_count = partition_censored_plp_traces(
        loaded,
        exclude_censored=config["trace"]["exclude_max_new_tokens_traces"],
    )
    if args.split == "train":
        current_training_digest = plp_dataset_digest(effective_loaded)
        if current_training_digest != metadata.get("training_dataset_sha256"):
            raise SystemExit(
                "PLP Train dataset no longer matches the checkpoint: "
                f"expected {metadata.get('training_dataset_sha256')!r}, "
                f"got {current_training_digest!r}"
            )
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
        raise SystemExit(f"{args.split} PLP traces contain no uncensored samples")
    effective_trace_count = len(effective_loaded)
    del loaded, effective_loaded
    predictions, _ = predict_hidden_state_plp(
        head,
        samples,
        batch_size=config["training"]["batch_size"],
        device=device,
    )
    output = args.output or checkpoint_path.parent / f"{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_evaluation(
        output,
        split=args.split,
        samples=samples,
        predictions=[float(value) for value in predictions],
        trace_accounting={
            "loaded_trace_count": loaded_trace_count,
            "effective_trace_count": effective_trace_count,
            "excluded_censored_trace_count": excluded_censored_trace_count,
        },
    )
    print(
        f"evaluated Hidden-State PLP v2 on {effective_trace_count} effective "
        f"{args.split} traces ({excluded_censored_trace_count} censored excluded) / "
        f"{len(samples)} points: {output}"
    )


if __name__ == "__main__":
    main()
