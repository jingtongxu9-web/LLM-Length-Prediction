"""Select terminal-zero PLP v3 from grouped OOF and freeze PLP-only models."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.hybrid import family_bootstrap_interval
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_dataset_digest,
    hybrid_samples,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.plp_v3 import (
    PLP_V3_METHOD_IDS,
    fit_plp_v3,
    load_plp_v3,
    method_settings,
    save_plp_v3,
)

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/plp_terminal_v3_protocol.json")
DEFAULT_HYBRID_REGISTRY = Path(
    "artifacts/runs/alps_plp_hybrid_v3/models/model_registry.json"
)


def _family_trace_mae(rows: list[dict[str, str]], method_id: str) -> dict[str, float]:
    traces: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["prompt_family_id"], row["prompt_id"], int(row["seed"]))
        traces[key].append(abs(float(row[method_id]) - float(row["remaining_tokens"])))
    families: dict[str, list[float]] = defaultdict(list)
    for (family, _, _), errors in traces.items():
        families[family].append(float(np.mean(errors)))
    return {family: float(np.mean(values)) for family, values in families.items()}


def _selection_report(
    protocol: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    selection = protocol["selection"]
    report_path = Path(selection["source_report"])
    predictions_path = Path(selection["source_predictions"])
    if not report_path.is_file() or not predictions_path.is_file():
        raise ValueError("the completed grouped-OOF report and predictions are required")
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if source.get("split") != "train_grouped_oof" or source.get("test_opened") is not False:
        raise ValueError("selection evidence must be unopened-Test grouped OOF")
    if source.get("config_sha256") != file_sha256(config_path):
        raise ValueError("shared trace config changed after grouped OOF")
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "prompt_family_id",
        "prompt_id",
        "seed",
        "remaining_tokens",
        str(selection["control_id"]),
        *[str(value) for value in selection["candidate_ablation_ids"]],
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("OOF predictions do not contain every PLP ablation column")
    control_id = str(selection["control_id"])
    control = _family_trace_mae(rows, control_id)
    comparisons = {}
    for index, candidate in enumerate(selection["candidate_ablation_ids"]):
        candidate = str(candidate)
        candidate_values = _family_trace_mae(rows, candidate)
        if candidate_values.keys() != control.keys():
            raise ValueError("PLP ablations do not cover identical prompt families")
        delta = {
            family: candidate_values[family] - control[family] for family in control
        }
        comparisons[candidate] = {
            "candidate_family_macro_mae_tokens": float(np.mean(list(candidate_values.values()))),
            "control_family_macro_mae_tokens": float(np.mean(list(control.values()))),
            "paired_ci_95": family_bootstrap_interval(
                delta,
                replicates=int(selection["bootstrap_replicates"]),
                confidence=0.95,
                seed=int(selection["bootstrap_seed"]) + index,
            ),
            "paired_familywise_ci": family_bootstrap_interval(
                delta,
                replicates=int(selection["bootstrap_replicates"]),
                confidence=float(selection["familywise_confidence_level"]),
                seed=int(selection["bootstrap_seed"]) + 100 + index,
            ),
        }
    selected = min(
        comparisons,
        key=lambda name: comparisons[name]["candidate_family_macro_mae_tokens"],
    )
    if (
        selected != "plp_terminal_zero_v3"
        or comparisons[selected]["paired_familywise_ci"]["upper"] >= 0
    ):
        raise ValueError("the frozen PLP v3 selection rule did not choose terminal-zero")
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "train_grouped_oof",
        "test_opened": False,
        "source_report_sha256": file_sha256(report_path),
        "source_predictions_sha256": file_sha256(predictions_path),
        "training_dataset_digest": source["training_dataset_digest"],
        "family_count": len(control),
        "selected_method": selected,
        "selection_rule": selection["selection_rule"],
        "comparisons_to_plp_v2": comparisons,
    }


def _import_hybrid_models(
    source_registry_path: Path,
    model_dir: Path,
    *,
    training_digest: str,
) -> dict[str, str]:
    registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    if registry.get("training_dataset_digest") != training_digest:
        raise ValueError("Hybrid and PLP-only model registries use different Train traces")
    files: dict[str, str] = {}
    for method_id in PLP_V3_METHOD_IDS:
        source_name = registry["methods"][method_id]["files"][0]
        source = source_registry_path.parent / source_name
        expected = registry["methods"][method_id]["sha256"][source_name]
        if file_sha256(source) != expected:
            raise ValueError(f"Hybrid checkpoint hash mismatch: {source}")
        target = model_dir / f"{method_id}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[method_id] = target.name
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--reuse-hybrid-models",
        type=Path,
        nargs="?",
        const=DEFAULT_HYBRID_REGISTRY,
        help="reuse the identical two checkpoints from a completed Hybrid model registry",
    )
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if tuple(protocol.get("methods", {})) != PLP_V3_METHOD_IDS:
        raise ValueError("PLP-only protocol must contain exactly v2 control and terminal-zero v3")
    selection_report = _selection_report(protocol, args.config)
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="train", trace_root=args.trace_root
    )
    effective, censored = partition_censored(loaded)
    censoring = enforce_censoring_policy(
        loaded_count=len(loaded),
        censored_count=censored,
        warning_rate=float(config["censoring"]["warning_rate"]),
        abort_rate=float(config["censoring"]["abort_rate"]),
    )
    digest = hybrid_dataset_digest(effective)
    if digest != selection_report["training_dataset_digest"]:
        raise ValueError("Train traces changed after grouped OOF")
    run_root = Path(protocol["outputs"]["run_root"])
    selection_path = run_root / protocol["outputs"]["selection_report"]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model_dir = run_root / protocol["outputs"]["models"]
    if args.reuse_hybrid_models:
        files = _import_hybrid_models(
            args.reuse_hybrid_models, model_dir, training_digest=digest
        )
        fitted = load_plp_v3(model_dir)
        source = {
            "type": "verified_hybrid_checkpoint_import",
            "registry": str(args.reuse_hybrid_models),
        }
    else:
        fitted = fit_plp_v3(
            hybrid_samples(effective), config=config, protocol=protocol, device=args.device
        )
        files = save_plp_v3(fitted, model_dir)
        source = {"type": "standalone_training"}
    for method_id in PLP_V3_METHOD_IDS:
        hidden_dim, terminal_zero, weighted = method_settings(protocol, method_id)
        metadata = fitted.metadata[method_id]
        if (
            int(metadata["hidden_dim"]) != hidden_dim
            or bool(metadata["terminal_zero_bin"]) != terminal_zero
            or bool(metadata["weighted_target_range"]) != weighted
        ):
            raise ValueError(f"checkpoint does not match frozen {method_id} settings")
    registry = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "test_opened": False,
        "config_sha256": file_sha256(args.config),
        "protocol_sha256": file_sha256(args.protocol),
        "selection_report_sha256": file_sha256(selection_path),
        "training_dataset_digest": digest,
        "censoring": censoring,
        "model_source": source,
        "methods": {
            method_id: {
                "file": files[method_id],
                "sha256": file_sha256(model_dir / files[method_id]),
                "metadata": fitted.metadata[method_id],
            }
            for method_id in PLP_V3_METHOD_IDS
        },
    }
    (model_dir / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"froze PLP-only v3 and its v2 control in {model_dir}")


if __name__ == "__main__":
    main()
