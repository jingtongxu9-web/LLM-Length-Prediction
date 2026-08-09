"""Fit and freeze the prompt-token Ridge comparator on all Hybrid Train traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_dataset_digest,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.prompt_token_baseline import (
    METHOD_ID,
    fit_prompt_token_ridge,
)

DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_main_comparison.json")


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "alps-plp-four-method-main-comparison-2026":
        raise ValueError("unexpected four-method comparison protocol")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    protocol = _load_protocol(args.protocol)
    config_path = Path(protocol["method_config"])
    config = load_hybrid_config(config_path)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    run_root = Path(protocol["outputs"]["run_root"])
    oof_path = run_root / protocol["outputs"]["oof_report"]
    if not oof_path.is_file():
        raise SystemExit(f"run evaluate_main_comparison_oof.py first; missing {oof_path}")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    if oof.get("test_opened") is not False or METHOD_ID not in oof.get("methods", {}):
        raise SystemExit("four-method OOF report is incomplete")
    if oof.get("protocol_sha256") != file_sha256(args.protocol):
        raise SystemExit("four-method protocol changed after OOF")
    if oof.get("config_sha256") != file_sha256(config_path):
        raise SystemExit("method config changed after OOF")

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
    if digest != oof.get("training_dataset_digest"):
        raise SystemExit("Train traces changed after four-method OOF")
    alpha = float(protocol["methods"][METHOD_ID]["ridge_alpha"])
    fitted = fit_prompt_token_ridge(
        [trace.prompt_tokens for _, _, _, trace in effective],
        [trace.output_tokens for _, _, _, trace in effective],
        alpha=alpha,
    )
    output = args.output_dir or run_root / "models"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "prompt_token_ridge.json"
    model_path.write_text(
        json.dumps(
            {
                **fitted.to_dict(),
                "method_id": METHOD_ID,
                "input_feature": "formatted_prompt_token_count",
                "decode_prediction": "max(predicted_total_output_tokens-step,0)",
                "ridge_alpha": alpha,
                "training_trace_count": len(effective),
                "training_dataset_digest": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_registry = Path(
        "artifacts/runs/alps_plp_hybrid_versions/models/model_registry.json"
    )
    registry = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "test_opened": False,
        "protocol_sha256": file_sha256(args.protocol),
        "config_sha256": file_sha256(config_path),
        "oof_report_sha256": file_sha256(oof_path),
        "training_dataset_digest": digest,
        "censoring": censoring,
        "method": METHOD_ID,
        "files": {model_path.name: file_sha256(model_path)},
        "source_hybrid_registry_sha256": (
            file_sha256(source_registry) if source_registry.is_file() else None
        ),
        "holdout_status": "no_new_hybrid_holdout_has_been_opened",
    }
    (output / "baseline_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fitted and froze {METHOD_ID} on {len(effective)} Train traces in {output}")


if __name__ == "__main__":
    main()
