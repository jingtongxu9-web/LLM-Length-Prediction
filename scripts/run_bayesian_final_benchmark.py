"""Run the frozen seven-method final benchmark once without selection or tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.stage8_benchmark import run_final_benchmark
from llm_length_prediction.stage8_freeze import final_holdout_gate_report

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-trace-hashes", action="store_true")
    args = parser.parse_args()
    gate = final_holdout_gate_report(args.config, model_root=args.model_root)
    if not gate["ready"]:
        raise SystemExit("final holdout gate is blocked: " + "; ".join(gate["failures"]))
    import json

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_root = args.model_root or Path(config["outputs"]["model_root"])
    output_dir = args.output_dir or Path(config["outputs"]["benchmark_run_root"])
    report = run_final_benchmark(
        args.config,
        model_root=model_root,
        output_dir=output_dir,
        device=args.device,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    print(
        f"Stage-8 final benchmark {report['status']}; "
        f"primary={report['primary_method']}; report={output_dir / 'final_benchmark_report.json'}"
    )


if __name__ == "__main__":
    main()
