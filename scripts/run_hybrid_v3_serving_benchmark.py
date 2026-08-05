"""Run the preregistered deterministic serving replay on final-Test predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from llm_length_prediction.hybrid_experiment import (
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
)
from llm_length_prediction.models.hybrid_suite import METHOD_IDS

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_v3_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_root = args.run_root or Path(config["outputs"]["run_root"])
    final_report = run_root / "final_test" / "final_report.json"
    predictions_path = run_root / "final_test" / "predictions.csv"
    if not final_report.is_file() or not predictions_path.is_file():
        raise SystemExit("run the one-time final evaluation before serving replay")
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="test", trace_root=args.trace_root
    )
    durations = {
        (trace.prompt_id, seed): float(trace.duration_ms or 0.0)
        for _, seed, _, trace in loaded
        if trace.stop_reason != "max_new_tokens"
    }
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["step"]) == 1]
    simulator = protocol["serving_evaluation"]["simulator"]
    batch_size = int(simulator["batch_size"])
    boundaries = [int(value) for value in simulator["length_bucket_boundaries"]]
    quantum = int(simulator["kv_allocation_quantum_tokens"])
    metrics = {}
    for method in METHOD_IDS:
        requests = []
        for row in rows:
            predicted_total = max(1.0, 1.0 + float(row[method]))
            bucket = next(
                (index for index, boundary in enumerate(boundaries) if predicted_total <= boundary),
                len(boundaries),
            )
            allocated = math.ceil(predicted_total / quantum) * quantum
            key = (row["prompt_id"], int(row["seed"]))
            requests.append(
                {
                    "key": key,
                    "bucket": bucket,
                    "actual": int(row["output_tokens"]),
                    "allocated": allocated,
                    "duration": durations[key],
                }
            )
        requests.sort(key=lambda item: (item["bucket"], *item["key"]))
        clock = 0.0
        completion = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            clock += max(float(item["duration"]) for item in batch)
            completion.extend([clock] * len(batch))
        actual = np.asarray([item["actual"] for item in requests], dtype=np.float64)
        allocated = np.asarray([item["allocated"] for item in requests], dtype=np.float64)
        completion_array = np.asarray(completion, dtype=np.float64)
        metrics[method] = {
            "request_count": len(requests),
            "mean_completion_time_ms": float(completion_array.mean()),
            "p95_completion_time_ms": float(np.quantile(completion_array, 0.95)),
            "throughput_tokens_per_second": float(actual.sum() / (clock / 1000.0)),
            "kv_cache_overreservation_tokens": float(np.maximum(allocated - actual, 0).sum()),
            "kv_cache_overreservation_rate": float(
                np.maximum(allocated - actual, 0).sum() / actual.sum()
            ),
            "underallocation_rate": float(np.mean(allocated < actual)),
        }
    hybrid = metrics["alps_plp_hybrid_v3"]
    dominates = all(
        hybrid["mean_completion_time_ms"] < metrics[method]["mean_completion_time_ms"]
        and hybrid["kv_cache_overreservation_tokens"]
        < metrics[method]["kv_cache_overreservation_tokens"]
        and hybrid["underallocation_rate"] <= metrics[method]["underallocation_rate"]
        for method in METHOD_IDS[:-1]
    )
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "simulator": simulator,
        "metrics": metrics,
        "hybrid_dominates_all_comparators_on_preregistered_serving_rule": dominates,
        "scope": "deterministic offline replay; not a production serving-system measurement",
    }
    output = run_root / protocol["outputs"]["serving"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "serving_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"serving replay complete; Hybrid dominates={dominates}; output={output}")


if __name__ == "__main__":
    main()
