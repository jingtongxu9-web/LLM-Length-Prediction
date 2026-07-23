"""Download the exact frozen Qwen model/tokenizer snapshot and write its revision marker."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.experiment import load_experiment

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen2.5-7B-Instruct"))
    args = parser.parse_args()
    experiment = load_experiment(args.experiment)
    repo_id = experiment["model"]["id"]
    revision = experiment["model"]["revision"]

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("install the Hugging Face dependencies before downloading") from error

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=args.output,
    )
    marker = args.output / ".frozen_revision"
    with marker.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(revision + "\n")
    print(f"downloaded {repo_id}@{revision} to {args.output}")


if __name__ == "__main__":
    main()
