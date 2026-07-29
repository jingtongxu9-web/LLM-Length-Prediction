"""Build the frozen ALPS v1 prompt manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.prompt_manifest import build_records, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/alps_v1_prompts.jsonl"),
    )
    args = parser.parse_args()
    records = build_records()
    output = write_manifest(args.output, records)
    print(f"wrote {len(records)} prompts to {output}")


if __name__ == "__main__":
    main()
