"""Deterministically build the Hybrid v3 Train plus unopened Test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.hybrid_manifest import (
    build_hybrid_v3_records,
    write_hybrid_v3_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-manifest", type=Path, default=Path("data/prompts/alps_v1_prompts.jsonl")
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/alps_plp_hybrid_v3_prompts.jsonl"),
    )
    args = parser.parse_args()
    records = build_hybrid_v3_records(args.v1_manifest)
    if args.check:
        expected = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale Hybrid v3 manifest: {args.output}")
        print(f"validated {len(records)} frozen prompts in {args.output}")
        return
    output = write_hybrid_v3_manifest(args.output, records)
    print(f"wrote {len(records)} prompts to {output}")


if __name__ == "__main__":
    main()
