"""Verify the downloaded Stage-8B archive and write committable final-result evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.stage8_archive import (
    build_summary,
    read_verified_archive,
    write_outputs,
)

DEFAULT_OUTPUT = Path("docs/results/bayesian_sequential")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--completed-date", default="2026-08-16")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle = read_verified_archive(
        args.archive,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    summary = build_summary(bundle, completed_date=args.completed_date)
    paths = write_outputs(summary, args.output_dir)
    print(
        f"archived Stage-8 final benchmark: files={len(paths)} "
        f"verified={bundle['internal_manifest_entry_count']} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
