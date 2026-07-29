"""Create a checksum-backed, immutable ALPS v1 archive inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs/alps_v1"))
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/alps_v1_manifest.json"),
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path("data/prompts/alps_v1_prompts.jsonl"),
    )
    args = parser.parse_args()

    if not args.run_root.is_dir():
        raise SystemExit(f"run root does not exist: {args.run_root}")
    archive_dir = args.run_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inputs = [path for path in (args.experiment, args.prompt_manifest) if path.is_file()]
    outputs = [
        path
        for path in args.run_root.rglob("*")
        if path.is_file() and archive_dir not in path.parents
    ]
    paths = sorted(set(inputs + outputs))
    checksum_lines = [f"{sha256(path)}  {path.as_posix()}" for path in paths]
    (archive_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "experiment_status": "final_test_opened_read_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": command_output(["git", "rev-parse", "HEAD"]),
        "python_version": command_output(["python", "--version"]),
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
        ),
        "file_count": len(paths),
        "experiment_manifest": args.experiment.as_posix(),
        "prompt_manifest": args.prompt_manifest.as_posix(),
        "policy": (
            "ALPS v1 final test has been opened. Do not use it to select alpha, "
            "features, calibration, or any ALPS v2 hyperparameter."
        ),
    }
    (archive_dir / "archive_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"archived {len(paths)} files: {archive_dir}")


if __name__ == "__main__":
    main()
