"""Run checks and irreversibly assign the shared holdout to PLP-only v3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.plp_v3_gate import begin_plp_v3_test_access

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/plp_terminal_v3_protocol.json")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(f"pre-open check failed ({result.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_test:
        raise SystemExit("refusing to consume the shared holdout without --confirm-final-test")
    _run([sys.executable, "-m", "pytest"])
    _run([sys.executable, "-m", "ruff", "check", "--no-cache", "."])
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_root = Path(protocol["outputs"]["run_root"])
    try:
        repo_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repo_commit = "unavailable"
    checks = {
        "schema_version": 1,
        "pytest": "passed",
        "ruff": "passed",
        "config_sha256": file_sha256(args.config),
        "protocol_sha256": file_sha256(args.protocol),
        "repo_commit": repo_commit,
    }
    path = run_root / "validation" / "pre_open_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    marker = begin_plp_v3_test_access(
        protocol_path=args.protocol, config_path=args.config, trace_root=args.trace_root
    )
    print(f"PLP-ONLY FINAL TEST OPENED ONCE at {marker['opened_at_utc']}")
    print("The same 12 families can no longer be an untouched Hybrid final Test.")


if __name__ == "__main__":
    main()
