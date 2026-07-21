import hashlib
import json
from pathlib import Path


def test_frozen_experiment_manifest() -> None:
    experiment = json.loads(
        Path("configs/experiments/alps_v1_manifest.json").read_text(encoding="utf-8")
    )
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    assert experiment["model"]["revision"] == revision
    assert experiment["model"]["tokenizer_revision"] == revision
    assert experiment["target"]["name"] == "log1p_output_tokens"
    assert experiment["inputs"]["rollout_count"] == 540
    prompt_path = Path(experiment["inputs"]["prompt_manifest"])
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == experiment["inputs"][
        "prompt_manifest_sha256"
    ]
