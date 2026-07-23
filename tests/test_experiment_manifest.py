from pathlib import Path

from llm_length_prediction.experiment import (
    file_sha256,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
)


def test_frozen_experiment_manifest() -> None:
    experiment = load_experiment(Path("configs/experiments/alps_v1_manifest.json"))
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    assert experiment["model"]["revision"] == revision
    assert experiment["model"]["tokenizer_revision"] == revision
    assert experiment["target"]["name"] == "log1p_output_tokens"
    assert experiment["ridge"] == {"standardize": True, "alpha": 1.0}
    assert experiment["inputs"]["rollout_count"] == 540
    prompt_path = Path(experiment["inputs"]["prompt_manifest"])
    assert file_sha256(prompt_path) == experiment["inputs"]["prompt_manifest_sha256"]
    records = load_frozen_prompts(experiment)
    assert len(list(rollout_jobs(records, split="train"))) == 432
    assert len(list(rollout_jobs(records, split="test"))) == 108
