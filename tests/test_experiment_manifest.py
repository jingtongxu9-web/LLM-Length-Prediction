from pathlib import Path

from llm_length_prediction.comparison import (
    load_base_experiment_for_method,
    load_method_config,
    validate_project_plp_contract,
)
from llm_length_prediction.experiment import (
    file_sha256,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
)
from llm_length_prediction.plp_experiment import (
    load_plp_base_experiment,
    load_plp_config,
    validate_plp_config,
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


def test_frozen_project_plp_manifest() -> None:
    config = load_method_config(Path("configs/experiments/plp_v1_manifest.json"))
    _, experiment, _ = load_base_experiment_for_method(config)
    validate_project_plp_contract(config, experiment)
    assert config["method_id"] == "dynamic-signal-mlp-v1"
    assert config["method_role"] == "project_adaptation_dynamic_baseline"
    assert config["scope"]["uses_alps_prior"] is False
    assert config["scope"]["paper_exact_plp"] is False
    assert config["provenance"]["exact_replication"] is False
    assert config["provenance"]["paper_replication_version"] == (
        "configs/experiments/plp_v2_manifest.json"
    )
    assert config["training"]["hyperparameter_selection"] == "none"


def test_hidden_state_plp_v2_manifest() -> None:
    config = load_plp_config(Path("configs/experiments/plp_v2_manifest.json"))
    experiment, _ = load_plp_base_experiment(config)
    validate_plp_config(config, experiment)
    assert config["scope"]["plp_only"] is True
    assert config["scope"]["uses_alps_prior"] is False
    assert config["scope"]["uses_prompt_hidden_state"] is True
    assert config["scope"]["uses_decode_hidden_state"] is True
    assert config["representation"]["hidden_layer"] == "final_transformer_layer"
    assert config["prediction_head"]["num_bins"] == 20
    assert config["prediction_head"]["lambda_ce"] == 0.95
