from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_length_prediction.bayesian_pilot import (
    BayesianPilotJob,
    bayesian_pilot_jobs,
    load_bayesian_pilot,
    validate_bayesian_pilot_trace,
)
from llm_length_prediction.data.bayesian_trace import (
    BayesianTraceV1,
    bayesian_trace_path,
    read_bayesian_trace,
    sequential_raw_trace_from_collected,
    write_bayesian_trace,
)
from llm_length_prediction.data.sequential import build_bayesian_sequence
from llm_length_prediction.instrumentation.bayesian import HuggingFaceBayesianCollector

PILOT_PATH = Path("configs/experiments/bayesian_sequential_pilot_v1.json")


class FakeTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0
    eos_token_id = 3
    pad_token = "<pad>"
    init_kwargs = {"_commit_hash": "fake-revision"}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        return f"user:{messages[0]['content']}\nassistant:"

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        assert text and return_tensors == "pt"
        assert add_special_tokens is False
        return {
            "input_ids": torch.tensor([[4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


class FakeCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            _commit_hash="fake-revision",
            num_hidden_layers=2,
            hidden_size=4,
        )
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, past_key_values
        assert use_cache and return_dict
        sequence_length = input_ids.shape[1]
        logits = torch.full((1, sequence_length, 6), -100.0)
        next_tokens = (1, 2, 3, 0)
        logits[:, -1, next_tokens[min(self.calls, 3)]] = 100.0
        hidden_states = None
        if output_hidden_states:
            base = torch.arange(sequence_length * 4, dtype=torch.float32).reshape(
                1,
                sequence_length,
                4,
            )
            hidden_states = tuple(base + self.calls + layer for layer in range(3))
        result = SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=(self.calls,),
        )
        self.calls += 1
        return result


def test_fake_collector_keeps_raw_token_signals_and_scheduled_hidden_states(
    tmp_path: Path,
) -> None:
    collector = HuggingFaceBayesianCollector(
        "fake/source",
        revision="fake-revision",
        device="cpu",
        max_new_tokens=6,
        temperature=0.7,
        top_p=0.95,
        trace_stride=5,
        prior_layer=0,
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        torch_module=torch,
        reported_model_name="fake/model",
    )
    trace = collector.collect_trace(
        "hello",
        prompt_id="p",
        prompt_family_id="f",
        task="qa",
        intended_length="short",
        split="train",
    )
    assert trace.stop_reason == "eos"
    assert trace.observed_tokens == 3
    assert trace.generated_token_ids.tolist() == [1, 2, 3]
    assert trace.saved_steps.tolist() == [1, 3]
    assert trace.decode_hidden_states.shape == (2, 4)
    assert trace.token_entropies.shape == (3,)
    assert trace.token_eos_probabilities.shape == (3,)
    assert trace.token_eos_probabilities[-1] > 0.99
    assert trace.metadata["probability_source"].endswith("before_top_p")

    path = write_bayesian_trace(tmp_path / "trace.npz", trace)
    restored = read_bayesian_trace(path)
    assert np.array_equal(restored.generated_token_ids, trace.generated_token_ids)
    raw = sequential_raw_trace_from_collected(
        restored,
        prior_mu=np.log1p(3),
        prior_log_variance=0.2,
        prior_mean_total_tokens=3.0,
    )
    sequence = build_bayesian_sequence(raw)
    assert [step.true_remaining for step in sequence.steps] == [2, 0]


def test_trace_paths_separate_sampling_temperatures() -> None:
    first = bayesian_trace_path(
        "root",
        split="train",
        prompt_id="p",
        temperature=0.7,
        seed=42,
    )
    second = bayesian_trace_path(
        "root",
        split="train",
        prompt_id="p",
        temperature=1.0,
        seed=42,
    )
    assert first != second
    assert "temperature_0p700" in str(first)


def test_pilot_uses_only_opened_train_families_and_full_task_length_grid() -> None:
    pilot, contract, records = load_bayesian_pilot(PILOT_PATH)
    jobs = bayesian_pilot_jobs(pilot, records)
    assert contract["method_id"] == "bayesian-sequential-v1"
    assert len(records) == 9
    assert len(jobs) == 9
    assert pilot["acceptance_report_schema"]["sha256"] == (
        "3d67159e735e7ef29a6e8a87378d9d5cef4738a43961408f756fde68b887dfce"
    )
    assert {record["split"] for record in records} == {"train"}
    assert {record["provenance"] for record in records} == {"opened_v1_design_data"}
    assert len({(record["task_type"], record["intended_length"]) for record in records}) == 9


def _pilot_trace(pilot: dict[str, object], job: BayesianPilotJob) -> BayesianTraceV1:
    model = pilot["model"]
    generation = pilot["generation"]
    trace_contract = pilot["trace"]
    representation = pilot["prompt_representation"]
    source = pilot["source_prompts"]
    hidden_size = int(model["hidden_size"])
    record = job.record
    metadata = {
        "pilot_id": pilot["pilot_id"],
        "scientific_contract_sha256": pilot["scientific_contract"]["sha256"],
        "source_prompt_manifest_sha256": source["sha256"],
        "trace_schema": trace_contract["schema_name"],
        "trace_schema_version": trace_contract["schema_version"],
        "trace_stride": trace_contract["stride"],
        "prior_feature_layer": model["prior_layer_zero_based"],
        "prior_layer_indexing": "zero_based_transformer_block",
        "decode_hidden_layer": model["decode_layer"],
        "hidden_size": hidden_size,
        "prompt_pooling": representation["pooling"],
        "prompt_pooling_temperature": representation["pooling_temperature"],
        "probability_source": trace_contract["probability_source"],
        "evidence_unit": trace_contract["evidence_unit"],
        "storage_dtype": trace_contract["storage_dtype"],
        "chat_template": generation["chat_template"],
        "output_length_includes_eos": True,
        "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
    }
    return BayesianTraceV1(
        prompt_id=record["prompt_id"],
        prompt_family_id=record["prompt_family_id"],
        task=record["task_type"],
        intended_length=record["intended_length"],
        split="train",
        prompt_tokens=8,
        observed_tokens=2,
        max_new_tokens=generation["max_new_tokens"],
        temperature=job.temperature,
        top_p=generation["top_p"],
        seed=job.seed,
        stop_reason="eos",
        eos_token_ids=(3,),
        prior_feature=np.zeros(hidden_size, dtype=np.float32),
        prompt_feature=np.zeros(hidden_size, dtype=np.float32),
        initial_decode_hidden_state=np.zeros(hidden_size, dtype=np.float32),
        decode_hidden_states=np.zeros((2, hidden_size), dtype=np.float32),
        saved_steps=np.asarray([1, 2], dtype=np.int32),
        generated_token_ids=np.asarray([2, 3], dtype=np.int32),
        token_entropies=np.asarray([1.0, 0.1], dtype=np.float32),
        token_eos_probabilities=np.asarray([0.1, 0.9], dtype=np.float32),
        model_name=model["id"],
        model_revision=model["revision"],
        tokenizer_revision=model["tokenizer_revision"],
        duration_ms=10.0,
        metadata=metadata,
    )


def test_pilot_trace_validation_pins_cuda_provenance_and_contract() -> None:
    pilot, _, records = load_bayesian_pilot(PILOT_PATH)
    job = bayesian_pilot_jobs(pilot, records)[0]
    trace = _pilot_trace(pilot, job)
    validate_bayesian_pilot_trace(trace, job=job, pilot=pilot)
    trace.metadata["probability_source"] = "after_top_p"
    with pytest.raises(ValueError, match="probability_source"):
        validate_bayesian_pilot_trace(trace, job=job, pilot=pilot)
