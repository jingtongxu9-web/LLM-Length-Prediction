"""Single-pass Qwen collector for Bayesian Sequential and all frozen baselines."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.bayesian_trace import (
    BAYESIAN_TRACE_SCHEMA_NAME,
    BAYESIAN_TRACE_SCHEMA_VERSION,
    BayesianTraceV1,
)
from llm_length_prediction.instrumentation.huggingface import _top_p_probabilities


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


class HuggingFaceBayesianCollector:
    """Collect raw non-overlapping evidence inputs from one causal decoding pass."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 42,
        trace_stride: int = 5,
        pooling_temperature: float = 1.0,
        prior_layer: int = 14,
        entropy_chunk_tokens: int = 32,
        model: Any | None = None,
        tokenizer: Any | None = None,
        torch_module: Any | None = None,
        reported_model_name: str | None = None,
    ) -> None:
        if max_new_tokens <= 0 or trace_stride <= 1:
            raise ValueError("max_new_tokens must be positive and stride must exceed one")
        if not 0.0 < temperature or not 0.0 < top_p <= 1.0:
            raise ValueError("Bayesian collection requires temperature > 0 and top_p in (0, 1]")
        if pooling_temperature <= 0 or prior_layer < 0 or entropy_chunk_tokens <= 0:
            raise ValueError("invalid representation settings")
        injected = (model is not None, tokenizer is not None, torch_module is not None)
        if any(injected) and not all(injected):
            raise ValueError("model, tokenizer, and torch_module must be injected together")
        if torch_module is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Bayesian trace collection requires torch and transformers"
                ) from error
            torch_module = torch
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            dtypes = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if dtype not in dtypes:
                raise ValueError("unsupported dtype")
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtypes[dtype],
            )
        else:
            resolved_device = "cpu" if device == "auto" else device
        self.torch = torch_module
        self.model_source = model_name
        self.model_name = reported_model_name or model_name
        self.revision = revision
        self.device = resolved_device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.trace_stride = trace_stride
        self.pooling_temperature = pooling_temperature
        self.prior_layer = prior_layer
        self.entropy_chunk_tokens = entropy_chunk_tokens
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval().requires_grad_(False)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        layer_count = self._layer_count()
        if self.prior_layer >= layer_count:
            raise ValueError("prior_layer falls outside the model's transformer blocks")
        self.model_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )

    def _layer_count(self) -> int:
        for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(self.model.config, attribute, None)
            if value is not None:
                return int(value)
        raise ValueError("model config does not expose a transformer-layer count")

    def _format_prompt(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=False,
            )
        else:
            formatted = prompt
            encoded = self.tokenizer(prompt, return_tensors="pt")
        return formatted, {
            name: value.to(self.device)
            for name, value in encoded.items()
        }

    def _eos_ids(self) -> tuple[int, ...]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            return ()
        return (int(value),) if isinstance(value, int) else tuple(int(item) for item in value)

    def _pool_prompt(self, hidden_states: Any, logits: Any) -> Any:
        entropies = []
        for start in range(0, logits.shape[0], self.entropy_chunk_tokens):
            chunk = logits[start : start + self.entropy_chunk_tokens].float()
            log_probabilities = self.torch.log_softmax(chunk, dim=-1)
            entropies.append(
                -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
            )
        entropy = self.torch.cat(entropies)
        weights = self.torch.softmax(entropy / self.pooling_temperature, dim=0)
        return (hidden_states.float() * weights.unsqueeze(-1)).sum(dim=0)

    def _cuda_device_index(self) -> int:
        device = self.torch.device(self.device)
        return (
            device.index
            if device.index is not None
            else self.torch.cuda.current_device()
        )

    def _metadata(self, *, prompt: str, formatted_prompt: str, input_ids: Any) -> dict[str, Any]:
        token_bytes = input_ids.detach().to("cpu").numpy().astype(np.int64).tobytes()
        metadata: dict[str, Any] = {
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
            "trace_schema": BAYESIAN_TRACE_SCHEMA_NAME,
            "trace_schema_version": BAYESIAN_TRACE_SCHEMA_VERSION,
            "trace_stride": self.trace_stride,
            "prior_feature_layer": self.prior_layer,
            "prior_layer_indexing": "zero_based_transformer_block",
            "decode_hidden_layer": "final_transformer_layer",
            "hidden_size": int(self.model.config.hidden_size),
            "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
            "prompt_pooling_temperature": self.pooling_temperature,
            "probability_source": "temperature_scaled_full_softmax_before_top_p",
            "evidence_unit": "non_overlapping_new_token_block_since_previous_update",
            "storage_dtype": "float32",
            "chat_template": (
                "tokenizer_default" if self.tokenizer.chat_template else "raw_text"
            ),
            "output_length_includes_eos": True,
            "model_parameter_bytes": self.model_parameter_bytes,
            "model_source_kind": (
                "local_snapshot" if Path(self.model_source).is_dir() else "hub_id"
            ),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "formatted_prompt_sha256": hashlib.sha256(
                formatted_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        }
        metadata["cuda_runtime"] = self.torch.version.cuda
        if self.device.startswith("cuda"):
            index = self._cuda_device_index()
            metadata.update(
                {
                    "gpu_name": self.torch.cuda.get_device_name(index),
                    "gpu_memory_bytes": self.torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
            )
        return metadata

    def collect_trace(
        self,
        prompt: str,
        *,
        prompt_id: str,
        prompt_family_id: str,
        task: str,
        intended_length: str,
        split: str,
    ) -> BayesianTraceV1:
        if not prompt:
            raise ValueError("prompt must not be empty")
        torch = self.torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        formatted_prompt, encoded = self._format_prompt(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self._cuda_device_index())
        started = time.perf_counter()

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None:
                raise RuntimeError("model did not return prefill hidden states")
            if self.prior_layer + 1 >= len(output.hidden_states):
                raise RuntimeError("prior layer falls outside returned hidden states")
            prior_feature = output.hidden_states[self.prior_layer + 1][0, -1].float()
            final_prompt_hidden = output.hidden_states[-1][0]
            prompt_feature = self._pool_prompt(final_prompt_hidden, output.logits[0])
            initial_decode_hidden = final_prompt_hidden[-1].float()
            logits = output.logits[:, -1, :].float()
            past_key_values = output.past_key_values
            del output

            eos_ids = self._eos_ids()
            generated: list[int] = []
            entropies: list[float] = []
            eos_probabilities: list[float] = []
            saved_steps: list[int] = []
            saved_hidden: list[np.ndarray] = []
            stop_reason = "max_new_tokens"
            for step in range(1, self.max_new_tokens + 1):
                scaled_logits = logits / self.temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                entropy = float(
                    -(
                        probabilities
                        * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1).item()
                )
                eos_probability = (
                    float(probabilities[:, list(eos_ids)].sum().item()) if eos_ids else 0.0
                )
                sampling_probabilities = _top_p_probabilities(probabilities, self.top_p)
                next_token = torch.multinomial(sampling_probabilities, num_samples=1)
                token_id = int(next_token.item())
                generated.append(token_id)
                entropies.append(entropy)
                eos_probabilities.append(eos_probability)
                is_eos = token_id in eos_ids
                should_save = step == 1 or step % self.trace_stride == 0 or is_eos
                needs_forward = step < self.max_new_tokens or should_save
                if needs_forward:
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                device=self.device,
                                dtype=attention_mask.dtype,
                            ),
                        ),
                        dim=-1,
                    )
                    output = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=should_save,
                        return_dict=True,
                    )
                    if should_save:
                        if output.hidden_states is None:
                            raise RuntimeError("model omitted a scheduled decode hidden state")
                        saved_steps.append(step)
                        saved_hidden.append(
                            output.hidden_states[-1][0, -1].float().cpu().numpy()
                        )
                    logits = output.logits[:, -1, :].float()
                    past_key_values = output.past_key_values
                    del output
                if is_eos:
                    stop_reason = "eos"
                    break

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self._cuda_device_index())
        metadata = self._metadata(
            prompt=prompt,
            formatted_prompt=formatted_prompt,
            input_ids=input_ids,
        )
        if self.device.startswith("cuda"):
            index = self._cuda_device_index()
            metadata.update(
                {
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                }
            )
        trace = BayesianTraceV1(
            prompt_id=prompt_id,
            prompt_family_id=prompt_family_id,
            task=task,
            intended_length=intended_length,
            split=split,
            prompt_tokens=int(input_ids.shape[-1]),
            observed_tokens=len(generated),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop_reason=stop_reason,
            eos_token_ids=eos_ids,
            prior_feature=prior_feature.cpu().numpy(),
            prompt_feature=prompt_feature.cpu().numpy(),
            initial_decode_hidden_state=initial_decode_hidden.cpu().numpy(),
            decode_hidden_states=np.stack(saved_hidden),
            saved_steps=np.asarray(saved_steps, dtype=np.int32),
            generated_token_ids=np.asarray(generated, dtype=np.int32),
            token_entropies=np.asarray(entropies, dtype=np.float32),
            token_eos_probabilities=np.asarray(eos_probabilities, dtype=np.float32),
            model_name=self.model_name,
            model_revision=str(self.resolved_revision),
            tokenizer_revision=str(self.resolved_tokenizer_revision),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )
        trace.validate(stride=self.trace_stride)
        return trace
