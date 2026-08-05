"""Single-pass Qwen collector for every Hybrid v3 comparator."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from typing import Any

import numpy as np

from llm_length_prediction.data.hybrid import HybridV3Trace
from llm_length_prediction.data.schema import MetadataValue
from llm_length_prediction.instrumentation.huggingface import (
    _rolling_summary,
    _top_p_probabilities,
)


class HuggingFaceHybridV3Collector:
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
        entropy_window: int = 20,
        pooling_temperature: float = 1.0,
        prior_layer: int = 14,
        entropy_chunk_tokens: int = 32,
    ) -> None:
        if max_new_tokens <= 0 or trace_stride <= 0 or entropy_window <= 1:
            raise ValueError("invalid generation or trace schedule")
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid sampling settings")
        if pooling_temperature <= 0 or prior_layer < 0 or entropy_chunk_tokens <= 0:
            raise ValueError("invalid representation settings")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Hybrid v3 collection requires torch and transformers") from error
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtypes:
            raise ValueError("unsupported dtype")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.trace_stride = trace_stride
        self.entropy_window = entropy_window
        self.pooling_temperature = pooling_temperature
        self.prior_layer = prior_layer
        self.entropy_chunk_tokens = entropy_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=dtypes[dtype]
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> dict[str, Any]:
        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        return {name: value.to(self.device) for name, value in encoded.items()}

    def _eos_ids(self) -> tuple[int, ...]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            return ()
        return (value,) if isinstance(value, int) else tuple(int(item) for item in value)

    def _pool_prompt(self, hidden_states: Any, logits: Any) -> Any:
        entropies = []
        for start in range(0, logits.shape[0], self.entropy_chunk_tokens):
            chunk = logits[start : start + self.entropy_chunk_tokens].float()
            log_probs = self.torch.log_softmax(chunk, dim=-1)
            entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1))
        entropy = self.torch.cat(entropies)
        weights = self.torch.softmax(entropy / self.pooling_temperature, dim=0)
        return (hidden_states.float() * weights.unsqueeze(-1)).sum(dim=0)

    def _metadata(self) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "python_version": platform.python_version(),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "trace_schema": "hybrid-v3-unified-trace",
            "trace_schema_version": 1,
            "trace_stride": self.trace_stride,
            "entropy_window": self.entropy_window,
            "prior_feature_layer": self.prior_layer,
            "prior_layer_indexing": "zero_based_transformer_block",
            "hidden_layer": "final_transformer_layer",
            "hidden_size": int(self.model.config.hidden_size),
            "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
            "prompt_pooling_temperature": self.pooling_temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "chat_template": "tokenizer_default" if self.tokenizer.chat_template else "raw_text",
            "output_length_includes_eos": True,
        }
        metadata["cuda_runtime"] = self.torch.version.cuda
        if self.device.startswith("cuda"):
            index = self.torch.cuda.current_device()
            metadata.update(
                {
                    "gpu_name": self.torch.cuda.get_device_name(index),
                    "gpu_memory_bytes": self.torch.cuda.get_device_properties(index).total_memory,
                }
            )
        return metadata

    def collect_trace(self, prompt: str, *, prompt_id: str, task: str) -> HybridV3Trace:
        if not prompt:
            raise ValueError("prompt must not be empty")
        torch = self.torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        encoded = self._format_prompt(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
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
                raise RuntimeError("model did not return prompt hidden states")
            if self.prior_layer + 1 >= len(output.hidden_states):
                raise RuntimeError("prior layer falls outside the model")
            prior_feature = output.hidden_states[self.prior_layer + 1][0, -1].float()
            prompt_feature = self._pool_prompt(output.hidden_states[-1][0], output.logits[0])
            logits = output.logits[:, -1, :].float()
            past_key_values = output.past_key_values
            del output
            eos_ids = self._eos_ids()
            generated: list[int] = []
            saved_steps: list[int] = []
            saved_ids: list[int] = []
            saved_hidden: list[np.ndarray] = []
            entropies: list[float] = []
            saved_entropy: list[float] = []
            saved_mean: list[float] = []
            saved_slope: list[float] = []
            saved_eos: list[float] = []
            stop_reason = "max_new_tokens"
            for step in range(1, self.max_new_tokens + 1):
                scaled = logits if self.temperature == 0 else logits / self.temperature
                probabilities = torch.softmax(scaled, dim=-1)
                entropy = float(
                    -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).item()
                )
                eos_probability = (
                    float(probabilities[:, list(eos_ids)].sum().item()) if eos_ids else 0.0
                )
                entropies.append(entropy)
                if self.temperature == 0:
                    next_token = probabilities.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(
                        _top_p_probabilities(probabilities, self.top_p), 1
                    )
                token_id = int(next_token.item())
                generated.append(token_id)
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ),
                    dim=-1,
                )
                output = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if output.hidden_states is None:
                    raise RuntimeError("model did not return decode hidden states")
                is_eos = token_id in eos_ids
                should_save = (
                    step == 1
                    or step % self.trace_stride == 0
                    or is_eos
                    or step == self.max_new_tokens
                )
                if should_save:
                    mean, slope = _rolling_summary(entropies, self.entropy_window)
                    saved_steps.append(step)
                    saved_ids.append(token_id)
                    saved_hidden.append(output.hidden_states[-1][0, -1].float().cpu().numpy())
                    saved_entropy.append(entropy)
                    saved_mean.append(mean)
                    saved_slope.append(slope)
                    saved_eos.append(eos_probability)
                logits = output.logits[:, -1, :].float()
                past_key_values = output.past_key_values
                if is_eos:
                    stop_reason = "eos"
                    break
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        output_tokens = len(generated)
        steps = np.asarray(saved_steps, dtype=np.int32)
        metadata = self._metadata()
        if self.device.startswith("cuda"):
            metadata.update(
                {
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
        trace = HybridV3Trace(
            prompt_id=prompt_id,
            task=task,
            prompt_tokens=int(input_ids.shape[-1]),
            output_tokens=output_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stop_reason=stop_reason,
            prior_feature=prior_feature.cpu().numpy(),
            prompt_feature=prompt_feature.cpu().numpy(),
            decode_hidden_states=np.stack(saved_hidden),
            steps=steps,
            remaining_lengths=output_tokens - steps,
            token_ids=np.asarray(saved_ids, dtype=np.int32),
            generated_token_ids=np.asarray(generated, dtype=np.int32),
            entropies=np.asarray(saved_entropy, dtype=np.float32),
            entropy_means=np.asarray(saved_mean, dtype=np.float32),
            entropy_slopes=np.asarray(saved_slope, dtype=np.float32),
            eos_probabilities=np.asarray(saved_eos, dtype=np.float32),
            model_name=self.model_name,
            model_revision=self.resolved_revision,
            tokenizer_revision=self.resolved_tokenizer_revision,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )
        trace.validate()
        return trace
