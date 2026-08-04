from __future__ import annotations

import importlib.metadata
import platform
import time
from typing import Any

import numpy as np

from llm_length_prediction.data.plp import PLP_TRACE_SCHEMA_VERSION, PLPHiddenStateTrace
from llm_length_prediction.data.schema import MetadataValue
from llm_length_prediction.instrumentation.huggingface import _top_p_probabilities


class HuggingFacePLPCollector:
    """Collect paper-style PLP prompt and decode hidden-state inputs."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 42,
        trace_stride: int = 5,
        pooling_temperature: float = 1.0,
        entropy_chunk_tokens: int = 32,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if trace_stride <= 0 or pooling_temperature <= 0 or entropy_chunk_tokens <= 0:
            raise ValueError("stride, pooling temperature and entropy chunk must be positive")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "PLP collection requires the optional Hugging Face dependencies; "
                "install with pip install -e '.[hf]'."
            ) from error

        self._torch = torch
        self.model_name = model_name
        self.revision = revision
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.trace_stride = trace_stride
        self.pooling_temperature = pooling_temperature
        self.entropy_chunk_tokens = entropy_chunk_tokens

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        resolved_dtype = self._resolve_dtype(dtype)
        if resolved_dtype is not None:
            load_kwargs["torch_dtype"] = resolved_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, dtype: str) -> Any | None:
        if dtype == "auto":
            return None
        supported = {
            "float32": self._torch.float32,
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
        }
        try:
            return supported[dtype]
        except KeyError as error:
            raise ValueError(f"unsupported dtype: {dtype}") from error

    def _eos_token_ids(self) -> tuple[int, ...]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            return ()
        if isinstance(value, int):
            return (value,)
        return tuple(int(token_id) for token_id in value)

    def _cuda_device_index(self) -> int:
        device = self._torch.device(self.device)
        return (
            device.index
            if device.index is not None
            else self._torch.cuda.current_device()
        )

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
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def _entropy_guided_pool(self, hidden_states: Any, logits: Any) -> Any:
        """Pool every formatted-prompt token without materializing full fp32 probabilities."""

        torch = self._torch
        entropies = []
        for start in range(0, logits.shape[0], self.entropy_chunk_tokens):
            chunk = logits[start : start + self.entropy_chunk_tokens].float()
            log_probs = torch.log_softmax(chunk, dim=-1)
            entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1))
        entropy = torch.cat(entropies)
        weights = torch.softmax(entropy / self.pooling_temperature, dim=0)
        return (hidden_states.float() * weights.unsqueeze(-1)).sum(dim=0)

    def _metadata(self) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "requested_revision": self.revision,
            "resolved_model_revision": self.resolved_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "trace_stride": self.trace_stride,
            "trace_schema_version": PLP_TRACE_SCHEMA_VERSION,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "chat_template": "tokenizer_default" if self.tokenizer.chat_template else "raw_text",
            "hidden_layer": "final_transformer_layer",
            "hidden_size": int(self.model.config.hidden_size),
            "prompt_pooling": "entropy_softmax_all_formatted_prompt_tokens",
            "prompt_pooling_temperature": self.pooling_temperature,
            "dynamic_aggregation": "concat_prompt_pool_with_current_causal_hidden_state",
            "storage_dtype": "float32",
            "model_parameter_bytes": self.model_parameter_bytes,
            "output_length_includes_eos": True,
        }
        metadata["cuda_runtime"] = self._torch.version.cuda
        if self._torch.cuda.is_available() and self.device.startswith("cuda"):
            index = self._cuda_device_index()
            capability = self._torch.cuda.get_device_capability(index)
            metadata.update(
                {
                    "gpu_name": self._torch.cuda.get_device_name(index),
                    "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
                    "gpu_memory_bytes": self._torch.cuda.get_device_properties(index).total_memory,
                }
            )
        return metadata

    def collect_trace(self, prompt: str, *, prompt_id: str, task: str) -> PLPHiddenStateTrace:
        if not prompt:
            raise ValueError("prompt must not be empty")
        torch = self._torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        encoded = self._format_prompt(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        prompt_tokens = int(input_ids.shape[-1])
        if self._torch.cuda.is_available() and self.device.startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats(self._cuda_device_index())
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
            prompt_feature = self._entropy_guided_pool(
                output.hidden_states[-1][0], output.logits[0]
            )
            logits = output.logits[:, -1, :].float()
            past_key_values = output.past_key_values
            del output
            eos_token_ids = self._eos_token_ids()
            generated_token_ids: list[int] = []
            saved_steps: list[int] = []
            saved_token_ids: list[int] = []
            saved_hidden_states: list[np.ndarray] = []
            stop_reason = "max_new_tokens"

            for step in range(1, self.max_new_tokens + 1):
                scaled_logits = logits if self.temperature == 0 else logits / self.temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                if self.temperature == 0:
                    next_token = probabilities.argmax(dim=-1, keepdim=True)
                else:
                    sampling_probabilities = _top_p_probabilities(probabilities, self.top_p)
                    next_token = torch.multinomial(sampling_probabilities, num_samples=1)
                token_id = int(next_token.item())
                generated_token_ids.append(token_id)

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
                current_hidden = output.hidden_states[-1][0, -1]
                is_eos = token_id in eos_token_ids
                should_save = (
                    step == 1
                    or step % self.trace_stride == 0
                    or is_eos
                    or step == self.max_new_tokens
                )
                if should_save:
                    saved_steps.append(step)
                    saved_token_ids.append(token_id)
                    saved_hidden_states.append(current_hidden.float().cpu().numpy())
                logits = output.logits[:, -1, :].float()
                past_key_values = output.past_key_values
                if is_eos:
                    stop_reason = "eos"
                    break

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize(self._cuda_device_index())
        output_tokens = len(generated_token_ids)
        steps = np.asarray(saved_steps, dtype=np.int32)
        metadata = self._metadata()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            device_index = self._cuda_device_index()
            metadata.update(
                {
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        device_index
                    ),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        device_index
                    ),
                }
            )
        trace = PLPHiddenStateTrace(
            prompt_id=prompt_id,
            task=task,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stop_reason=stop_reason,
            prompt_feature=prompt_feature.float().cpu().numpy(),
            decode_hidden_states=np.stack(saved_hidden_states),
            steps=steps,
            remaining_lengths=output_tokens - steps,
            token_ids=np.asarray(saved_token_ids, dtype=np.int32),
            generated_token_ids=np.asarray(generated_token_ids, dtype=np.int32),
            model_name=self.model_name,
            model_revision=self.resolved_revision,
            tokenizer_revision=self.resolved_tokenizer_revision,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )
        trace.validate()
        return trace
