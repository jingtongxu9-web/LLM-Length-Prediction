from __future__ import annotations

import importlib.metadata
import platform
import time
from collections.abc import Sequence
from typing import Any

from llm_length_prediction.data.schema import GenerationTrace, MetadataValue, TracePoint


def _rolling_summary(values: Sequence[float], window: int) -> tuple[float, float]:
    recent = values[-window:]
    mean = sum(recent) / len(recent)
    slope = 0.0 if len(recent) == 1 else (recent[-1] - recent[0]) / (len(recent) - 1)
    return mean, slope


def _top_p_probabilities(probabilities: Any, top_p: float) -> Any:
    if top_p >= 1.0:
        return probabilities
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    filtered = probabilities.new_zeros(probabilities.shape)
    filtered.scatter_(-1, sorted_indices, sorted_probabilities)
    return filtered / filtered.sum(dim=-1, keepdim=True)


class HuggingFaceSignalCollector:
    """Collect prefill and decode signals from an AutoModelForCausalLM model."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        candidate_layers: Sequence[int] | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 42,
        trace_stride: int = 5,
        entropy_window: int = 20,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if trace_stride <= 0 or entropy_window <= 0:
            raise ValueError("trace_stride and entropy_window must be positive")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Hugging Face collection requires the optional 'hf' dependencies; "
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
        self.entropy_window = entropy_window

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        torch_dtype = self._resolve_dtype(dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        layer_count = self._layer_count()
        default_layers = (layer_count // 2, layer_count - 1)
        self.candidate_layers = tuple(candidate_layers or default_layers)
        invalid = [layer for layer in self.candidate_layers if not 0 <= layer < layer_count]
        if invalid:
            raise ValueError(
                f"candidate layers {invalid} are outside the available range 0..{layer_count - 1}"
            )

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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

    def _layer_count(self) -> int:
        for attribute in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(self.model.config, attribute, None)
            if value is not None:
                return int(value)
        raise ValueError("model config does not expose its hidden-layer count")

    def _eos_token_ids(self) -> tuple[int, ...]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            return ()
        if isinstance(value, int):
            return (value,)
        return tuple(int(token_id) for token_id in value)

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
            "entropy_window": self.entropy_window,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "chat_template": "tokenizer_default" if self.tokenizer.chat_template else "raw_text",
            "layer_indexing": "zero_based_transformer_block",
            "output_length_includes_eos": True,
        }
        metadata["cuda_runtime"] = self._torch.version.cuda
        if self._torch.cuda.is_available():
            device_index = self._torch.cuda.current_device()
            capability = self._torch.cuda.get_device_capability(device_index)
            metadata.update(
                {
                    "gpu_name": self._torch.cuda.get_device_name(device_index),
                    "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
                    "gpu_memory_bytes": self._torch.cuda.get_device_properties(
                        device_index
                    ).total_memory,
                }
            )
        return metadata

    def collect_trace(self, prompt: str, *, prompt_id: str, task: str) -> GenerationTrace:
        """Generate text and return a validated trace for one prompt."""

        if not prompt:
            raise ValueError("prompt must not be empty")
        torch = self._torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        started = time.perf_counter()

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("model did not return hidden states")
            prefill_hidden_states = {
                layer: hidden_states[layer + 1][0, -1].float().cpu().tolist()
                for layer in self.candidate_layers
            }

            logits = output.logits[:, -1, :].float()
            past_key_values = output.past_key_values
            del hidden_states
            eos_token_ids = self._eos_token_ids()
            generated_token_ids: list[int] = []
            sampled: list[tuple[int, float, float, int]] = []
            stop_reason = "max_new_tokens"

            for step in range(1, self.max_new_tokens + 1):
                scaled_logits = logits if self.temperature == 0 else logits / self.temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
                eos_probability = (
                    probabilities[:, list(eos_token_ids)].sum().item() if eos_token_ids else 0.0
                )
                if self.temperature == 0:
                    next_token = probabilities.argmax(dim=-1, keepdim=True)
                else:
                    sampling_probabilities = _top_p_probabilities(probabilities, self.top_p)
                    next_token = torch.multinomial(sampling_probabilities, num_samples=1)

                token_id = int(next_token.item())
                generated_token_ids.append(token_id)
                sampled.append((step, float(entropy.item()), eos_probability, token_id))
                if token_id in eos_token_ids:
                    stop_reason = "eos"
                    break

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
                    return_dict=True,
                )
                logits = output.logits[:, -1, :].float()
                past_key_values = output.past_key_values

        duration_ms = (time.perf_counter() - started) * 1000.0
        output_tokens = len(generated_token_ids)
        entropies = [item[1] for item in sampled]
        points: list[TracePoint] = []
        for index, (step, entropy, eos_probability, token_id) in enumerate(sampled):
            if step != 1 and step % self.trace_stride != 0 and step != output_tokens:
                continue
            entropy_mean, entropy_slope = _rolling_summary(
                entropies[: index + 1], self.entropy_window
            )
            points.append(
                TracePoint(
                    step=step,
                    entropy=entropy,
                    eos_probability=eos_probability,
                    remaining_length=output_tokens - step,
                    entropy_mean=entropy_mean,
                    entropy_slope=entropy_slope,
                    token_id=token_id,
                )
            )

        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        trace = GenerationTrace(
            prompt_id=prompt_id,
            task=task,
            prompt_tokens=int(input_ids.shape[-1]),
            output_tokens=output_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stop_reason=stop_reason,
            points=points,
            model_name=self.model_name,
            model_revision=self.resolved_revision,
            tokenizer_revision=self.resolved_tokenizer_revision,
            generated_text=generated_text,
            prefill_hidden_states=prefill_hidden_states,
            duration_ms=duration_ms,
            metadata=self._metadata(),
        )
        trace.validate()
        return trace
