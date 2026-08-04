from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_length_prediction.data.plp import PLPHiddenStateTrace


@dataclass(frozen=True)
class HiddenStatePLPSample:
    """One saved PLP update point."""

    prompt_id: str
    prompt_family_id: str
    task: str
    intended_length: str
    seed: int
    step: int
    output_tokens: int
    remaining_tokens: int
    features: np.ndarray
    sequence_weight: float


def build_hidden_state_plp_samples(
    trace: PLPHiddenStateTrace,
    *,
    prompt_family_id: str,
    intended_length: str,
    exclude_censored: bool = True,
) -> list[HiddenStatePLPSample]:
    """Concatenate the pooled prompt state with each current causal decode state."""

    trace.validate()
    if exclude_censored and trace.stop_reason == "max_new_tokens":
        return []
    point_count = len(trace.steps)
    sequence_weight = 1.0 / point_count
    return [
        HiddenStatePLPSample(
            prompt_id=trace.prompt_id,
            prompt_family_id=prompt_family_id,
            task=trace.task,
            intended_length=intended_length,
            seed=trace.seed,
            step=int(step),
            output_tokens=trace.output_tokens,
            remaining_tokens=int(remaining),
            features=np.concatenate((trace.prompt_feature, decode_state)).astype(
                np.float32, copy=False
            ),
            sequence_weight=sequence_weight,
        )
        for step, remaining, decode_state in zip(
            trace.steps,
            trace.remaining_lengths,
            trace.decode_hidden_states,
            strict=True,
        )
    ]


def target_range_from_training(
    remaining_lengths: Sequence[int], lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> tuple[float, float]:
    values = np.asarray(remaining_lengths, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("remaining_lengths must be a non-empty non-negative vector")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("target percentiles must satisfy 0 <= lower < upper <= 100")
    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    lower = max(0.0, float(lower))
    upper = float(upper)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def length_bin_centers(
    target_range: tuple[float, float], num_bins: int
) -> np.ndarray:
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than one")
    lower, upper = target_range
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or upper <= lower:
        raise ValueError("target_range must be finite, non-negative and increasing")
    width = (upper - lower) / num_bins
    return lower + (np.arange(num_bins, dtype=np.float32) + 0.5) * width


def soft_length_labels(
    lengths: Sequence[float], target_range: tuple[float, float], num_bins: int
) -> np.ndarray:
    """Paper soft labels: p_j is proportional to exp(-abs(j-i))."""

    values = np.asarray(lengths, dtype=np.float32)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("lengths must be a finite one-dimensional vector")
    length_bin_centers(target_range, num_bins)
    lower, upper = target_range
    width = (upper - lower) / num_bins
    clipped = np.clip(values, lower, upper)
    target_bins = np.floor((clipped - lower) / width).astype(np.int64)
    target_bins = np.clip(target_bins, 0, num_bins - 1)
    indices = np.arange(num_bins, dtype=np.int64)[None, :]
    logits = -np.abs(indices - target_bins[:, None]).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def plp_head_parameter_count(input_dim: int, num_bins: int) -> int:
    """Return the exact trainable parameter count for the frozen PLP head skeleton."""

    if input_dim <= 0 or num_bins <= 1:
        raise ValueError("input_dim must be positive and num_bins must exceed one")
    hidden_dim = input_dim // 2
    first_linear = input_dim * hidden_dim + hidden_dim
    layer_norm = 2 * hidden_dim
    output_linear = hidden_dim * num_bins + num_bins
    return first_linear + layer_norm + output_linear


def build_plp_head(
    input_dim: int,
    *,
    num_bins: int,
    target_range: tuple[float, float],
    dropout: float,
) -> Any:
    """Build the paper head lazily so core installs do not require PyTorch."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("hidden-state PLP requires PyTorch") from error

    centers = torch.from_numpy(length_bin_centers(target_range, num_bins))

    class PLPHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_dim = input_dim // 2
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_bins),
            )
            self.register_buffer("bin_centers", centers)

        def forward(self, features: Any) -> tuple[Any, Any]:
            logits = self.network(features)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = (probabilities * self.bin_centers).sum(dim=-1)
            return logits, prediction

    return PLPHead()


def fit_hidden_state_plp(
    samples: Sequence[HiddenStatePLPSample],
    *,
    num_bins: int = 20,
    target_percentiles: tuple[float, float] = (1.0, 99.0),
    lambda_ce: float = 0.95,
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    """Train the paper-style soft-label head with per-sequence balanced loss."""

    if not samples:
        raise ValueError("at least one PLP sample is required")
    if not 0.0 <= lambda_ce <= 1.0:
        raise ValueError("lambda_ce must be in [0, 1]")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid PLP training hyperparameters")
    input_dim = int(samples[0].features.size)
    if any(sample.features.shape != (input_dim,) for sample in samples):
        raise ValueError("all PLP features must share one input dimension")

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            "PLP training requires PyTorch; use the AutoDL PyTorch image or install .[model]"
        ) from error

    features = np.stack([sample.features for sample in samples]).astype(
        np.float32, copy=False
    )
    targets = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float32)
    weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float32)
    # Preserve equal total weight per rollout while keeping the average weight at one.
    weights *= len(weights) / float(weights.sum())
    target_range = target_range_from_training(targets, *target_percentiles)
    soft_targets = soft_length_labels(targets, target_range, num_bins)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PLP config requests CUDA, but CUDA is unavailable")
    if resolved_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(resolved_device)

    head = build_plp_head(
        input_dim,
        num_bins=num_bins,
        target_range=target_range,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False
    )
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(soft_targets),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    epoch_losses: list[float] = []
    started = time.perf_counter()
    for _ in range(epochs):
        head.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for batch_features, batch_targets, batch_soft, batch_weights in loader:
            batch_features = batch_features.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            batch_soft = batch_soft.to(resolved_device)
            batch_weights = batch_weights.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits, predictions = head(batch_features)
            cross_entropy = -(
                batch_soft * functional.log_softmax(logits, dim=-1)
            ).sum(dim=-1)
            squared_error = (predictions - batch_targets).square()
            point_loss = lambda_ce * cross_entropy + (1.0 - lambda_ce) * squared_error
            loss = (batch_weights * point_loss).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("PLP training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float((batch_weights * point_loss.detach()).sum().item())
            weight_sum += float(batch_weights.sum().item())
        epoch_losses.append(weighted_loss_sum / weight_sum)

    if resolved_device.startswith("cuda"):
        torch.cuda.synchronize(resolved_device)
    training_duration_seconds = time.perf_counter() - started

    trainable_parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    expected_parameter_count = plp_head_parameter_count(input_dim, num_bins)
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"PLP head has {parameter_count} trainable parameters; "
            f"expected {expected_parameter_count}"
        )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in trainable_parameters
    )
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in trainable_parameters
        if parameter.grad is not None
    )
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )

    report = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "device": resolved_device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lambda_ce": lambda_ce,
        "num_bins": num_bins,
        "target_percentiles": list(target_percentiles),
        "target_range": list(target_range),
        "input_dim": input_dim,
        "trainable_parameter_count": parameter_count,
        "trainable_parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "feature_matrix_bytes": features.nbytes,
        "training_sample_count": len(samples),
        "training_duration_seconds": training_duration_seconds,
        "final_sequence_balanced_joint_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
    }
    if resolved_device.startswith("cuda"):
        requested_device = torch.device(resolved_device)
        device_index = (
            requested_device.index
            if requested_device.index is not None
            else torch.cuda.current_device()
        )
        report.update(
            {
                "gpu_name": torch.cuda.get_device_name(device_index),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                    device_index
                ).total_memory,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    resolved_device
                ),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                    resolved_device
                ),
            }
        )
    return head, report


def predict_hidden_state_plp(
    head: Any,
    samples: Sequence[HiddenStatePLPSample],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("at least one PLP sample is required")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("hidden-state PLP inference requires PyTorch") from error
    head.eval()
    predictions = []
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            matrix = np.stack(
                [sample.features for sample in samples[start : start + batch_size]]
            ).astype(np.float32, copy=False)
            logits, predicted = head(torch.from_numpy(matrix).to(device))
            predictions.append(predicted.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)
