from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llm_length_prediction.data.schema import GenerationTrace

PLP_FEATURE_NAMES = (
    "step",
    "entropy",
    "entropy_mean",
    "entropy_slope",
    "eos_probability",
)


@dataclass(frozen=True)
class DynamicFeatures:
    prior_length: float
    step: int
    entropy: float
    entropy_mean: float
    entropy_slope: float
    eos_probability: float


@dataclass(frozen=True)
class ProgressiveSample:
    """One non-terminal PLP training/evaluation point from a generation trace."""

    prompt_id: str
    prompt_family_id: str
    seed: int
    step: int
    output_tokens: int
    remaining_tokens: int
    features: tuple[float, ...]
    sequence_weight: float


@dataclass(frozen=True)
class StandardizedMLPRemainingLength:
    """Small MLP predicting log1p(remaining tokens) from decode-time signals."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    layer_weights: tuple[tuple[tuple[float, ...], ...], ...]
    layer_biases: tuple[tuple[float, ...], ...]
    residual_variance: float
    hidden_sizes: tuple[int, ...]
    dropout: float
    target: str = "log1p_remaining_tokens"

    def __post_init__(self) -> None:
        if self.feature_names != PLP_FEATURE_NAMES:
            raise ValueError(f"feature_names must be exactly {PLP_FEATURE_NAMES!r}")
        feature_count = len(self.feature_names)
        if len(self.feature_mean) != feature_count or len(self.feature_scale) != feature_count:
            raise ValueError("feature scaler dimensions do not match feature_names")
        if not all(math.isfinite(value) for value in self.feature_mean):
            raise ValueError("feature_mean must contain only finite values")
        if not all(math.isfinite(value) and value > 0 for value in self.feature_scale):
            raise ValueError("feature_scale must contain finite positive values")
        if not math.isfinite(self.residual_variance) or self.residual_variance < 0:
            raise ValueError("residual_variance must be finite and non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.target != "log1p_remaining_tokens":
            raise ValueError("target must be log1p_remaining_tokens")
        expected_sizes = (feature_count, *self.hidden_sizes, 1)
        if len(self.layer_weights) != len(expected_sizes) - 1:
            raise ValueError("MLP layer count does not match hidden_sizes")
        if len(self.layer_biases) != len(self.layer_weights):
            raise ValueError("each MLP weight matrix requires one bias vector")
        for index, (weights, biases, input_size, output_size) in enumerate(
            zip(
                self.layer_weights,
                self.layer_biases,
                expected_sizes[:-1],
                expected_sizes[1:],
                strict=True,
            )
        ):
            if len(weights) != output_size or len(biases) != output_size:
                raise ValueError(f"MLP layer {index} has the wrong output dimension")
            if any(len(row) != input_size for row in weights):
                raise ValueError(f"MLP layer {index} has the wrong input dimension")
            values = [value for row in weights for value in row]
            if not all(math.isfinite(value) for value in (*values, *biases)):
                raise ValueError(f"MLP layer {index} contains non-finite parameters")

    def _feature_matrix(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have shape (n, {len(self.feature_names)})"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must contain only finite values")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        return (matrix - mean) / scale

    def predict_mu_many(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        values = self._feature_matrix(features)
        for layer_index, (weights, biases) in enumerate(
            zip(self.layer_weights, self.layer_biases, strict=True)
        ):
            weight_matrix = np.asarray(weights, dtype=np.float64)
            bias_vector = np.asarray(biases, dtype=np.float64)
            values = values @ weight_matrix.T + bias_vector
            if layer_index < len(self.layer_weights) - 1:
                values = np.maximum(values, 0.0)
        return values.reshape(-1)

    def predict_remaining_many(
        self, features: Sequence[Sequence[float]]
    ) -> np.ndarray:
        mu = self.predict_mu_many(features)
        try:
            with np.errstate(over="raise", invalid="raise"):
                remaining = np.expm1(mu + 0.5 * self.residual_variance)
        except FloatingPointError as error:
            raise ValueError("PLP prediction overflowed during inverse log transform") from error
        return np.maximum(0.0, remaining)

    def predict_mu(self, features: Sequence[float]) -> float:
        return float(self.predict_mu_many([features])[0])

    def predict_remaining(self, features: Sequence[float]) -> float:
        return float(self.predict_remaining_many([features])[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_type": "standardized_mlp_shifted_lognormal_remaining_length",
            "method": "project_plp_only",
            "target": self.target,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "layer_weights": [
                [list(row) for row in layer] for layer in self.layer_weights
            ],
            "layer_biases": [list(layer) for layer in self.layer_biases],
            "residual_variance": self.residual_variance,
            "residual_variance_estimator": "sequence_balanced_mle",
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object]
    ) -> StandardizedMLPRemainingLength:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported dynamic model schema_version")
        if payload.get("model_type") != "standardized_mlp_shifted_lognormal_remaining_length":
            raise ValueError("unsupported dynamic model_type")
        if payload.get("method") != "project_plp_only":
            raise ValueError("dynamic model is not a Dynamic-Signal MLP v1 model")
        if payload.get("target") != "log1p_remaining_tokens":
            raise ValueError("PLP target must be log1p_remaining_tokens")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),  # type: ignore[arg-type]
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),  # type: ignore[arg-type]
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),  # type: ignore[arg-type]
            layer_weights=tuple(
                tuple(tuple(float(value) for value in row) for row in layer)
                for layer in payload["layer_weights"]  # type: ignore[union-attr]
            ),
            layer_biases=tuple(
                tuple(float(value) for value in layer)
                for layer in payload["layer_biases"]  # type: ignore[union-attr]
            ),
            residual_variance=float(payload["residual_variance"]),
            hidden_sizes=tuple(
                int(value) for value in payload["hidden_sizes"]  # type: ignore[union-attr]
            ),
            dropout=float(payload["dropout"]),
        )


def build_progressive_samples(
    trace: GenerationTrace,
    *,
    prompt_family_id: str,
) -> list[ProgressiveSample]:
    """Build sequence-balanced samples from all saved non-terminal trace points."""

    trace.validate()
    non_terminal = [point for point in trace.points if point.remaining_length > 0]
    if not non_terminal:
        return []
    sequence_weight = 1.0 / len(non_terminal)
    samples = []
    for point in non_terminal:
        entropy_mean = point.entropy if point.entropy_mean is None else point.entropy_mean
        entropy_slope = 0.0 if point.entropy_slope is None else point.entropy_slope
        samples.append(
            ProgressiveSample(
                prompt_id=trace.prompt_id,
                prompt_family_id=prompt_family_id,
                seed=trace.seed,
                step=point.step,
                output_tokens=trace.output_tokens,
                remaining_tokens=point.remaining_length,
                features=(
                    float(point.step),
                    point.entropy,
                    entropy_mean,
                    entropy_slope,
                    point.eos_probability,
                ),
                sequence_weight=sequence_weight,
            )
        )
    return samples


def fit_plp_mlp(
    features: Sequence[Sequence[float]],
    remaining_tokens: Sequence[int],
    sequence_weights: Sequence[float],
    *,
    hidden_sizes: Sequence[int] = (128, 64),
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> tuple[StandardizedMLPRemainingLength, dict[str, object]]:
    """Fit Dynamic-Signal MLP v1 with deterministic, sequence-balanced MSE."""

    matrix = np.asarray(features, dtype=np.float32)
    lengths = np.asarray(remaining_tokens, dtype=np.float32)
    weights = np.asarray(sequence_weights, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if matrix.shape[1] != len(PLP_FEATURE_NAMES):
        raise ValueError(
            f"Dynamic-Signal MLP v1 requires exactly {len(PLP_FEATURE_NAMES)} features"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PLP features must contain only finite values")
    if lengths.shape != (matrix.shape[0],) or weights.shape != (matrix.shape[0],):
        raise ValueError("targets and sequence weights must align with features")
    if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(weights)):
        raise ValueError("targets and sequence weights must be finite")
    if np.any(lengths < 0) or np.any(weights <= 0):
        raise ValueError("remaining lengths must be non-negative and weights positive")
    if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden_sizes must contain positive values")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid PLP training hyperparameters")

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            "PLP training requires PyTorch; use the AutoDL/服务器 PyTorch image or "
            "install the project model dependencies"
        ) from error

    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    target = np.log1p(lengths)
    # Every rollout contributes the same total loss even if it has more trace points.
    normalized_weights = weights * (len(weights) / float(weights.sum()))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PLP config requests CUDA, but CUDA is unavailable")

    layer_sizes = [matrix.shape[1], *(int(size) for size in hidden_sizes), 1]
    modules: list[nn.Module] = []
    for layer_index, (input_size, output_size) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], strict=True)
    ):
        modules.append(nn.Linear(input_size, output_size))
        if layer_index < len(layer_sizes) - 2:
            modules.extend((nn.ReLU(), nn.Dropout(dropout)))
    network = nn.Sequential(*modules).to(resolved_device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )
    dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(target),
        torch.from_numpy(normalized_weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    epoch_losses: list[float] = []
    for _ in range(epochs):
        network.train()
        weighted_squared_error = 0.0
        seen_weight = 0.0
        for batch_features, batch_target, batch_weights in loader:
            batch_features = batch_features.to(resolved_device)
            batch_target = batch_target.to(resolved_device)
            batch_weights = batch_weights.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predicted = network(batch_features).squeeze(-1)
            loss = (batch_weights * (predicted - batch_target).square()).mean()
            loss.backward()
            optimizer.step()
            weighted_squared_error += float(
                (batch_weights * (predicted.detach() - batch_target).square()).sum().item()
            )
            seen_weight += float(batch_weights.sum().item())
        epoch_losses.append(weighted_squared_error / seen_weight)

    network.eval()
    with torch.no_grad():
        fitted_mu = (
            network(torch.from_numpy(standardized).to(resolved_device))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    residuals = target - fitted_mu
    residual_variance = float(
        np.sum(normalized_weights * np.square(residuals)) / np.sum(normalized_weights)
    )
    linear_layers = [module for module in network if isinstance(module, nn.Linear)]
    layer_weights = tuple(
        tuple(tuple(float(value) for value in row) for row in layer.weight.detach().cpu().tolist())
        for layer in linear_layers
    )
    layer_biases = tuple(
        tuple(float(value) for value in layer.bias.detach().cpu().tolist())
        for layer in linear_layers
    )
    model = StandardizedMLPRemainingLength(
        feature_names=PLP_FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        layer_weights=layer_weights,
        layer_biases=layer_biases,
        residual_variance=residual_variance,
        hidden_sizes=tuple(int(size) for size in hidden_sizes),
        dropout=dropout,
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
        "training_sample_count": len(matrix),
        "final_sequence_balanced_mse_log1p": residual_variance,
        "epoch_losses": epoch_losses,
    }
    return model, report


def scheduled_gamma(step: int, midpoint: float = 64.0, scale: float = 24.0) -> float:
    """Weight that gradually transfers trust from the prior to decode evidence."""

    if step < 0 or scale <= 0:
        raise ValueError("step must be non-negative and scale must be positive")
    return 1.0 / (1.0 + math.exp(-(step - midpoint) / scale))


def hybrid_total_length(
    prior_length: float,
    step: int,
    predicted_remaining: float,
    gamma: float,
) -> float:
    if prior_length < 0 or step < 0 or predicted_remaining < 0:
        raise ValueError("length values must be non-negative")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    dynamic_total = step + predicted_remaining
    return (1.0 - gamma) * prior_length + gamma * dynamic_total
