"""Models and paired random-feature construction for the NARMA benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from common.reservoir import make_projection, make_recurrent_matrix


MODEL_NAMES = (
    "esn_ridge",
    "large_esn_ridge",
    "deep_esn_ridge",
    "random_bottleneck",
    "pca_bottleneck",
    "learned_linear",
    "learned_nonlinear",
    "gru",
)

GRADIENT_MODEL_NAMES = ("learned_linear", "learned_nonlinear", "gru")


@dataclass(frozen=True)
class ReservoirConfig:
    """Layer-specific reservoir and optimizer-independent model settings."""

    nodes_1: int = 150
    nodes_2: int = 150
    latent_size: int = 10
    spectral_radius_1: float = 0.9
    spectral_radius_2: float = 0.9
    leak_rate_1: float = 1.0
    leak_rate_2: float = 1.0
    density_1: float = 0.1
    density_2: float = 0.1
    input_scale: float = 0.5
    interlayer_scale: float = 1.0
    bias_scale_1: float = 0.0
    bias_scale_2: float = 0.0
    gru_hidden_size: int = 22

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _named_generator(seed: int, name: str) -> torch.Generator:
    """Create a construction-order-independent CPU RNG stream."""

    payload = f"bp-reservoir:narma:{seed}:{name}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return torch.Generator(device="cpu").manual_seed(derived)


def _matrix(config: ReservoirConfig, layer: int, seed: int) -> torch.Tensor:
    size = config.nodes_1 if layer == 1 else config.nodes_2
    radius = config.spectral_radius_1 if layer == 1 else config.spectral_radius_2
    density = config.density_1 if layer == 1 else config.density_2
    return make_recurrent_matrix(
        size, radius, density, _named_generator(seed, f"A{layer}")
    )


def _bias(size: int, scale: float, seed: int, name: str) -> torch.Tensor:
    if scale == 0.0:
        return torch.zeros(size)
    return make_projection(size, 1, scale, _named_generator(seed, name)).squeeze(-1)


class BenchmarkModel(nn.Module):
    """Common head-independent feature interface for every benchmark model."""

    feature_size: int
    recurrent_state_size: int
    bottleneck_size: int = 0
    gradient_model: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor_fitted = False

    def prepare(self, train_u: torch.Tensor, washout: int) -> None:
        """Fit unsupervised, training-only transforms, if present."""

        self.preprocessor_fitted = True

    def features(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(u))

    def set_readout(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        with torch.no_grad():
            self.readout.weight.copy_(
                weight.to(
                    device=self.readout.weight.device, dtype=self.readout.weight.dtype
                )
            )
            self.readout.bias.copy_(
                bias.to(device=self.readout.bias.device, dtype=self.readout.bias.dtype)
            )

    @property
    def encoder_gradient_parameters(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("readout.")
        )

    @property
    def ridge_parameters(self) -> int:
        return self.feature_size + 1

    @property
    def fitted_preprocessor_parameters(self) -> int:
        return 0

    @property
    def normalization_statistics(self) -> int:
        return 0

    @property
    def pca_fitted_parameters(self) -> int:
        return 0

    @property
    def fixed_coefficient_count(self) -> int:
        return sum(
            value.numel()
            for name, value in self.named_buffers()
            if name.startswith(("A", "B", "V", "Q", "R", "q"))
        )


class ReservoirLayerMixin:
    @staticmethod
    def _update(
        state: torch.Tensor,
        recurrent: torch.Tensor,
        drive: torch.Tensor,
        bias: torch.Tensor,
        leak: float,
    ) -> torch.Tensor:
        candidate = torch.tanh(recurrent @ state + drive + bias)
        return (1.0 - leak) * state + leak * candidate


class SingleESN(BenchmarkModel, ReservoirLayerMixin):
    """One fixed ESN whose output is always fitted by ridge regression."""

    def __init__(
        self, nodes: int, config: ReservoirConfig, seed: int, *, large: bool
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.feature_size = nodes
        self.recurrent_state_size = nodes
        radius = config.spectral_radius_1
        density = config.density_1
        label = "A_large" if large else "A1"
        self.register_buffer(
            "A",
            make_recurrent_matrix(
                nodes, radius, density, _named_generator(seed, label)
            ),
        )
        input_label = "B_large" if large else "B1"
        bias_label = "q_large" if large else "q1"
        self.register_buffer(
            "B",
            make_projection(
                nodes, 1, config.input_scale, _named_generator(seed, input_label)
            ),
        )
        self.register_buffer(
            "q",
            _bias(nodes, config.bias_scale_1, seed, bias_label),
        )
        self.leak = config.leak_rate_1
        self.readout = nn.Linear(nodes, 1)
        self.readout.requires_grad_(False)

    def features(self, u: torch.Tensor) -> torch.Tensor:
        state = u.new_zeros(self.nodes)
        values = []
        for sample in u:
            state = self._update(state, self.A, self.B @ sample, self.q, self.leak)
            values.append(state)
        return torch.stack(values)


class TwoReservoirBase(BenchmarkModel, ReservoirLayerMixin):
    """Shared paired construction for all two-reservoir variants."""

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.nodes_1 = config.nodes_1
        self.nodes_2 = config.nodes_2
        self.register_buffer("A1", _matrix(config, 1, seed))
        self.register_buffer("A2", _matrix(config, 2, seed))
        self.register_buffer(
            "B1",
            make_projection(
                config.nodes_1,
                1,
                config.input_scale,
                _named_generator(seed, "B1"),
            ),
        )
        self.register_buffer(
            "q1",
            _bias(config.nodes_1, config.bias_scale_1, seed, "q1"),
        )
        self.register_buffer(
            "q2",
            _bias(config.nodes_2, config.bias_scale_2, seed, "q2"),
        )
        self.leak_1 = config.leak_rate_1
        self.leak_2 = config.leak_rate_2

    def _first_states(self, u: torch.Tensor) -> torch.Tensor:
        state = u.new_zeros(self.nodes_1)
        values = []
        for sample in u:
            state = self._update(state, self.A1, self.B1 @ sample, self.q1, self.leak_1)
            values.append(state)
        return torch.stack(values)


class DeepESN(TwoReservoirBase):
    """Two fixed reservoirs with the stronger concatenated-layer ridge head."""

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__(config, seed)
        self.register_buffer(
            "V",
            make_projection(
                config.nodes_2,
                config.nodes_1,
                config.interlayer_scale,
                _named_generator(seed, "V"),
            ),
        )
        self.feature_size = config.nodes_1 + config.nodes_2
        self.recurrent_state_size = self.feature_size
        self.readout = nn.Linear(self.feature_size, 1)
        self.readout.requires_grad_(False)

    def features(self, u: torch.Tensor) -> torch.Tensor:
        first = self._first_states(u)
        second = u.new_zeros(self.nodes_2)
        values = []
        for x1 in first:
            second = self._update(second, self.A2, self.V @ x1, self.q2, self.leak_2)
            values.append(torch.cat((x1, second)))
        return torch.stack(values)


class BottleneckBase(TwoReservoirBase):
    """Shared train-only first-state normalization and second reservoir."""

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__(config, seed)
        self.bottleneck_size = config.latent_size
        self.feature_size = config.nodes_2
        self.recurrent_state_size = config.nodes_1 + config.nodes_2
        self.register_buffer(
            "R",
            make_projection(
                config.nodes_2,
                config.latent_size,
                config.interlayer_scale,
                _named_generator(seed, "R"),
            ),
        )
        self.register_buffer("x1_mean", torch.zeros(config.nodes_1))
        self.register_buffer("x1_scale", torch.ones(config.nodes_1))

    @property
    def fitted_preprocessor_parameters(self) -> int:
        return 2 * self.nodes_1

    @property
    def normalization_statistics(self) -> int:
        return 2 * self.nodes_1

    def prepare(self, train_u: torch.Tensor, washout: int) -> None:
        with torch.no_grad():
            first = self._first_states(train_u)
            selected = first[washout:]
            if len(selected) < 2:
                raise ValueError("not enough post-washout states for normalization")
            self.x1_mean.copy_(selected.mean(dim=0))
            scale = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
            self.x1_scale.copy_(scale)
        self.preprocessor_fitted = True

    def _latent(self, normalized_x1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def features(self, u: torch.Tensor) -> torch.Tensor:
        first = self._first_states(u)
        normalized = (first - self.x1_mean) / self.x1_scale
        second = u.new_zeros(self.nodes_2)
        values = []
        for x1 in normalized:
            latent = self._latent(x1)
            second = self._update(
                second, self.A2, self.R @ latent, self.q2, self.leak_2
            )
            values.append(second)
        return torch.stack(values)


class RandomBottleneck(BottleneckBase):
    """Fixed random 10D projection between paired fixed reservoirs."""

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__(config, seed)
        self.register_buffer(
            "Q",
            make_projection(
                config.latent_size,
                config.nodes_1,
                1.0,
                _named_generator(seed, "Q"),
            ),
        )
        self.readout = nn.Linear(self.feature_size, 1)
        self.readout.requires_grad_(False)

    def _latent(self, normalized_x1: torch.Tensor) -> torch.Tensor:
        return self.Q @ normalized_x1


class PCABottleneck(BottleneckBase):
    """Training-only PCA projection between the fixed reservoirs."""

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__(config, seed)
        self.register_buffer(
            "pca_components", torch.zeros(config.latent_size, config.nodes_1)
        )
        self.register_buffer("pca_mean", torch.zeros(config.nodes_1))
        self.readout = nn.Linear(self.feature_size, 1)
        self.readout.requires_grad_(False)

    @property
    def fitted_preprocessor_parameters(self) -> int:
        return (
            super().fitted_preprocessor_parameters
            + self.pca_components.numel()
            + self.pca_mean.numel()
        )

    @property
    def pca_fitted_parameters(self) -> int:
        return self.pca_components.numel() + self.pca_mean.numel()

    def prepare(self, train_u: torch.Tensor, washout: int) -> None:
        super().prepare(train_u, washout)
        with torch.no_grad():
            first = self._first_states(train_u)
            normalized = (first[washout:] - self.x1_mean) / self.x1_scale
            self.pca_mean.copy_(normalized.mean(dim=0))
            centered = normalized - self.pca_mean
            covariance = centered.T @ centered / max(1, len(centered) - 1)
            _, eigenvectors = torch.linalg.eigh(covariance)
            components = eigenvectors[:, -self.bottleneck_size :].T.flip(0)
            pivots = torch.argmax(torch.abs(components), dim=1)
            signs = torch.sign(
                components[
                    torch.arange(len(components), device=components.device), pivots
                ]
            )
            signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
            components = components * signs.unsqueeze(1)
            self.pca_components.copy_(components)

    def _latent(self, normalized_x1: torch.Tensor) -> torch.Tensor:
        return self.pca_components @ (normalized_x1 - self.pca_mean)


class LearnedBottleneck(BottleneckBase):
    """Task-trained linear or tanh bottleneck with frozen recurrent matrices."""

    gradient_model = True

    def __init__(self, config: ReservoirConfig, seed: int, *, nonlinear: bool) -> None:
        super().__init__(config, seed)
        self.nonlinear = nonlinear
        self.W = nn.Parameter(torch.empty(config.latent_size, config.nodes_1))
        self.b = nn.Parameter(torch.zeros(config.latent_size))
        self.readout = nn.Linear(self.feature_size, 1)
        nn.init.xavier_uniform_(self.W, generator=_named_generator(seed, "W"))
        nn.init.xavier_uniform_(
            self.readout.weight, generator=_named_generator(seed, "W_out")
        )
        nn.init.zeros_(self.readout.bias)

    def _latent(self, normalized_x1: torch.Tensor) -> torch.Tensor:
        latent = self.W @ normalized_x1 + self.b
        return torch.tanh(latent) if self.nonlinear else latent


class SmallGRU(BenchmarkModel):
    """Parameter-matched recurrent neural-network control."""

    gradient_model = True

    def __init__(self, config: ReservoirConfig, seed: int) -> None:
        super().__init__()
        self.hidden_size = config.gru_hidden_size
        self.feature_size = self.hidden_size
        self.recurrent_state_size = self.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_named_generator(seed, "gru").initial_seed())
            self.cell = nn.GRUCell(1, self.hidden_size)
            self.readout = nn.Linear(self.hidden_size, 1)

    def features(self, u: torch.Tensor) -> torch.Tensor:
        state = u.new_zeros(self.hidden_size)
        values = []
        for sample in u:
            state = self.cell(sample.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            values.append(state)
        return torch.stack(values)


def build_model(
    name: str,
    config: ReservoirConfig,
    seed: int,
) -> BenchmarkModel:
    """Build a model with named RNG streams shared across relevant ablations."""

    if name == "esn_ridge":
        return SingleESN(config.nodes_1, config, seed, large=False)
    if name == "large_esn_ridge":
        return SingleESN(config.nodes_1 + config.nodes_2, config, seed, large=True)
    if name == "deep_esn_ridge":
        return DeepESN(config, seed)
    if name == "random_bottleneck":
        return RandomBottleneck(config, seed)
    if name == "pca_bottleneck":
        return PCABottleneck(config, seed)
    if name == "learned_linear":
        return LearnedBottleneck(config, seed, nonlinear=False)
    if name == "learned_nonlinear":
        return LearnedBottleneck(config, seed, nonlinear=True)
    if name == "gru":
        return SmallGRU(config, seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
