"""Reservoir model variants used in the NARMA10 comparison."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


MODEL_NAMES = (
    "single_esn",
    "deep_esn",
    "proposed_nonlinear",
    "proposed_linear",
    "large_esn",
)


def _spectral_radius(matrix: torch.Tensor) -> float:
    values = np.linalg.eigvals(matrix.detach().cpu().numpy().astype(np.float64))
    return float(np.max(np.abs(values)))


def make_recurrent_matrix(
    size: int,
    radius: float,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create a sparse random matrix scaled to the requested spectral radius."""

    if not 0.0 < density <= 1.0:
        raise ValueError("reservoir density must be in (0, 1]")
    if radius < 0.0:
        raise ValueError("spectral radius must be nonnegative")
    matrix = 2.0 * torch.rand((size, size), generator=generator) - 1.0
    matrix *= torch.rand((size, size), generator=generator) < density
    observed = _spectral_radius(matrix)
    if observed < 1e-12:
        raise RuntimeError("sampled recurrent matrix has zero spectral radius")
    return (matrix * (radius / observed)).to(torch.float32)


def make_projection(
    out_features: int,
    in_features: int,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create a fixed fan-in-scaled random projection."""

    bound = scale / np.sqrt(in_features)
    return torch.empty(out_features, in_features).uniform_(
        -bound, bound, generator=generator
    )


class ReservoirBase(nn.Module):
    """Common helpers for sequence-to-sequence reservoirs."""

    def __init__(self, leak_rate: float) -> None:
        super().__init__()
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        self.leak_rate = leak_rate

    def _update(
        self, state: torch.Tensor, recurrent: torch.Tensor, drive: torch.Tensor
    ) -> torch.Tensor:
        candidate = torch.tanh(recurrent @ state + drive)
        return (1.0 - self.leak_rate) * state + self.leak_rate * candidate

    @property
    def fixed_matrix_names(self) -> tuple[str, ...]:
        raise NotImplementedError


class SingleESN(ReservoirBase):
    """One fixed reservoir with an Adam-trained linear readout."""

    def __init__(
        self,
        nodes: int,
        spectral_radius: float,
        input_scale: float,
        density: float,
        leak_rate: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__(leak_rate)
        self.nodes = nodes
        self.register_buffer(
            "A", make_recurrent_matrix(nodes, spectral_radius, density, generator)
        )
        self.register_buffer(
            "B", make_projection(nodes, 1, input_scale, generator)
        )
        self.W_out = nn.Parameter(torch.empty(1, nodes))
        self.c = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W_out, generator=generator)

    @property
    def fixed_matrix_names(self) -> tuple[str, ...]:
        return ("A", "B")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        state = u.new_zeros(self.nodes)
        outputs = []
        for value in u:
            state = self._update(state, self.A, self.B @ value)
            outputs.append(self.W_out @ state + self.c)
        return torch.stack(outputs)


class FixedDeepESN(ReservoirBase):
    """Standard two-layer DeepESN with a fixed direct interlayer projection."""

    def __init__(
        self,
        nodes_1: int,
        nodes_2: int,
        spectral_radius: float,
        input_scale: float,
        interlayer_scale: float,
        density: float,
        leak_rate: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__(leak_rate)
        self.nodes_1 = nodes_1
        self.nodes_2 = nodes_2
        self.register_buffer(
            "A1", make_recurrent_matrix(nodes_1, spectral_radius, density, generator)
        )
        self.register_buffer(
            "A2", make_recurrent_matrix(nodes_2, spectral_radius, density, generator)
        )
        self.register_buffer(
            "B1", make_projection(nodes_1, 1, input_scale, generator)
        )
        self.register_buffer(
            "V", make_projection(nodes_2, nodes_1, interlayer_scale, generator)
        )
        self.W_out = nn.Parameter(torch.empty(1, nodes_2))
        self.c = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W_out, generator=generator)

    @property
    def fixed_matrix_names(self) -> tuple[str, ...]:
        return ("A1", "A2", "B1", "V")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x1 = u.new_zeros(self.nodes_1)
        x2 = u.new_zeros(self.nodes_2)
        outputs = []
        for value in u:
            x1 = self._update(x1, self.A1, self.B1 @ value)
            x2 = self._update(x2, self.A2, self.V @ x1)
            outputs.append(self.W_out @ x2 + self.c)
        return torch.stack(outputs)


class IntermediateReadoutESN(ReservoirBase):
    """Two frozen reservoirs joined by a trainable low-dimensional readout.

    ``A1``, ``A2``, ``B1``, and ``R`` are buffers and never receive gradients.
    Autograd still differentiates through their matrix operations, allowing the
    loss at later times to train ``W`` and ``b`` through the second reservoir's
    recurrent dynamics.
    """

    def __init__(
        self,
        nodes_1: int,
        nodes_2: int,
        latent_size: int,
        spectral_radius: float,
        input_scale: float,
        interlayer_scale: float,
        density: float,
        leak_rate: float,
        nonlinear: bool,
        generator: torch.Generator,
    ) -> None:
        super().__init__(leak_rate)
        self.nodes_1 = nodes_1
        self.nodes_2 = nodes_2
        self.latent_size = latent_size
        self.nonlinear = nonlinear
        self.register_buffer(
            "A1", make_recurrent_matrix(nodes_1, spectral_radius, density, generator)
        )
        self.register_buffer(
            "A2", make_recurrent_matrix(nodes_2, spectral_radius, density, generator)
        )
        self.register_buffer(
            "B1", make_projection(nodes_1, 1, input_scale, generator)
        )
        self.register_buffer(
            "R", make_projection(nodes_2, latent_size, interlayer_scale, generator)
        )
        # These are deliberately the only trainable tensors and their names
        # mirror the mathematical specification exactly.
        self.W = nn.Parameter(torch.empty(latent_size, nodes_1))
        self.b = nn.Parameter(torch.zeros(latent_size))
        self.W_out = nn.Parameter(torch.empty(1, nodes_2))
        self.c = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W, generator=generator)
        nn.init.xavier_uniform_(self.W_out, generator=generator)

    @property
    def fixed_matrix_names(self) -> tuple[str, ...]:
        return ("A1", "A2", "B1", "R")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x1 = u.new_zeros(self.nodes_1)
        x2 = u.new_zeros(self.nodes_2)
        outputs = []
        for value in u:
            x1 = self._update(x1, self.A1, self.B1 @ value)
            h = self.W @ x1 + self.b
            if self.nonlinear:
                h = torch.tanh(h)
            x2 = self._update(x2, self.A2, self.R @ h)
            outputs.append(self.W_out @ x2 + self.c)
        return torch.stack(outputs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_frozen_reservoirs(model: ReservoirBase) -> None:
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    for name in model.fixed_matrix_names:
        if name in parameter_names:
            raise AssertionError(f"fixed matrix {name} was registered as a parameter")
        if name not in buffer_names:
            raise AssertionError(f"fixed matrix {name} was not registered as a buffer")
        if buffer_names[name].requires_grad:
            raise AssertionError(f"fixed matrix {name} requires gradients")


def build_model(
    name: str,
    *,
    nodes_1: int,
    nodes_2: int,
    latent_size: int,
    spectral_radius: float,
    input_scale: float,
    interlayer_scale: float,
    density: float,
    leak_rate: float,
    seed: int,
) -> ReservoirBase:
    """Build one comparison model from a deterministic CPU RNG."""

    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    common = dict(
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        density=density,
        leak_rate=leak_rate,
        generator=generator,
    )
    if name == "single_esn":
        model = SingleESN(nodes=nodes_1, **common)
    elif name == "large_esn":
        model = SingleESN(nodes=nodes_1 + nodes_2, **common)
    elif name == "deep_esn":
        model = FixedDeepESN(
            nodes_1=nodes_1,
            nodes_2=nodes_2,
            interlayer_scale=interlayer_scale,
            **common,
        )
    else:
        model = IntermediateReadoutESN(
            nodes_1=nodes_1,
            nodes_2=nodes_2,
            latent_size=latent_size,
            interlayer_scale=interlayer_scale,
            nonlinear=name == "proposed_nonlinear",
            **common,
        )
    assert_frozen_reservoirs(model)
    return model
