"""Reservoir encoder and nonlinear latent dynamics for predator-prey histories."""

from __future__ import annotations

import torch
from torch import nn

from common.reservoir import make_projection


def make_sparse_recurrent_matrix(
    size: int,
    radius: float,
    density: float,
    generator: torch.Generator,
    power_iterations: int = 50,
) -> torch.Tensor:
    """Create a sparse recurrence without a full eigendecomposition."""

    matrix = 2.0 * torch.rand((size, size), generator=generator) - 1.0
    matrix *= torch.rand((size, size), generator=generator) < density
    vector = torch.randn(size, generator=generator)
    vector /= torch.linalg.vector_norm(vector)
    observed_radius = 0.0
    for _ in range(power_iterations):
        product = matrix @ vector
        observed_radius = float(torch.linalg.vector_norm(product))
        if observed_radius < 1e-12:
            raise RuntimeError("sampled a recurrent matrix with zero radius")
        vector = product / observed_radius
    matrix *= radius / observed_radius
    return matrix.to_sparse().coalesce()


class RabbitReservoir(nn.Module):
    """Encode rabbit-wolf histories into a learned latent state.

    The autonomous recurrence contains every unique quadratic latent product.
    This is the smallest polynomial family that can express the bilinear terms
    in Lotka-Volterra dynamics. Rabbit and wolf observations enter together.
    """

    def __init__(
        self,
        *,
        nodes: int = 1000,
        latent_size: int = 8,
        spectral_radius: float = 1.05,
        density: float = 0.1,
        leak_rate: float = 0.1,
        latent_step_size: float = 0.1,
        history_taps: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if latent_size < 1:
            raise ValueError("latent_size must be positive")
        if not 0.0 < latent_step_size <= 1.0:
            raise ValueError("latent_step_size must be in (0, 1]")
        if history_taps < 1:
            raise ValueError("history_taps must be positive")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.nodes = nodes
        self.latent_size = latent_size
        self.leak_rate = leak_rate
        self.latent_step_size = latent_step_size
        self.history_taps = history_taps
        self.reservoir_feature_size = nodes * history_taps
        self.register_buffer(
            "reservoir_recurrence",
            make_sparse_recurrent_matrix(
                nodes,
                spectral_radius,
                density,
                generator,
            ),
        )
        self.register_buffer(
            "input_projection",
            make_projection(nodes, 2, 1.5, generator),
        )
        quadratic_indices = torch.triu_indices(latent_size, latent_size)
        self.register_buffer("quadratic_row", quadratic_indices[0])
        self.register_buffer("quadratic_column", quadratic_indices[1])

        quadratic_size = latent_size * (latent_size + 1) // 2
        self.encoder_weight = nn.Parameter(
            torch.empty(latent_size, self.reservoir_feature_size)
        )
        self.encoder_bias = nn.Parameter(torch.zeros(latent_size))
        self.transition_linear = nn.Parameter(torch.zeros(latent_size, latent_size))
        self.transition_quadratic = nn.Parameter(
            torch.zeros(latent_size, quadratic_size)
        )
        self.transition_bias = nn.Parameter(torch.zeros(latent_size))
        self.readout_weight = nn.Parameter(torch.empty(2, latent_size))
        self.readout_bias = nn.Parameter(torch.zeros(2))
        nn.init.xavier_uniform_(self.encoder_weight, generator=generator)
        nn.init.xavier_uniform_(self.readout_weight, generator=generator)

    def _reservoir_update(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        drive = observation @ self.input_projection.T
        recurrence = torch.sparse.mm(self.reservoir_recurrence, state.T).T
        candidate = torch.tanh(recurrence + drive)
        return (1.0 - self.leak_rate) * state + self.leak_rate * candidate

    def reservoir_state(self, history: torch.Tensor) -> torch.Tensor:
        """Concatenate evenly spaced fixed-reservoir states from a history."""

        if history.ndim != 3 or history.shape[2] != 2:
            raise ValueError("history must have shape [batch, history_length, 2]")
        history_length = history.shape[1]
        if history_length < self.history_taps:
            raise ValueError(
                f"history_length must be at least {self.history_taps}"
            )
        tap_indices = [
            (tap * history_length + self.history_taps - 1) // self.history_taps - 1
            for tap in range(1, self.history_taps + 1)
        ]
        state = history.new_zeros((len(history), self.nodes))
        tapped_states = []
        for index, observation in enumerate(history.unbind(dim=1)):
            state = self._reservoir_update(state, observation)
            if index in tap_indices:
                tapped_states.append(state)
        return torch.cat(tapped_states, dim=1)

    def encode_reservoir_state(self, state: torch.Tensor) -> torch.Tensor:
        """Project a cached fixed-reservoir state into the learned latent."""

        if state.ndim != 2 or state.shape[1] != self.reservoir_feature_size:
            raise ValueError(
                f"state must have shape [batch, {self.reservoir_feature_size}]"
            )
        return torch.tanh(state @ self.encoder_weight.T + self.encoder_bias)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Encode standardized rabbit-wolf histories of shape [batch, time, 2]."""

        return self.encode_reservoir_state(self.reservoir_state(history))

    def quadratic_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Return all products z_i*z_j with i <= j."""

        return latent[:, self.quadratic_row] * latent[:, self.quadratic_column]

    def latent_step(self, latent: torch.Tensor) -> torch.Tensor:
        """Advance one step with a bounded learned quadratic drift."""

        quadratic = self.quadratic_features(latent)
        drift = torch.tanh(
            latent @ self.transition_linear.T
            + quadratic @ self.transition_quadratic.T
            + self.transition_bias
        )
        return latent + self.latent_step_size * drift

    def rollout(
        self,
        history: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict standardized log-rabbit and log-wolf values autonomously."""

        return self.rollout_from_reservoir_state(
            self.reservoir_state(history),
            horizon,
        )

    def rollout_from_reservoir_state(
        self,
        reservoir_state: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out from a precomputed fixed-reservoir state."""

        if horizon < 1:
            raise ValueError("horizon must be positive")
        latent = self.encode_reservoir_state(reservoir_state)
        initial_latent = latent
        predictions = []
        latents = []
        for _ in range(horizon):
            latent = self.latent_step(latent)
            predictions.append(latent @ self.readout_weight.T + self.readout_bias)
            latents.append(latent)
        return (
            torch.stack(predictions, dim=1),
            torch.stack(latents, dim=1),
            initial_latent,
        )
