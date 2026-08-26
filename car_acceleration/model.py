"""A one-reservoir car model with a learned 2D latent and linear readout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from common.reservoir import make_projection, make_recurrent_matrix


class CarReservoir(nn.Module):
    """A fixed 150-neuron encoder followed by a direct 2D latent readout.

    The encoder reads a short position history. The learned affine latent
    transition advances the two-dimensional state, and a linear readout maps
    the latent directly to the next position. There is deliberately no second
    reservoir in this version.
    """

    def __init__(
        self,
        *,
        nodes: int = 150,
        latent_size: int = 2,
        spectral_radius: float = 0.9,
        density: float = 0.1,
        leak_rate: float = 0.8,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if nodes != 150:
            raise ValueError("this experiment is intentionally fixed at 150 nodes")
        #if latent_size != 2:
        #    raise ValueError("this experiment is intentionally fixed at a 2D latent")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.nodes = nodes
        self.latent_size = latent_size
        self.leak_rate = leak_rate
        self.register_buffer(
            "A1", make_recurrent_matrix(nodes, spectral_radius, density, generator)
        )
        self.register_buffer("B1", make_projection(nodes, 1, 1.5, generator))

        self.encoder_weight = nn.Parameter(torch.empty(latent_size, nodes))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_size))
        self.transition = nn.Parameter(torch.eye(latent_size))
        self.transition_bias = nn.Parameter(torch.zeros(latent_size))
        self.raw_readout = nn.Parameter(torch.full((1, latent_size), -2.0))
        self.readout_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.encoder_weight, generator=generator)

    @property
    def readout_weights(self) -> torch.Tensor:
        """Strictly positive latent-to-position weights."""

        return F.softplus(self.raw_readout)

    def _update(self, state: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(state @ self.A1.T + drive)
        return (1.0 - self.leak_rate) * state + self.leak_rate * candidate

    def encode(self, history: torch.Tensor) -> torch.Tensor: ## this is project history onto latent
        """Encode [batch, history_length] positions into a 2D latent."""
        if history.ndim != 2:
            raise ValueError("history must have shape [batch, history_length]")
        
        
        state = history.new_zeros((len(history), self.nodes))
        for observation in history.unbind(dim=1):
            state = self._update(state, observation[:, None] @ self.B1.T)
        return torch.tanh(state @ self.encoder_weight.T + self.encoder_bias)

    def rollout(
        self, history: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict positions using only the evolving latent and linear readout."""

        if horizon < 1:
            raise ValueError("horizon must be positive")
        latent = self.encode(history) ## project the history onto the latent
        initial_latent = latent ##this gives us our initial latent values this basically gives us the variables of the equation
        predictions = []
        latents = []
        for _ in range(horizon):
            latent = latent @ self.transition.T + self.transition_bias ## I have dynamics beetween my two latent space nodes 
            ## this is like going through one step
            predictions.append(latent @ self.readout_weights.T + self.readout_bias) ## predict for each time step
            latents.append(latent)
        return (
            torch.stack(predictions, dim=1).squeeze(-1),
            torch.stack(latents, dim=1),
            initial_latent,
        )
