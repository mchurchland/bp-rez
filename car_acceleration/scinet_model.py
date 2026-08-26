"""SciNet-style variational MLP adapted to the car experiment."""

from __future__ import annotations

import torch
from torch import nn


class CarSciNet(nn.Module):
    """A 100-100 ELU encoder/decoder with a 2D variational latent.

    The original SciNet idea is retained: an MLP produces a Gaussian latent
    and a decoder reconstructs the target from latent states. The car needs a
    learned affine latent transition rather than the solar experiment's fixed
    additive phase increment, because an accelerating car does not move at a
    constant speed.
    """

    def __init__(
        self,
        *,
        history_length: int = 3,
        latent_size: int = 2,
        hidden_size: int = 100,
        target_latent_std: float = 0.1,
        beta: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if history_length < 1 or latent_size != 2 or hidden_size < 1:
            raise ValueError("invalid SciNet dimensions")
        self.latent_size = latent_size
        self.target_latent_std = target_latent_std
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(history_length, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        self.transition = nn.Parameter(torch.eye(latent_size))
        self.transition_bias = nn.Parameter(torch.zeros(latent_size))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, generator=generator)
                nn.init.normal_(module.bias, std=0.1, generator=generator)

    def _distribution(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(history)
        mean = torch.tanh(encoded[:, : self.latent_size])
        # Clamping keeps the teaching experiment numerically stable while
        # retaining the variational mean/log-standard-deviation structure.
        log_sigma = encoded[:, self.latent_size :].clamp(-5.0, 1.0)
        return mean, log_sigma

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self._distribution(history)[0]

    def _latents_from_initial(
        self, initial_latent: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        latents = []
        latent = initial_latent
        for _ in range(horizon):
            latent = latent @ self.transition.T + self.transition_bias
            latents.append(latent)
        return torch.stack(latents, dim=1)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents.flatten(0, 1)).reshape(
            len(latents), latents.shape[1]
        )

    def rollout(
        self, history: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _ = self._distribution(history)
        latents = self._latents_from_initial(mean, horizon)
        return self._decode(latents), latents, mean

    def training_rollout(
        self, history: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_sigma = self._distribution(history)
        sample = mean + torch.exp(log_sigma) * torch.randn_like(mean)
        latents = self._latents_from_initial(sample, horizon)
        prediction = self._decode(latents)
        variance_ratio = torch.exp(2.0 * log_sigma) / self.target_latent_std**2
        mean_term = mean.square() / self.target_latent_std**2
        kl = 0.5 * torch.sum(
            mean_term
            + variance_ratio
            - 2.0 * log_sigma
            + 2.0 * torch.log(mean.new_tensor(self.target_latent_std))
            - 1.0,
            dim=-1,
        ).mean()
        return prediction, latents, self.beta * kl

