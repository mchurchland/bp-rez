"""Models for the solar-system latent-representation experiment."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .models import make_projection, make_recurrent_matrix


SOLAR_MODEL_NAMES = ("reservoir", "scinet")


class SolarModelBase(nn.Module, ABC):
    """Interface shared by the reservoir adaptation and SciNet reference."""

    latent_size: int
    variational_latent = False

    @abstractmethod
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic initial latent representation."""

    @abstractmethod
    def training_forward(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prediction, latent sequence, and representation penalty."""

    @abstractmethod
    def predict_with_latents(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic predictions and latent states."""

    def forward(self, observation: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.predict_with_latents(observation, horizon)[0]

    def latent_log_sigma(self, observation: torch.Tensor) -> torch.Tensor | None:
        """Return variational log standard deviations, when the model has them."""

        return None

    def evolution_l2_loss(self) -> torch.Tensor:
        """Return the released graph's Euler-weight regularizer, when present."""

        return next(self.parameters()).new_zeros(())


class SolarReservoir(SolarModelBase):
    """Frozen two-reservoir model with a simply evolving learned bottleneck.

    Only ``W``, ``b``, ``latent_delta``, ``W_out``, and ``c`` are trained. The
    initial Earth-view observation is encoded by the first frozen reservoir,
    after which the bottleneck is constrained to evolve by addition of one
    learned constant per week. The second frozen reservoir and its trainable
    linear readout decode the complete forecast.
    """

    def __init__(
        self,
        *,
        nodes_1: int,
        nodes_2: int,
        latent_size: int,
        spectral_radius: float,
        input_scale: float,
        interlayer_scale: float,
        density: float,
        leak_rate: float,
        encoder_steps: int,
        decoder_steps: int,
        seed: int,
        nonlinear: bool = True,
    ) -> None:
        super().__init__()
        if nodes_1 < 1 or nodes_2 < 1 or latent_size < 1:
            raise ValueError("reservoir and latent sizes must be positive")
        if encoder_steps < 1 or decoder_steps < 1:
            raise ValueError("encoder_steps and decoder_steps must be positive")
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        self.nodes_1 = nodes_1
        self.nodes_2 = nodes_2
        self.latent_size = latent_size
        self.leak_rate = leak_rate
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.nonlinear = nonlinear
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer(
            "A1", make_recurrent_matrix(nodes_1, spectral_radius, density, generator)
        )
        self.register_buffer(
            "A2", make_recurrent_matrix(nodes_2, spectral_radius, density, generator)
        )
        self.register_buffer("B1", make_projection(nodes_1, 2, input_scale, generator))
        self.register_buffer(
            "R", make_projection(nodes_2, latent_size, interlayer_scale, generator)
        )
        self.W = nn.Parameter(torch.empty(latent_size, nodes_1))
        self.b = nn.Parameter(torch.zeros(latent_size))
        self.latent_delta = nn.Parameter(torch.zeros(latent_size))
        self.W_out = nn.Parameter(torch.empty(2, nodes_2))
        self.c = nn.Parameter(torch.zeros(2))
        nn.init.xavier_uniform_(self.W, generator=generator)
        nn.init.xavier_uniform_(self.W_out, generator=generator)

    @property
    def fixed_matrix_names(self) -> tuple[str, ...]:
        return ("A1", "A2", "B1", "R")

    def _update(
        self, state: torch.Tensor, recurrent: torch.Tensor, drive: torch.Tensor
    ) -> torch.Tensor:
        candidate = torch.tanh(state @ recurrent.T + drive)
        return (1.0 - self.leak_rate) * state + self.leak_rate * candidate

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[-1] != 2:
            raise ValueError("observation must have shape [batch, 2]")
        state = observation.new_zeros((len(observation), self.nodes_1))
        drive = observation @ self.B1.T
        for _ in range(self.encoder_steps):
            state = self._update(state, self.A1, drive)
        latent = state @ self.W.T + self.b
        return torch.tanh(latent) if self.nonlinear else latent

    def _decode(
        self, initial_latent: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        latent = initial_latent
        predictions = []
        latents = []
        for _ in range(horizon):
            latents.append(latent)
            # The same memoryless frozen-reservoir decoder is applied to each
            # latent state. Resetting here prevents hidden temporal information
            # from bypassing the explicitly additive latent dynamics.
            state = initial_latent.new_zeros((len(initial_latent), self.nodes_2))
            drive = latent @ self.R.T
            for _ in range(self.decoder_steps):
                state = self._update(state, self.A2, drive)
            predictions.append(state @ self.W_out.T + self.c)
            latent = latent + self.latent_delta
        return torch.stack(predictions, dim=1), torch.stack(latents, dim=1)

    def training_forward(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_latent = self.encode(observation)
        prediction, latents = self._decode(initial_latent, horizon)
        # Deterministic analogue of SciNet's mean part of the beta-VAE KL.
        representation_penalty = 0.5 * initial_latent.square().sum(dim=-1).mean()
        return prediction, latents, representation_penalty

    def predict_with_latents(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decode(self.encode(observation), horizon)


class SolarSciNet(SolarModelBase):
    """PyTorch reference for the active SciNet Copernicus computation graph."""

    variational_latent = True

    def __init__(
        self,
        *,
        latent_size: int,
        hidden_size: int = 100,
        target_latent_std: float = 0.1,
        seed: int,
    ) -> None:
        super().__init__()
        if latent_size < 1 or hidden_size < 1:
            raise ValueError("latent_size and hidden_size must be positive")
        if target_latent_std <= 0.0:
            raise ValueError("target_latent_std must be positive")
        self.latent_size = latent_size
        self.target_latent_std = target_latent_std
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
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
            nn.Linear(hidden_size, 2),
        )
        # The released TensorFlow graph creates and L2-regularizes this matrix,
        # but its Euler update accidentally ignores the matrix and adds only
        # the bias. Retaining the unused parameter matches that active graph.
        self.euler_weight = nn.Parameter(torch.empty(latent_size, latent_size))
        self.latent_delta = nn.Parameter(torch.empty(latent_size))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, generator=generator)
                nn.init.normal_(module.bias, std=1.0, generator=generator)
        nn.init.xavier_normal_(self.euler_weight, generator=generator)
        nn.init.normal_(self.latent_delta, std=1.0, generator=generator)

    def _distribution(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        mean = torch.tanh(encoded[:, : self.latent_size])
        # The public TensorFlow source first clips this tensor and immediately
        # overwrites it with the raw encoder output. The released checkpoint
        # therefore uses an unrestricted log standard deviation. Reproduce the
        # active graph literally, including that accidental overwrite.
        log_sigma = encoded[:, self.latent_size :]
        return mean, log_sigma

    def latent_log_sigma(self, observation: torch.Tensor) -> torch.Tensor:
        return self._distribution(observation)[1]

    def evolution_l2_loss(self) -> torch.Tensor:
        # TensorFlow's tf.nn.l2_loss is sum(t ** 2) / 2.
        return 0.5 * self.euler_weight.square().sum()

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self._distribution(observation)[0]

    def _decode(
        self, initial_latent: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        steps = torch.arange(
            horizon, device=initial_latent.device, dtype=initial_latent.dtype
        )
        latents = initial_latent[:, None, :] + steps[None, :, None] * self.latent_delta
        prediction = self.decoder(latents.flatten(0, 1)).reshape(
            len(initial_latent), horizon, 2
        )
        return prediction, latents

    def _kl_divergence(
        self, mean: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        variance_ratio = torch.exp(2.0 * log_sigma) / self.target_latent_std**2
        mean_term = mean.square() / self.target_latent_std**2
        per_example = 0.5 * torch.sum(
            mean_term
            + variance_ratio
            - 2.0 * log_sigma
            + 2.0 * torch.log(mean.new_tensor(self.target_latent_std))
            - 1.0,
            dim=-1,
        )
        return per_example.mean()

    def training_forward(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_sigma = self._distribution(observation)
        sample = mean + torch.exp(log_sigma) * torch.randn_like(mean)
        prediction, latents = self._decode(sample, horizon)
        return prediction, latents, self._kl_divergence(mean, log_sigma)

    def predict_with_latents(
        self, observation: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decode(self.encode(observation), horizon)


def build_solar_model(
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
    encoder_steps: int,
    decoder_steps: int,
    scinet_hidden_size: int,
    seed: int,
) -> SolarModelBase:
    if name == "reservoir":
        return SolarReservoir(
            nodes_1=nodes_1,
            nodes_2=nodes_2,
            latent_size=latent_size,
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            interlayer_scale=interlayer_scale,
            density=density,
            leak_rate=leak_rate,
            encoder_steps=encoder_steps,
            decoder_steps=decoder_steps,
            seed=seed,
        )
    if name == "scinet":
        return SolarSciNet(
            latent_size=latent_size, hidden_size=scinet_hidden_size, seed=seed
        )
    raise ValueError(f"unknown solar model {name!r}; choose from {SOLAR_MODEL_NAMES}")
