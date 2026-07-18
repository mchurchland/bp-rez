import json

import numpy as np
import torch

from reservoir.solar_data import (
    EARTH_PERIOD_DAYS,
    MARS_PERIOD_DAYS,
    generate_solar_dataset,
    make_solar_splits,
)
from reservoir.solar_experiment import SolarExperimentConfig, run_solar_experiment
from reservoir.solar_models import SolarReservoir, SolarSciNet


def reservoir(seed: int = 3) -> SolarReservoir:
    return SolarReservoir(
        nodes_1=12,
        nodes_2=10,
        latent_size=2,
        spectral_radius=0.8,
        input_scale=0.4,
        interlayer_scale=0.7,
        density=0.4,
        leak_rate=0.8,
        encoder_steps=2,
        decoder_steps=2,
        seed=seed,
    )


def test_solar_data_are_deterministic_unwrapped_and_physical():
    first = generate_solar_dataset(8, 12, seed=17, sampling_mode="continuous")
    second = generate_solar_dataset(8, 12, seed=17, sampling_mode="continuous")
    assert torch.equal(first.observation, second.observation)
    assert torch.equal(first.target, second.target)
    assert torch.equal(first.heliocentric, second.heliocentric)
    assert first.observation.shape == (8, 2)
    assert first.target.shape == first.heliocentric.shape == (8, 12, 2)
    increments = first.heliocentric[:, 1:] - first.heliocentric[:, :-1]
    expected = torch.tensor(
        [2 * np.pi * 7 / EARTH_PERIOD_DAYS, 2 * np.pi * 7 / MARS_PERIOD_DAYS]
    )
    assert torch.allclose(increments, expected, atol=1e-6, rtol=1e-6)
    assert torch.max(torch.abs(torch.diff(first.target[..., 1], dim=1))) < np.pi


def test_solar_splits_are_independent():
    splits = make_solar_splits(10, 9, 8, 6, data_seed=23)
    assert not torch.equal(splits["train"].observation[:8], splits["test"].observation)


def test_reservoir_has_frozen_matrices_and_additive_latent_dynamics():
    model = reservoir()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert set(parameters) == {"W", "b", "latent_delta", "W_out", "c"}
    for name in model.fixed_matrix_names:
        assert name not in parameters
        assert name in buffers
        assert buffers[name].requires_grad is False
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    prediction, latents = model.predict_with_latents(observation, horizon=5)
    assert prediction.shape == (2, 5, 2)
    assert latents.shape == (2, 5, 2)
    expected_delta = model.latent_delta.detach().expand_as(latents[:, 1:])
    assert torch.allclose(latents[:, 1:] - latents[:, :-1], expected_delta)


def test_solar_forecast_backpropagates_through_frozen_second_reservoir():
    model = reservoir()
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    prediction, _, penalty = model.training_forward(observation, horizon=5)
    (prediction[:, -1].square().mean() + 0.01 * penalty).backward()
    assert model.W.grad is not None
    assert torch.count_nonzero(model.W.grad).item() > 0
    assert model.latent_delta.grad is not None
    for name in model.fixed_matrix_names:
        assert getattr(model, name).grad is None


def test_scinet_reference_is_deterministic_at_evaluation():
    model = SolarSciNet(latent_size=2, hidden_size=12, seed=4)
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    first, first_latent = model.predict_with_latents(observation, horizon=4)
    second, second_latent = model.predict_with_latents(observation, horizon=4)
    assert torch.equal(first, second)
    assert torch.equal(first_latent, second_latent)
    assert torch.allclose(
        first_latent[:, 1:] - first_latent[:, :-1],
        model.latent_delta.detach().expand_as(first_latent[:, 1:]),
    )


def test_solar_smoke_run_writes_analysis_artifacts(tmp_path):
    output = tmp_path / "solar-smoke"
    config = SolarExperimentConfig(
        output_dir=str(output),
        seeds=(0,),
        models=("reservoir", "scinet"),
        train_samples=20,
        validation_samples=10,
        test_samples=10,
        series_length=6,
        nodes_1=12,
        nodes_2=10,
        latent_size=2,
        encoder_steps=2,
        decoder_steps=2,
        scinet_hidden_size=12,
        density=0.4,
        phase_steps=(1,),
        phase_batch_sizes=(8,),
        phase_learning_rates=(1e-3,),
        phase_betas=(0.001,),
        phase_horizons=(6,),
        full_dataset_epochs=True,
        validation_interval=1,
        validation_subset=8,
        evaluation_batch_size=8,
        analysis_grid_size=5,
        device="cpu",
    )
    rows, summary = run_solar_experiment(config)
    assert len(rows) == len(summary) == 2
    assert all(row["optimization_steps"] == 2 for row in rows)
    for model_name in config.models:
        run_dir = output / model_name / "seed_0"
        for filename in (
            "checkpoint.pt",
            "history.json",
            "metrics.json",
            "predictions.npz",
            "latent_surface.npz",
            "predictions.png",
            "latent_surfaces.png",
            "training.png",
        ):
            assert (run_dir / filename).is_file()
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert np.isfinite(metrics["test_relative_rmse_2pi"])
        assert np.isfinite(metrics["heliocentric_to_latent_r2"])
    assert (output / "config.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "summary.csv").is_file()
