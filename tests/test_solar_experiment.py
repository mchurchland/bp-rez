import json

import numpy as np
import torch

from reservoir.solar_data import (
    EARTH_PERIOD_DAYS,
    MARS_PERIOD_DAYS,
    generate_solar_dataset,
    make_solar_splits,
)
from reservoir.solar_experiment import (
    SolarExperimentConfig,
    _mars_dynamics_losses,
    run_solar_experiment,
)
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
        second_reservoir_warmup_steps=4,
        second_reservoir_steps=2,
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


def test_reservoir_has_registered_matrices_and_additive_latent_dynamics():
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


def test_second_reservoir_carries_state_across_forecast_weeks():
    model = reservoir()
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    prediction, latents = model.predict_with_latents(observation, horizon=5)

    state = torch.zeros((len(observation), model.nodes_2))
    latent_sequence = latents.unbind(dim=1)
    initial_drive = latent_sequence[0] @ model.R.T
    for _ in range(model.second_reservoir_warmup_steps):
        state = model._update(state, model.A2, initial_drive)
    expected = [state @ model.W_out.T + model.c]
    for latent in latent_sequence[1:]:
        drive = latent @ model.R.T
        for _ in range(model.second_reservoir_steps):
            state = model._update(state, model.A2, drive)
        expected.append(state @ model.W_out.T + model.c)
    assert torch.allclose(prediction, torch.stack(expected, dim=1))


def test_solar_forecast_backpropagates_through_second_reservoir():
    model = reservoir()
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    prediction, _, penalty = model.training_forward(observation, horizon=5)
    (prediction[:, -1].square().mean() + 0.01 * penalty).backward()
    assert model.W.grad is not None
    assert torch.count_nonzero(model.W.grad).item() > 0
    assert model.latent_delta.grad is not None
    for name in model.fixed_matrix_names:
        assert getattr(model, name).grad is None


def test_ten_reservoir_layers_have_nine_two_neuron_bottlenecks():
    model = SolarReservoir(
        nodes_1=6,
        nodes_2=6,
        reservoir_layers=10,
        latent_size=2,
        spectral_radius=0.8,
        input_scale=0.4,
        interlayer_scale=0.7,
        density=0.5,
        leak_rate=0.8,
        encoder_steps=2,
        second_reservoir_warmup_steps=2,
        second_reservoir_steps=2,
        seed=5,
    )
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    prediction, all_latents = model.predict_with_all_latents(observation, horizon=4)
    assert prediction.shape == (2, 4, 2)
    assert all_latents.shape == (2, 4, 9, 2)
    assert len(model.intermediate_weights) == 8
    assert len(model.fixed_matrix_names) == 20

    _, primary_latents = model.predict_with_latents(observation, horizon=4)
    expected_delta = model.latent_delta.detach().expand_as(primary_latents[:, 1:])
    assert torch.allclose(
        primary_latents[:, 1:] - primary_latents[:, :-1], expected_delta
    )


def test_mars_dynamics_losses_match_first_and_second_differences():
    target = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [0.0, 7.0]]])
    prediction = torch.tensor(
        [[[5.0, 1.0], [5.0, 3.0], [5.0, 5.0], [5.0, 7.0]]],
        requires_grad=True,
    )
    velocity_loss, curvature_loss = _mars_dynamics_losses(prediction, target)
    assert torch.isclose(velocity_loss, torch.tensor(2.0 / 3.0))
    assert torch.isclose(curvature_loss, torch.tensor(1.0))
    (velocity_loss + curvature_loss).backward()
    assert prediction.grad is not None


def test_mars_dynamics_losses_are_safe_for_short_horizons():
    one_step = torch.zeros((2, 1, 2))
    velocity_loss, curvature_loss = _mars_dynamics_losses(one_step, one_step)
    assert velocity_loss.item() == curvature_loss.item() == 0.0

    two_steps = torch.zeros((2, 2, 2))
    velocity_loss, curvature_loss = _mars_dynamics_losses(two_steps, two_steps)
    assert velocity_loss.item() == curvature_loss.item() == 0.0


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
        atol=1e-6,
    )


def test_scinet_matches_released_unclipped_sigma_and_unused_euler_weight():
    model = SolarSciNet(latent_size=2, hidden_size=12, seed=4)
    observation = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.zero_()
        model.encoder[-1].bias.copy_(torch.tensor([0.0, 0.0, -7.0, 2.0]))

    log_sigma = model.latent_log_sigma(observation)
    assert log_sigma is not None
    assert torch.equal(log_sigma, torch.tensor([[-7.0, 2.0], [-7.0, 2.0]]))

    prediction_before, _ = model.predict_with_latents(observation, horizon=4)
    regularizer_before = model.evolution_l2_loss().detach().clone()
    with torch.no_grad():
        model.euler_weight.add_(3.0)
    prediction_after, _ = model.predict_with_latents(observation, horizon=4)
    assert torch.equal(prediction_before, prediction_after)
    assert model.evolution_l2_loss() > regularizer_before


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
        second_reservoir_warmup_steps=4,
        second_reservoir_steps=2,
        scinet_hidden_size=12,
        density=0.4,
        phase_steps=(1,),
        phase_batch_sizes=(8,),
        phase_learning_rates=(1e-3,),
        phase_betas=(0.001,),
        phase_horizons=(6,),
        full_dataset_epochs=True,
        training_log_interval=1,
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
        assert np.isfinite(metrics["validation_mars_mse"])
        assert np.isfinite(metrics["validation_mars_velocity_mse"])
        assert np.isfinite(metrics["validation_mars_curvature_mse"])
        assert np.isfinite(metrics["heliocentric_to_latent_r2"])
        history = json.loads((run_dir / "history.json").read_text())
        assert len(history["step"]) == 2
        assert len(history["reconstruction_loss"]) == 2
        assert len(history["latent_delta"]) == 1
        if model_name == "scinet":
            assert len(history["kl_loss"]) == 2
            assert len(history["evolution_l2_loss"]) == 2
            assert len(history["latent_sigma_mean"]) == 1
            assert np.isfinite(metrics["final_latent_sigma_mean"])
        else:
            assert len(history["representation_loss"]) == 2
            assert len(history["mars_velocity_loss"]) == 2
            assert len(history["mars_curvature_loss"]) == 2
    assert (output / "config.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "summary.csv").is_file()
