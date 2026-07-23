from pathlib import Path

from run_solar_grid_search import (
    GridPoint,
    build_config,
    build_grid,
    validation_selection_score,
)


def test_grid_has_162_unique_controlled_candidates():
    grid = build_grid()
    assert len(grid) == 162
    assert len({point.slug for point in grid}) == len(grid)
    assert {point.warmup_steps for point in grid} == {5, 20, 40}
    assert {point.steps_per_week for point in grid} == {1, 3, 5}
    assert {point.interlayer_scale for point in grid} == {1.0, 2.0, 4.0}
    assert {point.beta_mode for point in grid} == {"paper", "zero"}


def test_grid_config_uses_two_layers_and_15k_updates():
    point = GridPoint(
        warmup_steps=20,
        steps_per_week=3,
        interlayer_scale=2.0,
        velocity_weight=10.0,
        curvature_weight=10.0,
        beta_mode="zero",
    )
    config = build_config(
        point,
        Path("trial"),
        seed=0,
        data_seed=2026,
        device="cpu",
        smoke=False,
    )
    assert config.reservoir_layers == 2
    assert config.nodes_1 == config.nodes_2 == 150
    assert sum(config.phase_steps) == 15_000
    assert config.second_reservoir_warmup_steps == 20
    assert config.second_reservoir_steps == 3
    assert config.phase_betas == (0.0,) * 5
    assert config.mars_velocity_loss_weight == 10.0
    assert config.mars_curvature_loss_weight == 10.0


def test_grid_selection_score_uses_only_validation_mars_metrics():
    metrics = {
        "validation_mars_mse": 0.2,
        "validation_mars_velocity_mse": 0.03,
        "validation_mars_curvature_mse": 0.004,
        "test_mars_mse": 999.0,
    }
    assert validation_selection_score(metrics) == 0.54
