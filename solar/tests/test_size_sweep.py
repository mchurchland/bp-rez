import json
from pathlib import Path
from unittest.mock import patch

from solar.run_size_sweep import (
    DEFAULT_RESERVOIR_SIZES,
    build_config,
    fixed_matrix_scalars,
    run_size_sweep,
)


def test_default_size_grid_covers_small_to_large_reservoirs():
    assert DEFAULT_RESERVOIR_SIZES[0] == 10
    assert DEFAULT_RESERVOIR_SIZES[-1] == 300
    assert 150 in DEFAULT_RESERVOIR_SIZES
    assert list(DEFAULT_RESERVOIR_SIZES) == sorted(set(DEFAULT_RESERVOIR_SIZES))


def test_size_trial_has_two_matched_reservoirs_and_one_latent_bottleneck():
    config = build_config(
        75,
        Path("trial"),
        seeds=(0, 1, 2),
        data_seed=2026,
        device="cpu",
        smoke=False,
    )
    assert config.nodes_1 == config.nodes_2 == 75
    assert config.reservoir_layers == 2
    assert config.latent_size == 2
    assert sum(config.phase_steps) == 15_000
    assert config.second_reservoir_warmup_steps == 20
    assert config.second_reservoir_steps == 3
    assert config.interlayer_scale == 2.0
    assert config.mars_velocity_loss_weight == 1.0
    assert config.mars_curvature_loss_weight == 1.0
    assert fixed_matrix_scalars(75) == 2 * 75**2 + 4 * 75


def test_size_sweep_smoke_run_is_resumable(tmp_path):
    output = tmp_path / "size-sweep"
    records, summaries = run_size_sweep(
        output,
        sizes=(10,),
        seeds=(0,),
        data_seed=2026,
        device="cpu",
        smoke=True,
    )
    assert len(records) == len(summaries) == 1
    assert records[0]["reservoir_size"] == 10
    assert records[0]["seed"] == 0
    assert summaries[0]["seeds"] == 1

    for filename in (
        "size_sweep_definition.json",
        "size_sweep.json",
        "size_sweep.csv",
        "size_sweep_summary.json",
        "size_sweep_summary.csv",
        "best_validation_size.json",
        "size_sweep.png",
    ):
        assert (output / filename).is_file()

    run_dir = output / "runs" / "size_010"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["nodes_1"] == config["nodes_2"] == 10
    assert config["reservoir_layers"] == 2
    assert config["seeds"] == [0]
    assert (run_dir / "reservoir" / "seed_0" / "checkpoint.pt").is_file()

    with patch(
        "solar.run_size_sweep.run_solar_experiment",
        side_effect=AssertionError("completed size should have been skipped"),
    ):
        resumed_records, resumed_summaries = run_size_sweep(
            output,
            sizes=(10,),
            seeds=(0,),
            data_seed=2026,
            device="cpu",
            smoke=True,
        )
    assert resumed_records == records
    assert resumed_summaries == summaries
