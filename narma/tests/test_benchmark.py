import json

import numpy as np
import torch

from narma.benchmark import (
    SuiteConfig,
    _candidate_configs,
    _costs,
    aggregate_results,
    lock_selected_configs,
    run_final_pair,
    run_tuning_condition,
)
from narma.data import (
    TASKS,
    generate_narma,
    make_development_splits,
    make_paired_splits,
)
from narma.models import MODEL_NAMES, ReservoirConfig, build_model
from narma.ridge import tune_and_refit_ridge
from narma.statistics import (
    exact_sign_test_pvalue,
    paired_difference_in_differences,
    paired_regime_comparisons,
)
from narma.training import OptimizerConfig
from narma.run_benchmark import _final_index, _tune_index, _tune_subset_index


def test_every_registered_recurrence_is_explicit_and_finite():
    for order, task in TASKS.items():
        split = generate_narma(task, 500, seed=17)
        expected_first = task.delta
        if task.outer_activation == "tanh":
            expected_first = np.tanh(expected_first)
        assert np.isclose(split.y[0].item(), expected_first)
        assert torch.isfinite(split.y).all()
        assert f"t-{order - 1}" in task.equation


def test_small_data_are_prefix_and_validation_test_are_paired():
    small = make_paired_splits(
        10,
        train_length=80,
        long_train_length=120,
        validation_length=70,
        test_length=75,
        base_seed=91,
        pair_id=3,
    )
    long = make_paired_splits(
        10,
        train_length=120,
        long_train_length=120,
        validation_length=70,
        test_length=75,
        base_seed=91,
        pair_id=3,
    )
    assert torch.equal(small["train"].u, long["train"].u[:80])
    assert torch.equal(small["train"].y, long["train"].y[:80])
    assert small["validation"].digest() == long["validation"].digest()
    assert small["test"].digest() == long["test"].digest()
    assert {split.acceptance_horizon for split in small.values()} == {120}


def test_development_api_does_not_construct_test_data():
    splits = make_development_splits(
        5,
        train_length=50,
        long_train_length=60,
        validation_length=45,
        base_seed=82,
        tuning_pair_id=10_000,
    )
    assert set(splits) == {"train", "validation"}


def test_named_random_streams_pair_relevant_reservoirs():
    config = ReservoirConfig(nodes_1=12, nodes_2=11, latent_size=3)
    deep = build_model("deep_esn_ridge", config, seed=4)
    random = build_model("random_bottleneck", config, seed=4)
    learned = build_model("learned_linear", config, seed=4)
    assert torch.equal(deep.A1, random.A1)
    assert torch.equal(deep.A1, learned.A1)
    assert torch.equal(deep.A2, random.A2)
    assert torch.equal(random.R, learned.R)


def test_primary_and_gru_have_parameter_matched_trainable_coefficients():
    config = ReservoirConfig()
    proposed = build_model("learned_linear", config, seed=0)
    gru = build_model("gru", config, seed=0)
    assert _costs(proposed)["gradient_trained_parameters"] == 1_661
    assert _costs(gru)["gradient_trained_parameters"] == 1_673
    assert _costs(proposed)["trainable_parameters"] == 1_661
    assert _costs(gru)["trainable_parameters"] == 1_673


def test_default_slurm_array_endpoints_cover_the_complete_matrix():
    suite = SuiteConfig()
    fixed = MODEL_NAMES[:5]
    gradient = MODEL_NAMES[5:]
    assert _tune_index(suite, 0) == (5, "small", "esn_ridge")
    assert _tune_index(suite, 63) == (30, "long", "gru")
    assert _tune_subset_index(suite, 0, fixed) == (5, "small", "esn_ridge")
    assert _tune_subset_index(suite, 39, fixed) == (30, "long", "pca_bottleneck")
    assert _tune_subset_index(suite, 0, gradient) == (
        5,
        "small",
        "learned_linear",
    )
    assert _tune_subset_index(suite, 23, gradient) == (30, "long", "gru")
    assert _final_index(suite, 0) == (5, "small", 0)
    assert _final_index(suite, 79) == (30, "long", 9)


def test_default_tuning_candidates_are_unique_and_complete():
    suite = SuiteConfig()
    for model_name in MODEL_NAMES:
        candidates = _candidate_configs(
            suite,
            order=10,
            regime="small",
            model_name=model_name,
        )
        assert len(candidates) == suite.search_trials
        serialized = {
            (
                json.dumps(reservoir.to_dict(), sort_keys=True),
                json.dumps(optimizer.to_dict(), sort_keys=True),
            )
            for reservoir, optimizer in candidates
        }
        assert len(serialized) == suite.search_trials


def test_all_models_expose_head_independent_features():
    config = ReservoirConfig(
        nodes_1=10,
        nodes_2=9,
        latent_size=3,
        density_1=0.4,
        density_2=0.4,
        gru_hidden_size=4,
    )
    u = torch.linspace(0.0, 0.2, 30).unsqueeze(-1)
    for name in MODEL_NAMES:
        model = build_model(name, config, seed=2)
        model.prepare(u, washout=5)
        features = model.features(u)
        assert features.shape == (30, model.feature_size)
        assert model(u).shape == (30, 1)


def test_ridge_tunes_without_test_and_refits_an_unpenalized_intercept():
    x_train = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    y_train = 2.5 * x_train - 3.0
    x_validation = torch.arange(12, 18, dtype=torch.float32).reshape(-1, 1)
    y_validation = 2.5 * x_validation - 3.0
    fit = tune_and_refit_ridge(
        x_train,
        y_train,
        x_validation,
        y_validation,
        (0.0, 1e-3, 1.0),
    )
    assert fit.alpha == 0.0
    assert torch.allclose(fit.weight, torch.tensor([[2.5]]), atol=1e-5)
    assert torch.allclose(fit.bias, torch.tensor([-3.0]), atol=1e-5)


def test_training_regime_comparison_is_paired_by_seed():
    rows = []
    for pair_id in range(10):
        rows.extend(
            (
                {
                    "order": 10,
                    "regime": "small",
                    "model": "esn_ridge",
                    "pair_id": pair_id,
                    "test_nrmse": 0.5 + pair_id * 0.001,
                },
                {
                    "order": 10,
                    "regime": "long",
                    "model": "esn_ridge",
                    "pair_id": pair_id,
                    "test_nrmse": 0.4 + pair_id * 0.001,
                },
            )
        )
    comparison = paired_regime_comparisons(rows, bootstrap_samples=100)[0]
    assert comparison["pairs"] == 10
    assert np.isclose(comparison["mean_paired_nrmse_difference"], -0.1)
    assert comparison["long_regime_wins"] == 10
    assert comparison["permutation_p_raw"] == 2 / 1024
    assert comparison["sign_test_p_raw"] == 2 / 1024
    assert "permutation_p_holm" in comparison


def test_difference_in_differences_tracks_relative_data_benefit():
    rows = []
    for pair_id in range(10):
        for model, small, long in (
            ("learned_linear", 0.5, 0.3),
            ("deep_esn_ridge", 0.6, 0.5),
        ):
            rows.extend(
                (
                    {
                        "order": 10,
                        "regime": "small",
                        "model": model,
                        "pair_id": pair_id,
                        "test_nrmse": small + pair_id * 0.001,
                    },
                    {
                        "order": 10,
                        "regime": "long",
                        "model": model,
                        "pair_id": pair_id,
                        "test_nrmse": long + pair_id * 0.001,
                    },
                )
            )
    comparison = paired_difference_in_differences(rows, bootstrap_samples=100)[0]
    assert comparison["pairs"] == 10
    assert np.isclose(comparison["mean_paired_difference_in_differences"], -0.1)
    assert comparison["sign_test_p_raw"] == 2 / 1024
    assert np.isnan(exact_sign_test_pvalue(np.zeros(10)))
    assert exact_sign_test_pvalue(-np.ones(10)) == 2 / 1024


def test_tiny_complete_pipeline_writes_locked_paired_reports(tmp_path):
    suite = SuiteConfig(
        output_root=str(tmp_path / "benchmark"),
        orders=(5,),
        regimes=("small",),
        models=MODEL_NAMES,
        final_pair_ids=(0,),
        tuning_pair_ids=(10_000,),
        small_train_length=50,
        long_train_length=60,
        validation_length=45,
        test_length=45,
        washout=10,
        search_trials=1,
        bootstrap_samples=100,
        max_epoch_cap_fraction=1.0,
        device="cpu",
        reservoir=ReservoirConfig(
            nodes_1=8,
            nodes_2=8,
            latent_size=3,
            density_1=0.5,
            density_2=0.5,
            gru_hidden_size=4,
        ),
        optimizer=OptimizerConfig(
            max_epochs=2,
            early_stopping_patience=2,
            scheduler_patience=1,
        ),
    )
    for model_name in suite.models:
        selected = run_tuning_condition(
            suite, order=5, regime="small", model_name=model_name
        )
        assert selected["test_data_constructed"] is False
    locked = lock_selected_configs(suite)
    assert len(locked["locked_config_hash"]) == 64
    rows = run_final_pair(suite, order=5, regime="small", pair_id=0)
    assert len(rows) == len(MODEL_NAMES)
    raw, summary = aggregate_results(suite)
    assert len(raw) == len(MODEL_NAMES)
    assert len(summary) == len(MODEL_NAMES)
    root = tmp_path / "benchmark"
    for filename in (
        "locked_configs.json",
        "protocol.json",
        "PROTOCOL.md",
        "metrics.csv",
        "summary.csv",
        "paired_comparisons.csv",
        "paired_difference_in_differences.json",
        "convergence.json",
        "summary.png",
        "completeness.json",
    ):
        assert (root / filename).is_file()
    completeness = json.loads((root / "completeness.json").read_text())
    assert completeness["observed_rows"] == completeness["expected_rows"]
    comparisons = json.loads((root / "paired_comparisons.json").read_text())
    assert len(comparisons) == len(MODEL_NAMES) - 1
    assert all("permutation_p_holm_global" in row for row in comparisons)
    convergence = json.loads((root / "convergence.json").read_text())
    assert convergence["accepted"] is True
