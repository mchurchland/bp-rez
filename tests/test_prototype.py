import json

import numpy as np
import torch

from reservoir.data import make_narma10_splits
from reservoir.experiment import ExperimentConfig, run_experiment
from reservoir.models import (
    MODEL_NAMES,
    build_model,
    count_trainable_parameters,
    make_recurrent_matrix,
)


def model(name: str, seed: int = 3):
    return build_model(
        name,
        nodes_1=150,
        nodes_2=150,
        latent_size=10,
        spectral_radius=0.9,
        input_scale=0.5,
        interlayer_scale=1.0,
        density=0.1,
        leak_rate=0.8,
        seed=seed,
    )


def test_narma_splits_are_deterministic_and_independent():
    first = make_narma10_splits(30, 25, 20, data_seed=17)
    second = make_narma10_splits(30, 25, 20, data_seed=17)
    for split in first:
        assert torch.equal(first[split].u, second[split].u)
        assert torch.equal(first[split].y, second[split].y)
    assert not torch.equal(first["train"].u[:20], first["test"].u)


def test_recurrent_matrix_has_requested_spectral_radius():
    generator = torch.Generator().manual_seed(4)
    matrix = make_recurrent_matrix(40, 0.83, 0.2, generator)
    observed = np.abs(np.linalg.eigvals(matrix.numpy())).max()
    assert np.isclose(observed, 0.83, rtol=2e-5, atol=2e-5)


def test_parameter_counts_and_frozen_matrices():
    expected = {
        "single_esn": 151,
        "deep_esn": 151,
        "proposed_nonlinear": 1661,
        "proposed_linear": 1661,
        "large_esn": 301,
    }
    for name in MODEL_NAMES:
        instance = model(name)
        assert count_trainable_parameters(instance) == expected[name]
        parameters = dict(instance.named_parameters())
        buffers = dict(instance.named_buffers())
        for fixed_name in instance.fixed_matrix_names:
            assert fixed_name not in parameters
            assert fixed_name in buffers
            assert buffers[fixed_name].requires_grad is False


def test_bptt_trains_intermediate_through_frozen_second_reservoir():
    instance = model("proposed_nonlinear")
    u = torch.linspace(0.05, 0.5, 8).unsqueeze(-1)
    x1 = torch.zeros(instance.nodes_1)
    x2 = torch.zeros(instance.nodes_2)
    for step, value in enumerate(u):
        x1 = instance._update(x1, instance.A1, instance.B1 @ value)
        h = torch.tanh(instance.W @ x1 + instance.b)
        # Only the first h retains a path to W. A final-time loss can therefore
        # reach W only by traversing seven frozen A2 recurrent transitions.
        if step > 0:
            h = h.detach()
        x2 = instance._update(x2, instance.A2, instance.R @ h)
    loss = (instance.W_out @ x2 + instance.c).square().sum()
    loss.backward()
    assert set(dict(instance.named_parameters())) == {"W", "b", "W_out", "c"}
    assert instance.W.grad is not None
    assert torch.count_nonzero(instance.W.grad).item() > 0
    assert instance.b.grad is not None
    assert instance.W_out.grad is not None
    for fixed_name in instance.fixed_matrix_names:
        assert getattr(instance, fixed_name).grad is None

    # When an upstream quantity requests a gradient, the ordinary forward path
    # remains differentiable through A1, W/R, and A2 despite all fixed matrices
    # being non-trainable buffers.
    end_to_end = model("proposed_nonlinear")
    signal = torch.linspace(0.05, 0.5, 8).unsqueeze(-1).requires_grad_()
    end_to_end(signal)[-1].sum().backward()
    assert signal.grad is not None
    assert torch.count_nonzero(signal.grad).item() > 0


def test_smoke_run_writes_reproducibility_artifacts(tmp_path):
    output = tmp_path / "smoke"
    config = ExperimentConfig(
        output_dir=str(output),
        seeds=(0,),
        models=("proposed_nonlinear",),
        train_length=24,
        val_length=20,
        test_length=20,
        washout=10,
        max_epochs=2,
        patience=2,
        device="cpu",
    )
    rows, summary = run_experiment(config)
    assert len(rows) == len(summary) == 1
    run_dir = output / "proposed_nonlinear" / "seed_0"
    for filename in (
        "checkpoint.pt",
        "history.json",
        "metrics.json",
        "predictions.npz",
        "predictions.png",
        "training.png",
    ):
        assert (run_dir / filename).is_file()
    assert (output / "config.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "summary.png").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["trainable_parameters"] == 1661
    assert np.isfinite(metrics["test_mse"])
