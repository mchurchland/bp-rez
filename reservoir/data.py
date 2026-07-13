"""Deterministic NARMA10 data generation."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NARMASplit:
    """A single independently generated input/target sequence."""

    u: torch.Tensor
    y: torch.Tensor


def generate_narma10(length: int, seed: int) -> NARMASplit:
    """Generate the standard NARMA10 benchmark.

    The input is uniform on [0, 0.5]. Targets obey

        y[t+1] = 0.3 y[t] + 0.05 y[t] sum(y[t-i], i=0..9)
                 + 1.5 u[t-9] u[t] + 0.1.

    Returned ``u[t]`` is paired with ``y[t+1]``. The first ten points are
    transient values and should normally be excluded with ``washout >= 10``.
    """

    if length <= 10:
        raise ValueError("NARMA10 sequence length must be greater than 10")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=length).astype(np.float64)
    y = np.zeros(length + 1, dtype=np.float64)
    for t in range(length):
        delayed_product = u[t - 9] * u[t] if t >= 9 else 0.0
        history = y[max(0, t - 9) : t + 1].sum()
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * history
            + 1.5 * delayed_product
            + 0.1
        )
    return NARMASplit(
        u=torch.from_numpy(u.astype(np.float32)).unsqueeze(-1),
        y=torch.from_numpy(y[1:].astype(np.float32)).unsqueeze(-1),
    )


def make_narma10_splits(
    train_length: int,
    val_length: int,
    test_length: int,
    data_seed: int,
) -> dict[str, NARMASplit]:
    """Create deterministic, independent train/validation/test sequences."""

    seed_sequence = np.random.SeedSequence(data_seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(3)]
    return {
        "train": generate_narma10(train_length, child_seeds[0]),
        "validation": generate_narma10(val_length, child_seeds[1]),
        "test": generate_narma10(test_length, child_seeds[2]),
    }
