"""Explicit, stable and reproducible NARMA benchmark definitions.

There is no single canonical equation shared by every NARMA order.  This module
therefore uses a named, cited task registry instead of silently replacing the
``10`` in NARMA-10.  The selected definitions are the commonly reported stable
variants summarized by Wringe, Trefzer and Stepney (2024). Benchmark streams
apply a separately named, model-blind stable-trajectory conditioning policy;
they are not claimed to be unconditional replications of the cited inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch


Activation = Literal["identity", "tanh"]


@dataclass(frozen=True)
class NARMATask:
    """Complete scalar recurrence protocol for one named NARMA task."""

    name: str
    order: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    input_low: float
    input_high: float
    outer_activation: Activation
    citation: str
    citation_url: str

    @property
    def equation(self) -> str:
        core = (
            f"{self.alpha:g} y[t] + {self.beta:g} y[t] "
            f"sum(y[t-i], i=0..{self.order - 1}) + "
            f"{self.gamma:g} u[t-{self.order - 1}] u[t] + "
            f"{self.delta:g}"
        )
        if self.outer_activation == "tanh":
            return f"y[t+1] = tanh({core})"
        return f"y[t+1] = {core}"

    @property
    def benchmark_variant_name(self) -> str:
        return f"{self.name}_stable_trajectory_conditional"

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "equation": self.equation,
            "benchmark_variant_name": self.benchmark_variant_name,
        }


TASKS: dict[int, NARMATask] = {
    5: NARMATask(
        name="narma5_fujii_nakajima",
        order=5,
        alpha=0.3,
        beta=0.05,
        gamma=1.5,
        delta=0.1,
        input_low=0.0,
        input_high=0.2,
        outer_activation="identity",
        citation="Fujii and Nakajima (2017), Eq. 18 and Appendix A.2",
        citation_url="https://doi.org/10.1103/PhysRevApplied.8.024030",
    ),
    10: NARMATask(
        name="narma10_atiya_parlos",
        order=10,
        alpha=0.3,
        beta=0.05,
        gamma=1.5,
        delta=0.1,
        input_low=0.0,
        input_high=0.5,
        outer_activation="identity",
        citation="Atiya and Parlos (2000), Eq. 86",
        citation_url="https://doi.org/10.1109/72.846741",
    ),
    20: NARMATask(
        name="narma20_rodan_tino",
        order=20,
        alpha=0.3,
        beta=0.05,
        gamma=1.5,
        delta=0.01,
        input_low=0.0,
        input_high=0.5,
        outer_activation="tanh",
        citation="Rodan and Tino (2011), Eq. 6",
        citation_url="https://doi.org/10.1109/TNN.2010.2089641",
    ),
    30: NARMATask(
        name="narma30_schrauwen",
        order=30,
        alpha=0.2,
        beta=0.04,
        gamma=1.5,
        delta=0.001,
        input_low=0.0,
        input_high=0.5,
        outer_activation="identity",
        citation="Schrauwen et al. (2008), p. 1164",
        citation_url="https://doi.org/10.1016/j.neucom.2007.12.020",
    ),
}


@dataclass(frozen=True)
class NARMASplit:
    """One independently initialized input/target sequence and provenance."""

    u: torch.Tensor
    y: torch.Tensor
    task: NARMATask
    stream: str
    requested_seed: int
    realized_seed: int
    generation_attempt: int
    rejected_seeds: tuple[int, ...] = ()
    acceptance_horizon: int | None = None

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.u.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.y.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def manifest(self) -> dict[str, object]:
        acceptance_horizon = self.acceptance_horizon or len(self.u)
        return {
            "stream": self.stream,
            "length": len(self.u),
            "proposal_input_distribution": (
                f"iid Uniform({self.task.input_low:g}, "
                f"{self.task.input_high:g}) from NumPy PCG64"
            ),
            "accepted_input_distribution": (
                "proposal distribution conditioned on a finite target with "
                "|y[t]| <= 1e6 over the acceptance horizon"
            ),
            "acceptance_horizon": acceptance_horizon,
            "requested_seed": self.requested_seed,
            "realized_seed": self.realized_seed,
            "generation_attempt": self.generation_attempt,
            "rejected_seeds": list(self.rejected_seeds),
            "sha256_u_and_y": self.digest(),
        }


def _derive_seed(
    base_seed: int, order: int, pair_id: int, stream: int, attempt: int
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, order, pair_id, stream, attempt, 0x4E41524D]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def generate_narma(
    task: NARMATask,
    length: int,
    seed: int,
    *,
    stream: str = "unspecified",
    requested_seed: int | None = None,
    generation_attempt: int = 0,
    rejected_seeds: tuple[int, ...] = (),
) -> NARMASplit:
    """Generate one NARMA sequence in float64, then store it as float32.

    ``u[t]`` is paired with ``y[t+1]``.  Initial targets and unavailable delayed
    inputs are zero.  Non-finite or extreme targets raise instead of being
    clipped, preserving an auditable definition of data-generation failure.
    """

    if length <= task.order:
        raise ValueError(f"sequence length must exceed NARMA order {task.order}")
    rng = np.random.default_rng(seed)
    u = rng.uniform(task.input_low, task.input_high, size=length).astype(np.float64)
    y = np.zeros(length + 1, dtype=np.float64)
    for t in range(length):
        delayed = u[t - task.order + 1] * u[t] if t >= task.order - 1 else 0.0
        history = y[max(0, t - task.order + 1) : t + 1].sum()
        value = (
            task.alpha * y[t]
            + task.beta * y[t] * history
            + task.gamma * delayed
            + task.delta
        )
        if task.outer_activation == "tanh":
            value = np.tanh(value)
        y[t + 1] = value
        if not np.isfinite(value) or abs(value) > 1.0e6:
            raise FloatingPointError(
                f"{task.name} diverged at t={t} for seed={seed}: y={value}"
            )
    return NARMASplit(
        u=torch.from_numpy(u.astype(np.float32)).unsqueeze(-1),
        y=torch.from_numpy(y[1:].astype(np.float32)).unsqueeze(-1),
        task=task,
        stream=stream,
        requested_seed=seed if requested_seed is None else requested_seed,
        realized_seed=seed,
        generation_attempt=generation_attempt,
        rejected_seeds=rejected_seeds,
        acceptance_horizon=length,
    )


def _generate_stable_stream(
    task: NARMATask,
    length: int,
    *,
    base_seed: int,
    pair_id: int,
    stream_name: str,
    stream_code: int,
    max_attempts: int = 100,
) -> NARMASplit:
    """Apply the benchmark's deterministic, model-blind rejection sampler.

    Every proposal is i.i.d. uniform according to the task definition.  A
    proposal is accepted only when its target remains finite and bounded by
    ``1e6`` over the complete requested horizon.  This makes the accepted
    distribution conditional rather than exactly uniform; the manifest records
    the proposal seeds and acceptance horizon so that conditioning is explicit.
    """

    rejected: list[int] = []
    requested = _derive_seed(base_seed, task.order, pair_id, stream_code, 0)
    for attempt in range(max_attempts):
        realized = _derive_seed(base_seed, task.order, pair_id, stream_code, attempt)
        try:
            return generate_narma(
                task,
                length,
                realized,
                stream=stream_name,
                requested_seed=requested,
                generation_attempt=attempt,
                rejected_seeds=tuple(rejected),
            )
        except FloatingPointError:
            rejected.append(realized)
    raise RuntimeError(
        f"could not generate stable {task.name} {stream_name} data after "
        f"{max_attempts} deterministic attempts; rejected={rejected}"
    )


def _prefix(split: NARMASplit, length: int) -> NARMASplit:
    """Return a requested prefix while preserving acceptance provenance."""

    if not 0 < length <= len(split.u):
        raise ValueError("prefix length must be within the generated split")
    return NARMASplit(
        u=split.u[:length].clone(),
        y=split.y[:length].clone(),
        task=split.task,
        stream=split.stream,
        requested_seed=split.requested_seed,
        realized_seed=split.realized_seed,
        generation_attempt=split.generation_attempt,
        rejected_seeds=split.rejected_seeds,
        acceptance_horizon=split.acceptance_horizon,
    )


def make_paired_splits(
    order: int,
    *,
    train_length: int,
    long_train_length: int,
    validation_length: int,
    test_length: int,
    base_seed: int,
    pair_id: int,
) -> dict[str, NARMASplit]:
    """Create paired splits with small training data as a long-data prefix.

    Stream seeds do not contain the data-regime name.  Consequently validation
    and final-test tensors are bit-identical between small and long conditions,
    while ``train_length`` selects a prefix from the same accepted training
    stream. All streams use the common acceptance horizon
    ``max(long_train_length, validation_length, test_length)``. Thus the small
    prefix is intentionally conditioned on stability beyond its observed
    portion, but train/validation/test share the same accepted law.
    """

    if order not in TASKS:
        raise ValueError(f"unsupported NARMA order {order}; choose {sorted(TASKS)}")
    if not 0 < train_length <= long_train_length:
        raise ValueError("train_length must be in (0, long_train_length]")
    task = TASKS[order]
    acceptance_horizon = max(long_train_length, validation_length, test_length)
    full_train = _generate_stable_stream(
        task,
        acceptance_horizon,
        base_seed=base_seed,
        pair_id=pair_id,
        stream_name="train",
        stream_code=0x54524149,
    )
    full_validation = _generate_stable_stream(
        task,
        acceptance_horizon,
        base_seed=base_seed,
        pair_id=pair_id,
        stream_name="validation",
        stream_code=0x56414C49,
    )
    full_test = _generate_stable_stream(
        task,
        acceptance_horizon,
        base_seed=base_seed,
        pair_id=pair_id,
        stream_name="final_test",
        stream_code=0x54455354,
    )
    return {
        "train": _prefix(full_train, train_length),
        "validation": _prefix(full_validation, validation_length),
        "test": _prefix(full_test, test_length),
    }


def make_development_splits(
    order: int,
    *,
    train_length: int,
    long_train_length: int,
    validation_length: int,
    base_seed: int,
    tuning_pair_id: int,
) -> dict[str, NARMASplit]:
    """Create train/validation data without ever constructing a final test set."""

    if order not in TASKS:
        raise ValueError(f"unsupported NARMA order {order}; choose {sorted(TASKS)}")
    if not 0 < train_length <= long_train_length:
        raise ValueError("train_length must be in (0, long_train_length]")
    task = TASKS[order]
    acceptance_horizon = max(long_train_length, validation_length)
    full_train = _generate_stable_stream(
        task,
        acceptance_horizon,
        base_seed=base_seed,
        pair_id=tuning_pair_id,
        stream_name="development_train",
        stream_code=0x44455654,
    )
    full_validation = _generate_stable_stream(
        task,
        acceptance_horizon,
        base_seed=base_seed,
        pair_id=tuning_pair_id,
        stream_name="development_validation",
        stream_code=0x44455656,
    )
    return {
        "train": _prefix(full_train, train_length),
        "validation": _prefix(full_validation, validation_length),
    }
