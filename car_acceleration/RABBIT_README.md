# Forecasting rabbit and wolf populations

This experiment observes paired histories of rabbit and wolf populations and
autonomously forecasts both future populations. The Lotka–Volterra parameters
are hidden from the model and retained only for latent-state diagnostics.

The generated trajectories follow

```text
dx/dt = alpha*x - beta*x*y
dy/dt = delta*x*y - gamma*y
```

## Easier parameter ranges

The trajectory-specific rates now vary over narrow linear ranges:

| Parameter | Range |
| --- | ---: |
| `alpha` | 0.8–1.2 |
| `beta` | 0.08–0.12 |
| `gamma` | 0.8–1.2 |
| `delta` | 0.08–0.12 |

Previously, `alpha` and `gamma` ranged from approximately 0.1 to 3.16, while
`beta` and `delta` ranged from 0.01 to 1. The narrower ranges reduce variation
in oscillation frequency, phase, and equilibrium populations.

Because the rate labels now have much less variance, their diagnostic R² values
will be more sensitive to small errors and should not be compared directly with
scores from the old broad-range dataset.

## Model

```text
200 paired rabbit-wolf observations
          |
          v
fixed 1,000-node reservoir
          |
          v
states tapped at observations 40, 80, 120, 160, 200
          |
          v
concatenated 5,000D reservoir features
          |
          v
learned 8D latent
          |
          v
quadratic autonomous latent recurrence
          |
          v
linear rabbit-wolf population readout
```

Both population channels receive independent `log1p` transforms and
training-set standardization. The model is trained against future standardized
log-rabbit and log-wolf populations. No rates or population derivatives are
injected.

## Training

Run from the repository root:

```bash
python -m car_acceleration.run_rabbit_experiment
```

The defaults use 2,000 optimizer steps, 200 observed points, a 600-step future,
and `dt=0.1`. Results are written to:

```text
car_acceleration/results/rabbit_wolf_narrow_rates_seed7
```

The compressed forecast curriculum is:

| Optimizer steps | Forecast horizon |
| --- | ---: |
| 1–400 | 25 |
| 401–800 | 50 |
| 801–1,200 | 100 |
| 1,201–1,600 | 250 |
| 1,601–2,000 | 600 |

## Outputs

The result directory contains the checkpoint, numeric predictions,
normalization statistics, training history, joint population rollouts,
ground-truth dynamics, latent plots, and held-out physical-information probes.

The diagnostic state is

```text
log(rabbits), log(wolves), log(alpha), log(beta), log(gamma), log(delta)
```

These labels are used only after training to measure recoverable information.
They do not contribute to the forecast loss.
