# Discovering car dynamics with a reservoir

The rabbit-only Lotka-Volterra counterpart is documented in
[`RABBIT_README.md`](RABBIT_README.md).

This experiment asks a deliberately small representation-learning question:

> Can a model infer position, velocity, and a car-specific constant
> acceleration when it observes only a short sequence of positions?

Every trajectory has independently sampled initial conditions and acceleration.
The model receives no velocity or acceleration input. It must compress five
observed positions into a three-dimensional latent state and use that state to
predict the next 30 positions.

## Problem setup

For each car,

```text
initial position  x0 ~ Uniform(-5, 5)
initial velocity  v0 ~ Uniform(-2, 2)
acceleration       a ~ Uniform(-1, 1)
time step         dt = 1
```

Acceleration is constant within a trajectory but differs between cars. The
dynamics are therefore

```text
x[t+1] = x[t] + v[t] dt + 0.5 a dt^2
v[t+1] = v[t] + a dt
a[t+1] = a[t]
```

The default dataset contains five observed positions at times `0,...,4` and 30
forecast targets at times `5,...,34`. Because each car has its own unknown
acceleration, the minimal physical state has three degrees of freedom:
`(position, velocity, acceleration)`.

## Model

```text
five positions
      |
      v
fixed 150-node reservoir
      |
      v
learned 3D latent z[0]
      |
      |  z[t+1] = A z[t] + b
      v
30-step latent rollout
      |
      v
positive linear position readout
```

The position history is divided by `10` before entering the reservoir. The
reservoir is fixed after deterministic random initialization; only the encoder,
latent transition, and final readout are trained.

| Component | Default |
| --- | ---: |
| Reservoir nodes | 150 |
| Spectral radius | 0.99 |
| Recurrent density | 0.10 |
| Leak rate | 0.80 |
| Input projection scale | 1.50 |
| Latent dimensions | 3 |
| History length | 5 |
| Forecast horizon | 30 |
| Time step | 1.0 |

The latent-to-position weights are constrained to be positive with a softplus
parameterization. The latent transition and its bias are unconstrained and
learned end to end.

## Training

Training minimizes normalized future-position error plus an off-diagonal
covariance penalty on the initial latent:

```text
loss = position_mse + lambda_cov * sum(offdiag(Cov(z[0]))^2)
```

The command-line defaults are:

| Setting | Value |
| --- | ---: |
| Training trajectories | 3,000 |
| Validation trajectories | 500 |
| Test trajectories | 500 |
| Batch size | 128 |
| Optimizer | Adam |
| Optimizer steps | 100,000 |
| Initial learning rate | 1e-3 |
| Learning rate after step 5,000 | 1e-4 |
| Learning rate after step 40,000 | 1e-5 |
| Covariance weight | 0.5 |
| Gradient-norm limit | 1.0 |

Training, validation, and test sets are generated independently from adjacent
seeds. Mini-batches sample training trajectories with replacement.

## Running the experiment

From the repository root:

```bash
pip install -r requirements.txt
python -m car_acceleration.run_experiment
```

Useful overrides:

```bash
python -m car_acceleration.run_experiment \
  --seed 8 \
  --steps 10000 \
  --covariance-weight 0.5 \
  --device cpu \
  --output-dir car_acceleration/results/car_dt1_seed8
```

`--device auto` selects CUDA when available and otherwise uses the CPU.

### Correlation-based latent ordering

After a run, compare the physical and latent correlation matrices and find the
best one-to-one latent-axis ordering with:

```bash
python -m car_acceleration.analyze_latent_correlations \
  --predictions car_acceleration/results/car_dt1_seed8/predictions.npz \
  --dt 1
```

The script tests all six permutations of `(z1, z2, z3)` and selects the one
whose pairwise correlation pattern most closely matches `(position, velocity,
acceleration)`. It also loads the neighboring `checkpoint.pt` and reports the
latent transition matrix, transition bias, eigenvalues, and the correctly
permuted recurrence. Add `--output-json path/to/analysis.json` to save every
matrix and comparison value. Use `--checkpoint path/to/checkpoint.pt` when the
checkpoint is not beside the predictions file.

To print the final recurrence stored in a checkpoint and its fitted physical
coordinate transform, `A_physicalized = W A W^-1`:

```bash
python -m car_acceleration.print_latent_recurrence \
  --checkpoint car_acceleration/results/car_dt1_seed8/checkpoint.pt
```

The script fits `physical = W z + c` from the `predictions.npz` beside the
checkpoint, using physical order `(position, velocity, acceleration)`. Pass
`--predictions path/to/predictions.npz` to use a different file.

## Outputs

Each run writes the following files to its output directory:

| File | Contents |
| --- | --- |
| `checkpoint.pt` | Model parameters and fixed reservoir matrices |
| `history.json` | Step, training loss, validation loss, and learning rate |
| `metrics.json` | Position RMSE, latent probes, and latent covariance |
| `predictions.npz` | Numeric inputs, outputs, states, and latents |
| `training.png` | Training and validation curves |
| `predictions.png` | Example held-out position forecasts |
| `ground_truth_dynamics.png` | Physical position, velocity, and acceleration |
| `latent_dynamics.png` | Physical coloring of the latent and one rollout |

The default output-directory name still contains `2latent` for historical
reasons; the current model uses a three-dimensional latent.

## Interpreting the latent space

The three latent coordinates are not required to become position, velocity,
and acceleration individually. Any invertible rotation or linear mixing can
represent the same three-dimensional state. Smooth color structure and linear
probe R² are therefore more informative than assigning a physical name to a
single axis.

Covariance also requires care. At the end of the observed history,

```text
v(4) = v0 + 4a
x(4) = x0 + 4v0 + 8a
```

so current position, velocity, and acceleration are naturally correlated even
though `x0`, `v0`, and `a` were sampled independently. A physically meaningful
latent may consequently contain correlated coordinates. The implemented
regularizer penalizes raw covariance—not correlation or statistical
dependence—and a larger weight does not guarantee coordinate-wise physical
disentanglement.

Most importantly, acceleration is never injected into the encoder. It appears
only as the curvature of the observed position sequence. Velocity and
acceleration are retained in the dataset solely for diagnostics and plotting.

## Current caveats

- With `dt=1` and a 30-step horizon, the model forecasts 30 physical time
  units. Quadratic position growth makes this substantially harder than the
  earlier `dt=0.1` task and can make the fixed position scale of `10` too small.
- The saved checkpoint is the final training state, not necessarily the state
  with the lowest validation loss.
- The covariance penalty is evaluated on mini-batches and only on the initial
  latent, not on every future latent state.
- The direct Python function has research-oriented defaults that differ from
  the command-line runner; use the module command above for the documented
  10,000-step configuration.

## Source map

- [`data.py`](data.py): trajectory generation and train/target construction
- [`model.py`](model.py): fixed reservoir, latent encoder, dynamics, and readout
- [`experiment.py`](experiment.py): optimization, evaluation, and figures
- [`run_experiment.py`](run_experiment.py): command-line entry point
- [`analyze_latent_correlations.py`](analyze_latent_correlations.py):
  latent-axis alignment
- [`print_latent_recurrence.py`](print_latent_recurrence.py): final recurrence
