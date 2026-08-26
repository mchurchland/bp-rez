# Car acceleration experiment

This is a small teaching experiment separate from the solar-system runs. It
simulates one-dimensional cars with random initial positions and velocities,
plus one fixed nonzero acceleration shared by all cars. The model sees only
three initial positions and predicts the next 30 positions.

This version has one fixed 150-neuron encoder reservoir and a direct linear
readout from a 2D latent. The two latent-to-position weights are constrained to
be strictly positive, and a covariance penalty discourages the two latent
coordinates from moving together:

```text
three positions -> fixed reservoir (150) -> learned 2D latent z
                 -> learned affine latent dynamics
                 -> positive-weight linear position readout
```

The default covariance penalty weight is `0.01`. It encourages uncorrelated
latent coordinates but does not by itself guarantee that latent 1 is position
and latent 2 is velocity.

The latent transition is learned as

```text
z[t+1] = A z[t] + b
```

Because acceleration is fixed globally, the latent only needs to retain the
changing position and velocity. The acceleration contribution can be stored in
the shared transition bias.

Run one experiment from the repository root:

```bash
python -m car_acceleration.run_experiment
```

Results are written to `car_acceleration/results/linear_readout_2latent/`:

- `ground_truth_dynamics.png`: simulated position, velocity, and acceleration;
- `predictions.png`: held-out future-position predictions;
- `latent_dynamics.png`: 2D latent/state relationships and one latent trajectory;
- `training.png`: training and validation loss;
- `metrics.json`, `predictions.npz`, and `checkpoint.pt`: numeric results.

The latent plots are diagnostics, not proof that latent 1 is position and
latent 2 is velocity. A latent can rotate or mix those physical quantities and
still represent the same state. The reported linear-probe R² values test how
well position and velocity can be recovered from the learned latent.
