# Copernicus solar-system replication

This secondary experiment asks whether the trainable intermediate
representation in the deep reservoir can discover the same kind of
heliocentric coordinates reported by Iten et al. in
[*Discovering physical concepts with neural networks*](https://arxiv.org/abs/1807.10300).
It is kept separate from the NARMA10 comparison so that neither dataset,
architecture constraints, nor conclusions are mixed.

## What is replicated

The simulation assumes circular Earth and Mars orbits with constant angular
velocity. A training example contains only the initial angles of the Sun and
Mars as seen from Earth,

```text
observation = (theta_S(t0), theta_M(t0)),
```

and the target is the sequence of the same two Earth-view angles at weekly
intervals. Mars angles are unwrapped within each sequence before applying MSE,
as in the public SciNet data generator. The paper protocol uses 95,000 training
examples, 5,000 validation examples, 5,000 test examples, a two-neuron latent,
and a curriculum from 20 to 50 observations.

The key architectural constraint is a constant latent update:

```text
z[t+1] = z[t] + delta
```

This is what makes heliocentric angles attractive: under the circular-orbit
assumption, both advance by a constant every week. Merely predicting future
angles is not sufficient evidence that the representation is heliocentric.

## Reservoir adaptation

The `reservoir` model preserves the project's central design: both recurrent
weight matrices and input/interlayer projections remain fixed after random
initialization. Only the bottleneck and final readout are optimized.

```text
x1[k+1] = (1-a)x1[k] + a tanh(A1 x1[k] + B1 observation)
z[0]    = tanh(W x1[K] + b)
z[t+1]  = z[t] + delta
x2[0,0] = 0
x2[t,0] = x2[t-1,K_second]                 for t > 0
x2[t,k+1] = (1-a)x2[t,k] + a tanh(A2 x2[t,k] + R z[t])
yhat[t] = Wout x2[t,K_second] + c
```

`A1`, `A2`, `B1`, and `R` are fixed. `W`, `b`, `delta`, `Wout`, and `c` are
trained end to end. The first reservoir is driven for `--encoder-steps` updates
by the one initial observation. The second reservoir is initialized once and
then retains its recurrent state across the complete forecast. It receives the
current latent for `--second-reservoir-steps` recurrent updates per week before
the readout produces that week's two angles.

This is a conceptual replication rather than an identical architecture. The
paper uses fully trainable 100-100 MLP encoder/decoder networks and a beta-VAE;
the reservoir model deliberately substitutes this project's learned bottleneck
between fixed random recurrent dynamics.

## SciNet reference

The optional `scinet` model is a PyTorch reference for the paper's active
Copernicus graph:

- 100-100 ELU encoder producing a Gaussian two-dimensional latent;
- beta-VAE KL penalty with target latent standard deviation 0.1;
- learned additive latent update; and
- shared 100-100 ELU decoder at every forecast time.

The reference follows the active public TensorFlow graph literally, including
two accidental-looking behaviors. The source clips the latent log standard
deviation and then immediately overwrites the clipped tensor, so this port also
uses the unrestricted encoder output. The graph also creates and L2-regularizes
an Euler weight matrix but never applies it: the update adds only its learned
bias. The PyTorch model retains that unused matrix and its default coefficient
of `1.0` so that its objective and trainable parameter count match the released
graph, rather than a cleaned-up interpretation of it.

## Data sampling

The default `--sampling-mode independent_catalog` independently selects the
initial Earth and Mars phases from their weekly phase catalogs over Copernicus'
25,657-day lifetime. This gives the two-dimensional state coverage used by the
public experiment while using the correct Mars catalog for Mars. The public
generator appears to index the Earth catalog for both initial phases; that
likely typographical bug is not reproduced.

Alternative modes are available for sensitivity checks:

- `coupled_catalog`: Earth and Mars use the same historical week index;
- `continuous`: both initial phases are sampled continuously and independently
  on `[0, 2*pi)`.

Train, validation, and test sets use deterministic child seeds. Heliocentric
angles are saved for analysis only and never enter training.

## Training schedules

The default phases follow the public notebook:

| Phase | Counts | Batch | Learning rate | Beta | Horizon |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,000 | 256 | 1e-4 | 0.1 | 20 |
| 2 | 1,000 | 1,024 | 1e-4 | 0.1 | 20 |
| 3 | 1,000 | 1,024 | 1e-4 | 0.1 | 50 |
| 4 | 1,000 | 2,048 | 1e-5 | 0.01 | 50 |
| 5 | 11,000 | 2,048 | 1e-5 | 0.001 | 50 |

By default, counts mean minibatch optimizer updates. This makes the secondary
experiment feasible for multi-seed comparison. In the original TensorFlow
training loop, they mean full shuffled passes through all 95,000 examples. Use
`--full-dataset-epochs` to reproduce that literal behavior. It expands the
schedule to millions of optimizer updates and should normally be run only for
the `scinet` reference on suitable compute.

The beta term is the original KL divergence for `scinet`. For the deterministic
reservoir latent, it is a mean-squared activation penalty; this is the closest
deterministic analogue, not an exact beta-VAE objective.

`history.json` records reconstruction and KL losses every
`--training-log-interval` optimizer updates (default 100). At each validation
point it also records the minimum, mean, and maximum latent standard deviation,
the corresponding log-sigma values, per-dimension mean sigma, and the complete
latent update vector. These diagnostics are printed in the Slurm log and shown
in `training.png`. Sampling keeps the full-budget history compact enough to
store and inspect.

## Running it

Fast pipeline and artifact check:

```bash
python run_solar_experiment.py --quick --device cpu \
  --output-dir results/solar_quick
```

Paper-sized data and the practical 15,000-update curriculum:

```bash
python run_solar_experiment.py \
  --models reservoir scinet \
  --seeds 0 1 2 3 4 \
  --train-samples 95000 --validation-samples 5000 --test-samples 5000 \
  --series-length 50 --latent-size 2 \
  --nodes-1 150 --nodes-2 150 \
  --output-dir results/solar_replication
```

Literal full-pass reference schedule (very expensive):

```bash
python run_solar_experiment.py \
  --models scinet --seeds 0 \
  --full-dataset-epochs \
  --output-dir results/scinet_literal_replication
```

The included Slurm script runs five seeds of both models:

```bash
sbatch run_solar_experiment.sbatch
```

All schedule values, reservoir hyperparameters, sizes, sampling modes, and
devices can be changed from the CLI; run `python run_solar_experiment.py --help`.

## Deciding whether the latent is heliocentric

Each run reports:

- `test_relative_rmse_2pi`: forecast RMSE divided by `2*pi`; compare with the
  paper's reported value below `0.004`;
- `heliocentric_to_latent_r2`: held-out R-squared when each latent activation is
  fit as a linear combination of the heliocentric angles on the same
  angle-unwrapped grid used for the Figure 3-style surface;
- `geocentric_to_latent_r2`: the same diagnostic using observed Earth-view
  angles as a competing explanation;
- `latent_to_heliocentric_r2`: held-out R-squared for decoding heliocentric
  angles linearly from the latent;
- `latent_delta_cosine_similarity` and `latent_delta_relative_error`: agreement
  between the learned update and the update implied by the heliocentric fit.

The raw test distribution necessarily contains an arbitrary `+/-pi` branch cut
for cyclic angles. Its intentionally secondary scores are saved as
`test_branch_heliocentric_to_latent_r2` and
`test_branch_latent_to_heliocentric_r2`; they should not replace the unwrapped
coordinate-chart scores when comparing with Figure 3.

A convincing positive result has low forecast error, heliocentric fit values
near one (preferably higher than the geocentric fit), update cosine near one,
and small update relative error across several seeds. The latent coordinates do
not need to equal `phi_E` and `phi_M` individually: any invertible linear mixture
is equivalent and matches the paper's claim.

## Outputs

```text
results/solar_replication/
├── config.json
├── metrics.csv
├── metrics.json
├── summary.csv
├── summary.json
└── reservoir/seed_0/
    ├── checkpoint.pt
    ├── history.json
    ├── metrics.json
    ├── predictions.npz
    ├── predictions.png
    ├── latent_surface.npz
    ├── latent_surfaces.png
    └── training.png
```

`predictions.npz` stores the Earth-view observations, targets, predictions,
latent trajectories, and withheld heliocentric states. `latent_surfaces.png`
is the direct analogue of the paper's Figure 3c; it evaluates the encoder over
a grid of heliocentric Earth/Mars angles and plots up to four latent dimensions.
