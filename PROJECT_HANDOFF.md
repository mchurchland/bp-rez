# Reservoir-encoded autonomous latent dynamics

Research handoff — 28 August 2026

This document packages the current project for a new student or collaborator. It
separates the proposed scientific contribution from the claims that the saved
results presently support. All commands assume execution from the repository
root. It covers the physical-latent line of work; the separate NARMA predictive
benchmark remains documented in [`narma/README.md`](narma/README.md).

## 1. Concept note

### Working title

Reservoir Encoders for Compact Autonomous Latent Dynamics

### One-sentence summary

Use a fixed random reservoir to summarize observations, learn a compact initial
latent state from its features, and train a simple autonomous latent recurrence
through multi-step forecasting; then test whether that task-trained state is
linearly equivalent to the hidden physical state.

### Motivation and question

Forecasting from a short observation history requires two different operations.
The model must first infer the state at the end of the history, and it must then
advance that state through time. A conventional recurrent predictor can perform
both operations inside one large hidden state, but this makes the simulated
state and its evolution difficult to inspect.

This project explicitly separates those operations. A fixed reservoir supplies
a high-dimensional nonlinear summary of the observations. A small trainable
projection extracts an initial latent state. A low-dimensional, autonomous
dynamical law then advances that state without receiving future observations.
Only the projection, latent dynamics, and observation decoder are trained.

The central research question is:

> Under what conditions does a compact state trained only for future
> prediction become equivalent to the hidden physical state, rather than merely
> retaining whichever information is sufficient for the forecast distribution?

The question is deliberately weaker than claiming that prediction always
discovers physical variables. The predator–prey result is evidence that it does
not.

### General architecture

For observations \(u_{0:H}\), fixed reservoir states satisfy

\[
r_t=(1-\lambda)r_{t-1}
    +\lambda\tanh(A_r r_{t-1}+B_r u_t+b_r),
\]

where \(A_r\), \(B_r\), and any fixed bias are sampled once and never optimized.
One or more reservoir states are collected into a feature vector \(R\). The
trainable part is

\[
z_0=E_\theta(R),\qquad
z_{k+1}=F_\theta(z_k),\qquad
\hat y_k=D_\theta(z_k).
\]

A multi-step forecast loss is backpropagated through \(D_\theta\), the entire
latent rollout, and \(E_\theta\), but not through trainable reservoir weights
because the reservoir matrices are fixed. Hidden physical variables are used
only after training for diagnostic probes.

This division gives each component a specific role:

- The reservoir is a nonlinear temporal feature map.
- The encoder chooses a compact initial condition from those features.
- The autonomous recurrence is the learned simulator.
- The decoder maps simulated state back to measured quantities.

The latent axes themselves need not equal named physical coordinates. If
\(p=Wz+c\) is an invertible affine change of coordinates, \(z\) and \(p\) carry
the same state information. Interpretation should therefore use held-out
bidirectional probes and, where possible, compare the transformed recurrence
\(W F W^{-1}\) with the known physical law.

### Three implementations

<!-- markdownlint-disable MD013 -->

| System | Observations | Latent simulator | Decoder | Diagnostic physical state |
| --- | --- | --- | --- | --- |
| Solar | Initial Earth-view Sun and Mars angles | 2D additive update, \(z_{k+1}=z_k+\Delta z\) | Fixed downstream reservoir stack plus linear readout | Heliocentric Earth and Mars angles |
| Car | Five positions | 3D affine update, \(z_{k+1}=Az_k+b\) | Positive linear position readout | Position, velocity, acceleration |
| Rabbits–wolves | 200 paired log-population observations; five reservoir taps | 8D bounded quadratic residual update | Linear two-population readout | Populations and trajectory-specific \(\alpha,\beta,\gamma,\delta\) |

<!-- markdownlint-enable MD013 -->

The implementations instantiate the same idea but are not identical. Solar
uses an initial observation rather than a temporal history and employs fixed
reservoirs in the decoder. Car and predator–prey use a fixed temporal reservoir
only as the encoder and decode directly from the autonomous latent.

### Proposed contribution

The defensible contribution is:

1. **Architecture.** A fixed nonlinear history encoder initializes a compact,
   trainable, and explicitly autonomous latent simulator.
2. **Training.** Multi-step backpropagation trains the reservoir-to-latent
   projection, latent recurrence, and decoder while leaving recurrent reservoir
   matrices fixed.
3. **Physical-state audit.** Held-out probes separately measure information in
   the reservoir features and the learned bottleneck. This distinguishes a
   failure of observation encoding from information discarded by the
   predictive latent.
4. **Positive and negative evidence.** Solar provides the strongest current
   evidence of a physically aligned latent. Predator–prey shows that a model
   can represent and forecast populations while discarding identifiable rate
   information available in its reservoir features.

The last point is part of the contribution, not an experiment to hide. It
supports the narrower conclusion that architectural structure and prediction
pressure can encourage physical representations, but prediction accuracy alone
does not identify hidden parameters.

### Claims this project does not yet support

- It does not establish that the method generally discovers physical
  parameters.
- It does not establish coordinate-wise disentanglement; invertible mixtures
  are physically equivalent representations.
- It does not yet establish a forecasting advantage over parameter- and
  compute-matched alternatives.
- It does not yet establish the car acceleration claim with a proper held-out
  acceleration probe.
- It does not establish robustness from a single favorable model seed.
- It is not an identical reproduction of SciNet; the solar experiment is a
  reservoir-based conceptual adaptation.

A suitable first destination is therefore a workshop, short paper, student
project, or technical preprint. A full paper would require a locked protocol,
several seeds, matched baselines, held-out physical probes, and an explicit
related-work comparison.

### Recommended next-owner task

The best bounded question for a new owner is:

> Which combinations of observation history, bottleneck dimension, latent-law
> structure, and forecast horizon make the hidden physical state necessary for
> prediction?

The first milestone should not be another reservoir-size sweep. It should be a
reproducible evaluation pass: save every run configuration, restore the best
validation checkpoint, use train/validation/test-separated physical probes,
and compare against direct-history, reservoir-only, PCA, and trainable recurrent
baselines.

## 2. One-command reproductions

### Environment

Python 3.10 or newer is recommended. Install the four declared dependencies
once:

```bash
pip install -r requirements.txt
```

`--device auto` selects CUDA when available. The car and predator–prey commands
set `MPLBACKEND=Agg` so plots also work in headless and Wayland environments.

### Solar: current ten-layer protocol

This runs the reservoir model for five model seeds with the paper-sized data
and 15,000-update curriculum. It is the most expensive command in this handoff.

```bash
python -m solar.run_experiment \
  --models reservoir --seeds 0 1 2 3 4 --data-seed 2026 \
  --train-samples 95000 --validation-samples 5000 --test-samples 5000 \
  --series-length 50 --latent-size 2 \
  --nodes-1 150 --nodes-2 150 --reservoir-layers 10 \
  --device auto \
  --output-dir solar/results/handoff_solar_10x150_5seed
```

For a fast pipeline check, replace the command above with
`python -m solar.run_experiment --quick --models reservoir --device cpu
--output-dir solar/results/handoff_solar_smoke`.

### Car acceleration: current protocol

```bash
MPLBACKEND=Agg python -m car_acceleration.run_experiment \
  --seed 8 --steps 100000 --covariance-weight 0.5 --device auto \
  --output-dir car_acceleration/results/handoff_car_seed8
```

The live code uses five observed positions, a 50-step forecast, `dt=1`, a fixed
150-node reservoir, and a 3D latent. Some older names and documentation still
say `2latent` or 30 steps; those labels are stale.

After training, inspect the affine recurrence and its fitted physical-coordinate
transform with:

```bash
python -m car_acceleration.print_latent_recurrence \
  --checkpoint car_acceleration/results/handoff_car_seed8/checkpoint.pt
```

### Rabbit–wolf populations: current narrow-rate protocol

```bash
MPLBACKEND=Agg python -m car_acceleration.run_rabbit_experiment \
  --seed 7 --steps 2000 --history-length 200 --forecast-horizon 600 \
  --dt 0.1 --covariance-weight 0 --device auto \
  --output-dir car_acceleration/results/handoff_rabbit_wolf_seed7
```

This command receives both populations, predicts both populations, and never
injects \(\alpha,\beta,\gamma,\delta\) into the learned model. The rates are
used only by the data generator and the post-training probes.

### Reproduction status

These commands reproduce the **current code protocols**, not bit-for-bit every
historical directory. Solar saves a complete `config.json`. Car and
predator–prey currently save only a model `state_dict`, metrics, histories, and
arrays; they do not save the full configuration or source revision. Before
handoff, the new owner should add configuration and commit-hash capture to both
runners.

## 3. Current results and known failures

The values below were read from saved JSON/NPZ artifacts; statistics recomputed
from the arrays are labeled as such. They should be treated as exploratory until
the commands above are rerun from a clean commit.

### Headline results

<!-- markdownlint-disable MD013 -->

| Saved experiment | Forecast result | Physical-information result | Interpretation |
| --- | --- | --- | --- |
| [Solar, earlier two-reservoir replication, 5 seeds](solar/results/solar-replication-98728/summary.json) | Relative test RMSE / \(2\pi\): **0.0577 ± 0.0011** | Heliocentric→latent \(R^2\): **0.9847 ± 0.0014**; latent→heliocentric: **0.9864 ± 0.0010** | Strong multi-seed latent alignment, but weak forecasting. |
| [Solar, ten-layer sequential, seed 0](results/solar-10x150-sequential-15k-111155/reservoir/seed_0/metrics.json) | Relative test RMSE / \(2\pi\): **0.0290** | Heliocentric→latent \(R^2\): **0.9915**; latent→heliocentric: **0.9919**; update cosine: **0.9995** | Best saved combined result, but only one seed. |
| [Solar, latest two-layer decoder variant, seed 0](solar/results/solar-2x150-decoder-fix-15k-118154/reservoir/seed_0/metrics.json) | Relative test RMSE / \(2\pi\): **0.0247** | Heliocentric→latent \(R^2\): **0.9538**; latent→heliocentric: **0.9584**; update cosine: **0.9996** | Lowest saved solar forecast error, with weaker physical alignment and only one seed. |
| [Car, current 100k-step archive](car_acceleration/results/linear_readout_2latent_seed8/metrics.json) | Position RMSE: **18.672** over the saved 50-step forecasts; recomputed forecast \(R^2\): **0.9978**; exact finite-difference baseline RMSE: **0.00021** | The separate physicalization utility reports initial-state \(R^2\): position **0.896**, velocity **0.986**, acceleration **0.997** | The initial latent contains acceleration, but the probe is fitted and scored on the same 500 trajectories and the learned recurrence is not physical. |
| [Car, older three-latent archive](car_acceleration/results/three_latent_run/metrics.json) | Position RMSE: **0.694** over an older 30-step task | Latent→(position, velocity, acceleration) pooled \(R^2\): **0.991**; acceleration alone: **0.009** | Pooled \(R^2\) hides failure to linearly recover acceleration. This is not the current protocol. |
| [Rabbit–wolf narrow rates, seed 7](car_acceleration/results/rabbit_wolf_narrow_rates_seed7/metrics.json) | Rabbit/wolf log RMSE: **0.886 / 0.878**; recomputed log-forecast \(R^2\): **0.449 / 0.461** | Latent \(R^2\): rabbits **0.907**, wolves **0.907**, \(\alpha\) **0.078**, \(\beta\) **0.001**, \(\gamma\) **0.047**, \(\delta\) **0.007** | The bottleneck retains population state but almost completely discards rate identity. |

For the rabbit–wolf run, the corresponding reservoir-feature \(R^2\) values
are rabbits **0.992**, wolves **0.990**, \(\alpha\) **0.673**, \(\beta\)
**0.359**, \(\gamma\) **0.636**, and \(\delta\) **0.318**. Thus the fixed
reservoir contains substantially more rate information than the learned latent.
That is evidence of task-driven information removal, not simply a deficient
reservoir.

### Known failures and threats to validity

| Issue | Evidence | Required correction |
| --- | --- | --- |
| Car acceleration is absent from the primary saved metric. | [`_evaluate`](car_acceleration/experiment.py) constructs only position and velocity; the prediction archive omits explicit acceleration. The separate utility reconstructs acceleration by finite differencing velocity. | Save acceleration and make the held-out three-coordinate probe part of the canonical evaluation. |
| The car physicalization utility reuses the same trajectories for fitting and reporting. | [`print_latent_recurrence.py`](car_acceleration/print_latent_recurrence.py) fits \(W\) from all 500 trajectories in the prediction archive and reports fit quality on those same rows. | Fit on training trajectories, select choices on validation, and report only test \(R^2\) with uncertainty across seeds. |
| The current car recurrence is not the physical constant-acceleration recurrence under its fitted chart. | Learned \(A_{physicalized}\) is `[[0.621, 2.575, -1.835], [-0.070, 1.466, -0.198], [0.007, -0.040, 1.099]]`; ideal `dt=1` kinematics gives `[[1, 1, 0.5], [0, 1, 1], [0, 0, 1]]`. Using the fitted chart over the rollout gives \(R^2\) of **0.977**, **-4.604**, and **-240.831** for position, velocity, and acceleration. | Evaluate physical equivalence over the complete rollout, constrain or regularize the latent law if physical dynamics are the goal, and compare with the exact finite-difference baseline. |
| Pooled \(R^2\) can conceal a failed physical coordinate. | The older car archive reports pooled \(R^2=0.991\) while acceleration \(R^2=0.009\). | Always report every physical dimension alongside any pooled score. |
| Rabbit–wolf validation values are not comparable across the curriculum. | [History](car_acceleration/results/rabbit_wolf_narrow_rates_seed7/history.json): validation loss is **0.0780** at step 500 on horizon 50 and **0.5078** at step 2,000 on horizon 600. | Add a fixed-horizon validation curve and select checkpoints with the final evaluation horizon. |
| Final rather than best checkpoints are saved in car and predator–prey. | Both runners serialize only after their optimization loops. | Track and restore a validation-selected checkpoint before test evaluation. |
| Narrow predator–prey rates make parameter recovery less necessary. | Rates vary only over `alpha,gamma=0.8–1.2` and `beta,delta=0.08–0.12`; population forecasts can approximate an average system. | Evaluate broad and narrow ranges as named conditions and measure whether parameter variation changes future trajectories enough to require identification. |
| Solar forecasting still trails the released SciNet reference. | The latest saved reservoir error is **2.469%** of \(2\pi\); the [released-checkpoint comparison](solar/results/solar-10x150-preserved-vs-scinet-presentation/comparison_metrics.json) reports SciNet at **0.218%**. The roughly 11× ratio is an inference across those saved evaluations. | Treat physical alignment and forecast performance as separate outcomes; rerun a locked, same-split comparison before making a performance claim. |
| Solar physical alignment is not uniquely heliocentric. | In the ten-layer seed, geocentric→latent \(R^2=0.969\) versus heliocentric→latent \(R^2=0.992\). | Report both explanations, their difference across seeds, nonlinear-probe controls, and latent update agreement. |
| The best ten-layer solar evidence is a single seed. | Its summary reports one seed. | Rerun the locked configuration for at least 5 exploratory and 10 final seeds. |
| Historical and live protocols have drifted. | Car horizon, latent labels, output names, function/CLI defaults, and saved configurations disagree. | Lock one protocol, save the complete configuration and source revision, and archive old runs as explicitly historical. |

<!-- markdownlint-enable MD013 -->

### Minimum handoff acceptance criteria

A new owner should consider the package ready for a paper only when:

1. A clean checkout can reproduce every headline table row from one command.
2. Every main result has at least five exploratory seeds and the final claim has
   at least ten locked seeds.
3. Physical probes are fitted without using test labels and report each state
   dimension separately.
4. The same held-out split and tuning budget are used for reservoir, PCA,
   direct-history, and trainable recurrent baselines.
5. Best-checkpoint selection uses validation data, and the test set is evaluated
   once after the protocol is locked.
