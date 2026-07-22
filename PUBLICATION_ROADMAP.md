# Publication roadmap

## Proposed paper

**Working title:** *Learning Interpretable Low-Dimensional Dynamics Between
Stacked Reservoir Networks*

**Central claim:** Small trainable intermediate representations between
successive reservoir layers can improve prediction and recover physically
meaningful state variables while most recurrent parameters remain fixed after
random initialization.

Keep the claim narrow. The current results do **not** establish that the
nonlinear bottleneck is better than the linear bottleneck, or that the solar
model matches SciNet's prediction accuracy.

## Current evidence

- On NARMA10, the learned-bottleneck models outperform the tested ESN and
  DeepESN baselines, but use more trainable parameters. The linear bottleneck is
  currently marginally better than the nonlinear one.
- In the earlier two-reservoir solar experiment, the two-neuron representation
  is strongly aligned with heliocentric coordinates (`R²` approximately
  `0.98`). This is the most interesting result. The new 10-layer configuration
  has not yet produced evidence and must be evaluated separately.
- Solar prediction remains limited: RMSE is approximately `4.5–5.6%` of
  `2π`, versus approximately `0.218%` from the released SciNet checkpoint. The
  Earth-view Mars trajectory still misses important local curvature.

## Minimum experiments before submission

### 1. Lock the study design

- Choose one primary configuration before the final runs.
- Keep exploratory changes, such as velocity or curvature losses, as named
  ablations rather than silently changing the primary method.
- Reserve one untouched dataset seed or test split for the final evaluation.

### 2. Strengthen the NARMA result

- Run at least 20 paired seeds.
- Add parameter-matched and compute-matched controls.
- Tune every model with the same validation budget.
- Sweep bottleneck sizes, reservoir sizes, and linear versus nonlinear
  intermediate mappings.
- Report paired effect sizes, bootstrap 95% confidence intervals, training
  time, and trainable parameter counts.

### 3. Complete the solar study

- Run at least 10 seeds for the final reservoir configuration.
- Evaluate the released SciNet checkpoint as a fixed reference and report
  separately whether training SciNet from scratch reproduces it.
- Compare second-reservoir state handling, preliminary-update counts, and
  interlayer scales as explicit ablations.
- Compare the earlier 2-layer architecture against the new 10-layer stack;
  report depth as an ablation rather than attributing old results to the new
  model.
- Report Sun and Mars RMSE separately, error by forecast week, latent `R²` in
  both directions, learned-update agreement, and latent-surface plots.
- Include velocity/curvature losses only as follow-up variants, with ordinary
  MSE retained as the main comparison.
- Show several randomly selected trajectories; do not select only the most
  visually favorable example.

### 4. Demonstrate generality

Add at least one task beyond NARMA10 and the circular solar toy problem. Strong
options are noisy or elliptical planetary motion, a damped pendulum, or another
system with known low-dimensional state. The key test is whether the learned
intermediate representation remains interpretable when the dynamics are less
idealized.

## Publication gates

Proceed to a full paper when all of the following hold:

1. The NARMA improvement survives parameter-matched controls and at least 20
   paired seeds.
2. Heliocentric alignment remains high across at least 10 seeds, not just one.
3. The Mars trajectory captures its nonlinear changes in velocity on held-out
   examples.
4. A third task shows that the method is not specific to one benchmark or one
   physical simulation.
5. Every reported result can be reproduced from one command and a saved
   configuration.

If only gates 1–2 are achieved, frame the work as a focused workshop or short
paper. If all five are achieved, target a full reservoir-computing,
computational-neuroscience, or machine-learning-for-science venue.

## Reproducibility package

- Save exact configurations, commit hashes, environment files, and Slurm logs.
- Publish per-seed metrics and compact diagnostic histories.
- Provide scripts that regenerate every table and figure.
- Include the official SciNet checkpoint evaluation and clearly distinguish it
  from newly trained SciNet runs.
- Document unsuccessful or unstable configurations in supplementary material.

## Suggested manuscript structure

1. Motivation and related work
2. Stacked-reservoir method and trainable intermediate representations
3. NARMA10 predictive comparison
4. Solar-system representation experiment
5. Ablations, robustness, and additional dynamical system
6. Limitations and reproducibility

## Immediate order of work

1. Resolve the Mars trajectory shape and select the primary solar objective.
2. Run the final 10-seed solar experiment.
3. Run the 20-seed matched NARMA study.
4. Add one harder dynamical system.
5. Generate publication figures and release a preprint with the complete code
   and per-seed results.
