# Publication NARMA benchmark

This package evaluates whether a task-trained, low-dimensional communication
channel between two fixed reservoirs improves nonlinear temporal computation.
It replaces the original five-seed NARMA-10 pilot with a development/final
protocol designed for paired statistical comparison.

The legacy pilot and its archived results remain under `narma/legacy/` and
`narma/results/narma10-300nodes-95970/`.

## Locked study design

The full study contains:

- four explicitly named NARMA tasks: 5, 10, 20 and 30;
- small (2,000 generated point) and long (10,000 generated point) training
  conditions;
- 2,000 generated validation and 10,000 generated final-test points;
- 200 discarded washout points in every independently reset sequence;
- ten paired final replicates;
- eight models receiving bit-identical data within every paired replicate; and
- three separate development seeds used only for hyperparameter selection.

The small training sequence is a prefix of the long sequence. Validation and
test streams are identical between these data regimes. A SHA-256 digest of
every input/target split is written to each pair's `data_manifest.json`.
Because the first 200 samples are washout, the corresponding supervised/scored
counts are 1,800 for each 2,000-point split and 9,800 for each 10,000-point
split.

The development phase constructs only training and validation data. It cannot
construct a final test split through its API. Once tuning finishes, the selected
configuration for every task, regime and model is hashed into
`locked_configs.json`. Only then can the final workers construct and evaluate
the fresh final-test streams.

## Exact NARMA definitions

The generic unwrapped recurrence is

```text
y[t+1] = alpha*y[t]
       + beta*y[t]*sum(y[t-i], i=0..N-1)
       + gamma*u[t-N+1]*u[t]
       + delta
```

`u[t]` is consumed before the model predicts `y[t+1]`. Missing delayed inputs
and the initial output history are zero. Data generation is performed in
float64, checked for divergence, then stored as float32.

| Cited recurrence | `(alpha, beta, gamma, delta)` | Proposed input | Outer function |
|---|---|---|---|
| `narma5_fujii_nakajima` | `(0.3, 0.05, 1.5, 0.1)` | `Uniform(0, 0.2)` | identity |
| `narma10_atiya_parlos` | `(0.3, 0.05, 1.5, 0.1)` | `Uniform(0, 0.5)` | identity |
| `narma20_rodan_tino` | `(0.3, 0.05, 1.5, 0.01)` | `Uniform(0, 0.5)` | `tanh` |
| `narma30_schrauwen` | `(0.2, 0.04, 1.5, 0.001)` | `Uniform(0, 0.5)` | identity |

Those uniform laws are the **proposal** distributions. To handle the known
instability of unbounded NARMA recurrences without model-dependent retries, the
benchmark uses a predeclared deterministic rejection sampler: accept the first
derived seed whose target is finite and satisfies `|y[t]| <= 1e6` over the
common 10,000-point acceptance horizon, then take the requested prefix.
Accepted inputs are therefore uniform proposals conditioned on target
stability, not exactly unconditional uniform samples. The manifest records
every rejected seed, attempt and acceptance horizon. Using one horizon for
training, validation and final test keeps their accepted law consistent; the
small training prefix remains intentionally conditioned on its unseen
continuation.

Accordingly, reported task names append
`_stable_trajectory_conditional`; they are not labeled exact unconditional
replications of the cited data processes.

These are cited definitions rather than one synthetic coefficient family.
Consequently, cross-order error is **not** interpreted as a pure curve of
increasing memory difficulty. The generated `PROTOCOL.md` and `protocol.json`
record the equations, citations, indexing, initialization, washout,
normalization and metric definitions used by a particular locked run.

## Models

The required controls are:

| CLI model | Description |
|---|---|
| `esn_ridge` | 150-node fixed ESN with validation-tuned ridge readout |
| `large_esn_ridge` | 300-node single ESN matched to total reservoir nodes |
| `deep_esn_ridge` | fixed 150+150 DeepESN with a stronger `[x1,x2]` readout |
| `random_bottleneck` | fixed random 150→10→150 communication channel |
| `pca_bottleneck` | training-only PCA 150→10 projection |
| `learned_linear` | proposed task-trained linear 150→10 bottleneck |
| `learned_nonlinear` | proposed bottleneck with intermediate `tanh` |
| `gru` | 22-unit GRU parameter-matched to the original 1,661-parameter proposal |

Named random streams make `A1`, `A2`, `B1` and `R` bit-identical when two
models use the same reservoir configuration and pair seed. Every model receives
the same number of unique, model-effective validation trials, but is tuned
independently; selected final configurations can therefore have different
spectral radii, densities or scales. The recurrent layers independently tune
spectral radius, leak rate, sparsity and fixed bias scale. The first-layer input
scale and second-layer interconnection scale are tuned separately.

All readouts use the same float64 SVD ridge implementation:

- features are standardized using training statistics;
- the intercept is not penalized;
- lambda is selected from `10^-12` through `10^4` by validation MSE; and
- after selection, the readout is refitted on training plus validation.

PCA and first-reservoir bottleneck normalization use training data only. Learned
models use AdamW, gradient clipping, `ReduceLROnPlateau`, validation early
stopping, best-checkpoint restoration and a final ridge refit. The default
budget is 1,000 epochs, scheduler patience 20 and early-stopping patience 100.
The configuration lock and final aggregation enforce a convergence gate:
within every learned-model/order/regime group, at most 10% of runs may hit the
epoch cap. A failed gate writes diagnostics and requires a larger
`NARMA_MAX_EPOCHS` in a new output directory (or a deliberately documented
threshold change).

## Submit the Slurm workflow

For a roughly one-hour run on 20 GPUs, use the light profile:

```bash
bash narma/jobs/submit_light_benchmark.sh
```

It retains all four NARMA tasks, both data regimes, all eight models and the
standard sequence lengths, while using one tuning seed, two trials, one paired
final seed, a 150-epoch ceiling, early-stopping patience 25 and up to 20
concurrent workers. The final stage therefore has eight tasks. Compute workers
request a one-hour Slurm limit, and results go under
`narma/results/light_benchmark/`.

This is a pipeline and coarse-ranking check, not a statistical experiment.
Runtime is expected to be approximately 30–60 minutes once resources are
allocated; cluster queue time and hardware variation can make it longer.

From the repository root, the bare launcher now runs the screening profile:

```bash
bash narma/jobs/submit_publication_benchmark.sh
```

The screen keeps the complete benchmark matrix—four NARMA tasks, two data
regimes and all eight models—but reduces the replication and tuning budget:

- two tuning seeds per hyperparameter trial;
- four unique trials per model/order/regime;
- three paired final seeds;
- a 600-epoch ceiling with validation early stopping;
- up to 20 concurrent workers per array; and
- output under `narma/results/screening_benchmark/`.

It submits five dependency-linked stages:

1. a 40-task CPU tuning array for the five fixed/ridge models;
2. a 24-task GPU tuning array for the three gradient-trained models;
3. a configuration-lock job dependent on both tuning arrays;
4. a 24-task paired final array (`4 × 2 × 3 pairs`), with every task running all
   eight models on the same tensors; and
5. an aggregation/statistics job.

The screening profile records how many learned runs reach the epoch ceiling but
does not reject the run on that basis. It is intended for ranking
configurations, finding failures and estimating runtime. Three final pairs do
not support confirmatory confidence intervals or hypothesis tests.

Override the environment or output location with:

```bash
CONDA_ENV=my-environment \
NARMA_OUTPUT_ROOT=narma/results/my_screen \
bash narma/jobs/submit_publication_benchmark.sh
```

For the full publication design—three tuning seeds, eight trials, ten final
pairs, a 1,000-epoch ceiling, the strict convergence gate and eight concurrent
workers—use:

```bash
NARMA_PROFILE=publication \
bash narma/jobs/submit_publication_benchmark.sh
```

That profile creates an 80-task final array and writes to
`narma/results/publication_benchmark/`. For a higher-power 20-pair run, use a
new output directory:

```bash
NARMA_PROFILE=publication \
NARMA_FINAL_PAIR_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19" \
NARMA_OUTPUT_ROOT=narma/results/publication_benchmark_20_pairs \
bash narma/jobs/submit_publication_benchmark.sh
```

The learned-tuning and final workers request one GPU, eight CPUs and 24 GB RAM;
fixed-model tuning uses CPU nodes. Adjust the `#SBATCH` resource lines for the
target cluster. A monolithic 64-task GPU tuning script remains available for
clusters where that is preferable:

```bash
sbatch narma/jobs/tune.sbatch
```

Direct submission of `final.sbatch` succeeds only after the matching tuning
matrix has been locked.

Tuning resumes at completed-trial granularity and final workers resume at
completed-model granularity. Every resumable shard is checked against the
suite and source hashes before reuse.

### Analyze pulled results

Aggregate a completed result directory directly from its locked manifest:

```bash
python -m narma.analysis.aggregate_locked narma/results/light_benchmark
```

This is preferable to rebuilding the original command-line arguments after
moving results between machines. It preserves the exact locked floating-point
settings even when Python or NumPy versions serialize equivalent values
differently.

## Local smoke test

The automated test runs all eight models through a tiny tune-lock-final-
aggregate cycle:

```bash
python -m pytest -q narma/tests/test_benchmark.py
```

Individual development and final commands are also available:

```bash
python -m narma.run_benchmark tune \
  --order 10 --regime small --model learned_linear \
  --orders 10 --regimes small --models learned_linear \
  --final-pair-ids 0 \
  --output-root narma/results/development

python -m narma.run_benchmark lock \
  --orders 10 --regimes small --models learned_linear \
  --final-pair-ids 0 \
  --output-root narma/results/development

python -m narma.run_benchmark final \
  --order 10 --regime small --pair-id 0 \
  --orders 10 --regimes small --models learned_linear \
  --final-pair-ids 0 \
  --output-root narma/results/development
```

Locking requires selected configurations for the complete model/order/regime
matrix specified on the command line.

## Metrics and statistical unit

After the 200-point washout,

```text
MSE   = mean((prediction - target)^2)
NRMSE = sqrt(MSE / population_variance(target))
```

The primary statistical unit is the complete paired replicate, never an
individual timestep. Aggregation produces:

- mean, sample SD and paired-bootstrap 95% intervals;
- paired NRMSE differences and ratios;
- paired long-minus-small contrasts for every model;
- paired difference-in-differences testing whether the learned-linear model's
  advantage over each control changes with added training data;
- primary-model win counts;
- paired standardized effects;
- two-sided exact sign-flip tests plus exact binomial sign-test sensitivity;
- Holm-adjusted p-values within each order/regime family of seven
  primary-versus-control comparisons, plus a supplementary global Holm value
  across all 56 comparisons.

The single predeclared confirmatory contrast is `learned_linear` versus
`deep_esn_ridge` on NARMA-10/long; other order, regime and interaction tables
are separate secondary analysis families. The sign-flip test requires
symmetric paired differences or label exchangeability. Since model labels are
not randomized, the magnitude-free exact sign test is reported as a sensitivity
analysis; neither should be described as assumption-free merely because runs
are paired.

Bootstrap intervals are unadjusted descriptive 95% intervals. With only ten
pairs, the minimum two-sided exact sign-flip p-value is `2/1024`; consequently
the supplementary Holm corrections across all 56 model contrasts, 32 regime
contrasts, or 28 interaction contrasts cannot reach 0.05 even under unanimous
signs. Emphasize paired effects and intervals, and extend `--final-pair-ids` to
20 or more for a confirmatory global analysis.

Cost reporting separates:

- gradient-trained parameters;
- encoder-only gradient-trained parameters;
- ridge-fitted parameters;
- PCA-fitted coefficients and normalization statistics separately;
- supervised trainable parameters, total data-fitted coefficients, and
  fixed coefficients with their nonzero count;
- total fixed-plus-trainable parameters, total model coefficients, and values
  required for inference;
- ridge standardization values as fit metadata, excluded from inference
  storage because the fitted readout is converted back to raw coordinates;
- recurrent-state and bottleneck dimensions;
- gradient-training and ridge-fit time;
- final-test inference throughput; and
- peak allocated GPU memory.

## Outputs

Workers write only to unique shard directories, avoiding Slurm races:

```text
narma/results/<profile>_benchmark/
├── tuning/order_10/small/learned_linear/
│   ├── trials.json
│   └── selected.json
├── locked_configs.json
├── PROTOCOL.md
├── protocol.json
├── final/order_10/small/pair_00/
│   ├── data_manifest.json
│   ├── learned_linear/
│   │   ├── metrics.json
│   │   └── history.json
│   └── complete.json
├── metrics.csv
├── summary.csv
├── paired_comparisons.csv
├── paired_regime_comparisons.csv
├── paired_difference_in_differences.csv
├── convergence.json
├── completeness.json
└── summary.png
```

Checkpoints and per-timestep predictions are disabled in the Slurm protocol to
keep the result tree compact. The CLI supports `--save-checkpoints` and
`--save-predictions` when those artifacts are needed outside ordinary Git.
