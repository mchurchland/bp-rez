# Backpropagation Through a Deep Reservoir

PyTorch research prototype for learning a low-dimensional intermediate readout
between two fixed recurrent reservoirs. The model is evaluated on deterministic
NARMA10 sequences against four reservoir-computing baselines.

The proposed architecture contains two 150-neuron reservoir networks (300
recurrent nodes total) and a trainable 10-dimensional intermediate
representation. Only the intermediate readout and final output layer are
trained; gradients still propagate through the recurrent dynamics.

![Two-reservoir architecture](figures/two_reservoir_architecture.png)

## Models

All recurrent states are reset to zero at the beginning of each independently
generated train, validation, or test sequence. In code, a time step first
consumes `u[t]`, then emits the prediction paired with NARMA target `y[t+1]`.
This is the usual post-update convention and is the question's recurrence with
indices shifted by one.

The proposed nonlinear model is

```text
x1[t+1] = (1-a)x1[t] + a tanh(A1 x1[t] + B1 u[t])
h[t+1]  = tanh(W x1[t+1] + b)                 # dimension 10
x2[t+1] = (1-a)x2[t] + a tanh(A2 x2[t] + R h[t+1])
yhat[t+1] = Wout x2[t+1] + c
```

Here `a` is the leak rate. `A1`, `A2`, `B1`, and `R` are fixed random tensors,
registered as PyTorch buffers with `requires_grad=False`. Adam receives only
`W`, `b`, `Wout`, and `c`. They remain inside the autograd graph, however, so
full-sequence backpropagation passes through both reservoir networks and through
the recurrent dynamics of the second reservoir. The linear ablation removes
only the `tanh` applied to `h`.

| CLI model | Description | Recurrent nodes | Trainable parameters |
|---|---|---:|---:|
| `single_esn` | Single ESN baseline | 150 | 151 |
| `deep_esn` | 150+150 DeepESN, fixed direct interlayer projection | 300 | 151 |
| `proposed_nonlinear` | 150→10→150 learned nonlinear bottleneck | 300 | 1,661 |
| `proposed_linear` | Same model without intermediate `tanh` | 300 | 1,661 |
| `large_esn` | Single ESN matched to the deep models' total nodes | 300 | 301 |

The distinction between `single_esn` and `large_esn` makes comparison 5 a true
total-node-control for the two-layer networks.

## Results

The full comparison used five random-matrix/model seeds, a common deterministic
NARMA10 dataset, 2,000 training points, 500 validation points, 500 test points,
100 washout steps, and 200 Adam epochs. Values are test-set mean ± sample
standard deviation across seeds.

| Model | Test MSE ↓ | Test NRMSE ↓ | Trainable parameters |
|---|---:|---:|---:|
| Single ESN | 0.00683 ± 0.00108 | 0.673 ± 0.053 | 151 |
| Fixed-interlayer DeepESN | 0.00676 ± 0.00140 | 0.669 ± 0.067 | 151 |
| Proposed nonlinear | **0.00352 ± 0.00053** | **0.484 ± 0.037** | 1,661 |
| Proposed linear | **0.00345 ± 0.00046** | **0.479 ± 0.032** | 1,661 |
| Large 300-node ESN | 0.00450 ± 0.00035 | 0.548 ± 0.022 | 301 |

Against the fixed-interlayer DeepESN, the nonlinear proposal reduced mean MSE
by approximately 48% and mean NRMSE by 28%. Against the total-node-matched large
ESN, it reduced MSE by approximately 22% and NRMSE by 12%.

The linear intermediate ablation was marginally better than the nonlinear
version, although their seed-level results were very close. The current evidence
therefore supports the benefit of training the low-dimensional intermediate
mapping, but does not demonstrate an additional benefit from applying `tanh` in
that intermediate space.

All configurations selected epoch 200, the maximum tested epoch. These results
are an equal-budget comparison rather than evidence that every optimizer run had
fully converged. The proposed models also use more trainable parameters than the
baselines, which should be considered when interpreting the accuracy gain.

## Secondary experiment: does the latent become heliocentric?

`run_solar_experiment.py` adapts the solar-system experiment from
[Iten et al., *Discovering physical concepts with neural networks*](https://arxiv.org/abs/1807.10300)
to the two-reservoir architecture. From only the initial Earth-view angles of
the Sun and Mars, the model predicts 50 weekly observations through a learned
latent state constrained to evolve as `z[t+1] = z[t] + delta`. The default
two-dimensional bottleneck matches the paper. A PyTorch `scinet` reference is
included as a protocol check and follows the released TensorFlow graph's active
behavior, including its overwritten log-sigma clip and unused regularized Euler
matrix.

The experiment tests two separate claims:

1. prediction: test RMSE divided by `2*pi` (the paper reports below 0.4%); and
2. representation: held-out linear fits on the paper's angle-unwrapped
   heliocentric chart between the learned latent and the unobserved Earth/Mars
   angles, plus agreement between the learned latent update and the update
   implied by that fit.

A short end-to-end development run is:

```bash
python run_solar_experiment.py --quick --device cpu \
  --output-dir results/solar_quick
```

Run the five-phase, 15,000-update schedule with the paper's dataset sizes using:

```bash
python run_solar_experiment.py \
  --models reservoir scinet --seeds 0 1 2 3 4 \
  --latent-size 2 --nodes-1 150 --nodes-2 150 \
  --output-dir results/solar_replication
```

Submit the equivalent GPU job with `sbatch run_solar_experiment.sbatch`. The
original public TensorFlow code interpreted the 15,000 phase counts as complete
dataset passes, which is millions of optimizer updates. Add
`--full-dataset-epochs` for that literal reference-model reproduction; it is not
the practical default. Architecture, sampling, metrics, artifacts, and known
replication differences are detailed in [SOLAR_EXPERIMENT.md](SOLAR_EXPERIMENT.md).

## Setup

Python 3.10 or newer is recommended. Create a Conda environment and install the
dependencies into it:

```bash
conda create -n bp-reservoir python=3.11 pip -y
conda activate bp-reservoir
pip install -r requirements.txt
python -m pytest -q
```

CUDA is selected automatically when available. Force a device with `--device
cpu`, `--device cuda`, or (for example) `--device cuda:1`.

## Reproduce the full five-seed comparison

The defaults specify 150 nodes per layer, 300 total nodes, a 10-dimensional
latent space, and five model/random-matrix seeds. Run:

```bash
python run_experiments.py \
  --seeds 0 1 2 3 4 \
  --models single_esn deep_esn proposed_nonlinear proposed_linear large_esn \
  --nodes-1 150 --nodes-2 150 --latent-size 10 \
  --train-length 2000 --val-length 500 --test-length 500 \
  --washout 100 --spectral-radius 0.9 --density 0.1 \
  --leak-rate 1.0 --input-scale 0.5 --interlayer-scale 1.0 \
  --learning-rate 1e-3 --max-epochs 200 --patience 25 \
  --grad-clip 1.0 --data-seed 2026 \
  --output-dir results/narma10_300_nodes
```

On a Slurm cluster, submit the included batch script from the project root:

```bash
sbatch run_experiments.sbatch
```

The batch script activates the `bp-reservoir` Conda environment. To use a
different existing environment, submit with:

```bash
sbatch --export=ALL,CONDA_ENV=my-environment run_experiments.sbatch
```

Monitor it with `squeue -u "$USER"` and follow its combined output/error log
with `tail -f slurm-narma10-reservoir-JOBID.out`. The batch script requests one
GPU, eight CPU cores, 16 GB RAM, and 24 hours. Adjust its `#SBATCH` lines if
your cluster uses different partitions, account names, GPU syntax, or limits.

This uses full-sequence BPTT, not a detached state cache or a ridge-regression
shortcut. It is consequently slower than a conventional closed-form ESN,
especially on CPU. A quick end-to-end check is:

```bash
python run_experiments.py \
  --seeds 0 1 2 3 4 \
  --train-length 80 --val-length 40 --test-length 40 \
  --washout 10 --max-epochs 2 --patience 2 \
  --output-dir results/smoke_5_seeds
```

Run one model or change optimization settings in the same way:

```bash
python run_experiments.py --models proposed_nonlinear --device cpu \
  --leak-rate 0.3 --spectral-radius 0.95 --input-scale 0.2 \
  --grad-clip 0.5 --weight-decay 1e-6
```

Use `python run_experiments.py --help` for every option. To protect the stated
study design, the runner rejects configurations where the two layers do not sum
to 300 nodes or where the latent size is not 10.

## Data and metrics

Train, validation, and test inputs are independent sequences drawn from
`Uniform(0, 0.5)` using child seeds derived from `--data-seed`. They are shared
by every model seed for paired comparisons. NARMA10 is generated in float64 and
then stored as float32. Model initialization is deterministic for each model
seed. The first `--washout` samples are excluded from every loss and metric.

Early stopping monitors validation MSE and restores the best checkpoint.
Reported test metrics are

```text
MSE   = mean((yhat - y)^2)
NRMSE = sqrt(MSE / population_variance(y))
```

The aggregate standard deviation is the sample standard deviation across model
seeds. Parameter counts include only tensors optimized by Adam.

## Outputs

Each invocation writes incrementally, so completed runs survive an interrupted
job:

```text
results/narma10_300_nodes/
├── config.json
├── metrics.csv                 # one row per model and seed
├── metrics.json
├── summary.csv                 # mean and SD across seeds
├── summary.json
├── summary.png                 # aggregate NRMSE plot
└── proposed_nonlinear/seed_0/
    ├── checkpoint.pt           # config, fixed buffers, trained tensors, metrics
    ├── history.json
    ├── metrics.json
    ├── predictions.npz         # input, target, prediction, washout
    ├── predictions.png
    └── training.png
```

The other model/seed directories have the same layout. The console and summary
files report MSE, NRMSE, trainable parameter count, best epoch, and runtime.

## Implementation map

- `reservoir/data.py`: deterministic NARMA10 generation and splits.
- `reservoir/models.py`: reservoir matrices and all five variants.
- `reservoir/experiment.py`: Adam/BPTT, clipping, early stopping, metrics, and artifacts.
- `reservoir/solar_data.py`: circular Earth/Mars simulation and Earth-view observations.
- `reservoir/solar_models.py`: reservoir adaptation and PyTorch SciNet reference.
- `reservoir/solar_experiment.py`: phased training and heliocentric latent analysis.
- `run_experiments.py`: command-line interface.
- `run_solar_experiment.py`: secondary Copernicus experiment command-line interface.
- `plot_architecture.py`: configurable neuron-level PNG/SVG/PDF diagram. Its
  default is a five-reservoir extension with 60 neurons per reservoir and four
  10-neuron latent stages. Run `python plot_architecture.py`, or reproduce the
  original two-reservoir layout with `python plot_architecture.py --reservoirs
  2 --output figures/two_reservoir_architecture.svg`.
- `tests/test_prototype.py`: determinism, spectral scaling, fixed-buffer,
  BPTT-gradient, parameter-count, and artifact tests.
