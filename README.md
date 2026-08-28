# Backpropagation through fixed reservoirs

Research code for learning compact intermediate representations and autonomous
latent dynamics from fixed recurrent reservoirs. The main experiment packages
are:

- [`narma/`](narma/README.md): the publication-oriented NARMA benchmark,
  including ridge baselines, learned bottlenecks, paired statistics and Slurm
  arrays.
- [`solar/`](solar/README.md): the Copernicus/SciNet physical-representation
  experiment and its analysis jobs.
- [`car_acceleration/`](car_acceleration/README.md): car acceleration and
  rabbit–wolf systems with fixed temporal reservoir encoders and autonomous
  learned latent dynamics.

The research question, exact handoff commands, current evidence, and known
failures are consolidated in [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).

Generic matrix-construction and deterministic runtime helpers live in
[`common/`](common/). The cross-domain publication plan remains in
[`PUBLICATION_ROADMAP.md`](PUBLICATION_ROADMAP.md).

## Setup

Python 3.10 or newer is recommended:

```bash
conda create -n bp-reservoir python=3.11 pip -y
conda activate bp-reservoir
pip install -r requirements.txt
python -m pytest -q
```

Run commands from this repository root so module imports and Slurm submission
paths remain stable.

## NARMA benchmark

Submit the reduced screening workflow with:

```bash
bash narma/jobs/submit_publication_benchmark.sh
```

Run the roughly one-hour, 20-GPU light profile with:

```bash
bash narma/jobs/submit_light_benchmark.sh
```

The default screening profile covers NARMA-5/10/20/30, small and long training
conditions, all eight required baselines, two tuning seeds and three paired
final replicates. Use `NARMA_PROFILE=light` for a roughly one-hour, 20-GPU
pipeline check or `NARMA_PROFILE=publication` for the larger confirmatory
design. See the [NARMA protocol](narma/README.md) before interpreting
cross-order scores: different orders have incompatible definitions in the
literature and are recorded explicitly here.

## Solar experiments

For a quick local check:

```bash
python -m solar.run_experiment --quick --device cpu \
  --output-dir solar/results/solar_quick
```

The cluster jobs are under [`solar/jobs/`](solar/jobs/), including the grid
search, preserved/sequential 10×150 reservoirs and the full SciNet reproduction.

## Repository layout

```text
common/                 shared, domain-neutral reservoir/runtime helpers
narma/
├── data.py             cited NARMA task registry and paired splits
├── models.py           ridge, bottleneck and GRU comparisons
├── benchmark.py        tuning, final evaluation and aggregation
├── statistics.py       pair-level confidence intervals and tests
├── jobs/               dependency-linked Slurm workflow
├── legacy/             original NARMA-10 pilot
└── results/            archived and newly generated NARMA results
solar/
├── data.py
├── models.py
├── experiment.py
├── analysis/
├── jobs/
└── results/            archived solar/SciNet results
car_acceleration/
├── data.py             car and Lotka–Volterra trajectory generators
├── model.py            fixed-reservoir car encoder and affine latent dynamics
├── rabbit_model.py     tapped reservoir and quadratic latent dynamics
└── results/            exploratory physical-latent runs
```

Generated publication-benchmark shards are ignored by Git by default. Keep
large checkpoints and histories in cluster/archive storage; commit compact
protocol files, aggregate tables and plotting code intentionally.

Archived checkpoints and JSON configurations retain their original
`results/...` output-directory strings as provenance. Live defaults and
analysis paths now use `narma/results/` or `solar/results/`; the solar grid
analysis includes a fallback for its relocated archived winner.
