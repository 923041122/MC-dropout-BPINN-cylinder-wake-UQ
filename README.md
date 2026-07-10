# MC-dropout B-PINN for Physics-Constrained Flow Reconstruction and UQ

This repository accompanies the revised manuscript:

**Uncertainty Quantification for Physics-Constrained Reconstruction of a High-Reynolds-Number Cylinder Wake Using an MC-Dropout Physics-Informed Neural Network**

This repository provides the implementation used for physics-constrained flow-field reconstruction and conditional uncertainty quantification. The cylinder-wake reference fields originate from a publicly available k-omega SST CFD dataset, but the PINN loss used here contains only the dimensionless two-dimensional incompressible momentum residuals as soft physical regularization. It does **not** include the k or omega transport equations, SST blending functions, eddy-viscosity closure, or modeled Reynolds-stress terms.

The reported MC-dropout uncertainty should therefore be interpreted as an approximate conditional reconstruction-reliability indicator associated with the trained model, available reference data, training procedure, model capacity, and dropout approximation. It is not a complete representation of all physical uncertainty sources in turbulent flow.

---

## Formal manuscript settings

### Cylinder wake

- Input/output: `(x, y, t) -> (psi, p)`
- Velocity recovery: `u = dpsi/dy`, `v = -dpsi/dx`
- Network: 10 hidden layers, 100 neurons per layer, Tanh activation
- Network trainable parameters: 91,502
- Optimizer: Adam
- Initial learning rate: 0.001
- Epochs: 2,000
- Fixed supervised sampling ratio: 0.02
- Collocation points: 100,000
- Data / physics loss weights: 10 / 1
- Main MC-dropout training rate: 0.002
- Main uncertainty evaluation: 50 stochastic forward passes
- Weight decay baseline coefficient: 1e-5

The `supervised_ratio` selects a fixed labelled subset once per run. The `batch_fraction` argument is a memory-management parameter and does not change the labelled-data fraction.

### NASA 2D wall-mounted hump

- Same 10 x 100 network architecture
- Network trainable parameters: 91,502
- Epochs: 2,000
- Supervised LES velocity points: all 4,761 points (`supervised_ratio=1.0`)
- Momentum-residual collocation points: 10,000
- LES wall-pressure coefficient points: 428
- Velocity / physics / Cp loss weights: 10 / 1e-5 / 1
- Reynolds number: 935,892
- Pressure conversion: `Cp = 2p`
- MC-dropout training rate / inference samples: 0.002 / 50
- No fitted constant Cp offset is used by default

---

## Repository structure

```text
cylinder_wake/
  benchmark_config.py
  benchmark_train.py
  benchmark_evaluate.py
  learning_schedule.py
  train_dropout_ablation.py
  uncertainty_ablation.py
  wake_probe_spectrum.py
  benchmark_tools.py
  pinn_model.py
  read_data.py
  bayesian_uncertainty_plot.py       # compatibility wrapper
  plot_dimensionless.py              # compatibility wrapper
  train_uv_modify.py                 # compatibility wrapper

nasa_hump/
  hump_train.py
  hump_validation.py
  hump_heldout_evaluate.py
  hump_repeated_timing.py
  benchmark_tools.py
  LES_cp_nasahump2009.dat
  LES_meanfield_nasahump2009_tec.dat
  LES_statistics_profiles_nasahump2009.dat
  noflow_cf.exp.dat
  noflow_cp.exp.dat
  noflow_vel_and_turb.exp.dat

verify_manuscript_config.py
requirements.txt
LICENSE
README.md
```

---

## Installation

A Python environment with PyTorch is required. One possible setup is:

```bash
conda create -n bpinn_uq python=3.10 -y
conda activate bpinn_uq
pip install -r requirements.txt
```

The tested scripts use standard scientific Python packages, including PyTorch, NumPy, SciPy, pandas, matplotlib, scikit-learn, and tqdm.

---

## Data

### Cylinder wake

The cylinder-wake dataset is not redistributed as part of this repository. Please obtain:

```text
2d_cylinder_Re3900_100x100_kw_sst.mat
```

from the original public source:

```text
https://github.com/Shengfeng233/PINN-for-turbulence
```

The file can be placed either in the repository root:

```text
./2d_cylinder_Re3900_100x100_kw_sst.mat
```

or inside the cylinder-wake folder:

```text
./cylinder_wake/2d_cylinder_Re3900_100x100_kw_sst.mat
```

Use the corresponding `--data-path` argument when running the scripts.

### NASA hump

The NASA hump files are included under `nasa_hump/` with the following exact filenames:

```text
LES_cp_nasahump2009.dat
LES_meanfield_nasahump2009_tec.dat
LES_statistics_profiles_nasahump2009.dat
noflow_cf.exp.dat
noflow_cp.exp.dat
noflow_vel_and_turb.exp.dat
```

Note that the experimental no-flow files use `.exp.dat`, not `_exp.dat`.

---

## Integrity check

Before training, run:

```bash
python verify_manuscript_config.py
```

Expected output:

```text
PASS: key code defaults match the revised manuscript configuration.
Network parameters: 91502
NASA hump data filenames: OK
learning_schedule module: OK
Reminder: rerun training/evaluation before claiming that numerical tables are reproduced.
```

This script checks the published network size, key default settings, NASA hump filenames, and the `learning_schedule.py` module. It is a configuration-alignment check only. Numerical tables must still be regenerated from the trained checkpoints.

---

## Reproduce the cylinder-wake results

If the cylinder dataset is placed in the repository root, use:

```bash
python cylinder_wake/benchmark_train.py --method all \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --epochs 2000 \
  --supervised-ratio 0.02 \
  --batch-fraction 0.02 \
  --n-equation-points 100000 \
  --data-loss-weight 10 \
  --equation-loss-weight 1 \
  --seed 2025
```

If the cylinder dataset is placed in `cylinder_wake/`, use:

```bash
python cylinder_wake/benchmark_train.py --method all \
  --data-path ./cylinder_wake/2d_cylinder_Re3900_100x100_kw_sst.mat \
  --epochs 2000 \
  --supervised-ratio 0.02 \
  --batch-fraction 0.02 \
  --n-equation-points 100000 \
  --data-loss-weight 10 \
  --equation-loss-weight 1 \
  --seed 2025
```

The command trains the four formal methods:

```text
standard_pinn
weight_decay_pinn
adaptive_weight_pinn
bpinn_dropout
```

The experimental Fourier-feature entry, if present in development code, is not included in the formal manuscript run.

Generate global errors, local-region errors, pointwise absolute-error maps, uncertainty maps, calibration curves, and CSV outputs corresponding to the cylinder-wake evaluation:

```bash
python cylinder_wake/benchmark_evaluate.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 50
```

or, if the data file is inside `cylinder_wake/`:

```bash
python cylinder_wake/benchmark_evaluate.py \
  --data-path ./cylinder_wake/2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 50
```

Generate the wake-probe signal and FFT at `(x, y) = (1.500, 0.000)`:

```bash
python cylinder_wake/wake_probe_spectrum.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 50
```

---

## Cylinder-wake dropout ablation

First train one independent checkpoint for each dropout rate:

```bash
python cylinder_wake/train_dropout_ablation.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --dropout-rates 0.002 0.005 0.010 0.020 0.050 \
  --epochs 2000 \
  --supervised-ratio 0.02 \
  --batch-fraction 0.02 \
  --n-equation-points 100000 \
  --data-loss-weight 10 \
  --equation-loss-weight 1 \
  --seed 2025
```

Then evaluate each rate-specific checkpoint at 5, 10, 20, 30, 50, and 100 MC samples:

```bash
python cylinder_wake/uncertainty_ablation.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --checkpoint-dir benchmark_results/models/dropout_ablation \
  --dropout-rates 0.002 0.005 0.010 0.020 0.050 \
  --mc-samples 5 10 20 30 50 100 \
  --seed 2025
```

`uncertainty_ablation.py` fails if any rate-specific checkpoint is missing. It does not substitute a new inference dropout probability into one shared checkpoint for the formal cylinder-wake ablation.

---

## Reproduce the NASA hump results

Train all four methods:

```bash
python nasa_hump/hump_train.py --method all \
  --data-dir nasa_hump \
  --epochs 2000 \
  --supervised-ratio 1.0 \
  --n-equation-points 10000 \
  --data-loss-weight 10 \
  --equation-loss-weight 1e-5 \
  --cp-loss-weight 1 \
  --cp-source les \
  --cp-scale 2.0 \
  --seed 2025
```

Evaluate the trained models:

```bash
python nasa_hump/hump_validation.py \
  --data-dir nasa_hump \
  --mc-samples 50 \
  --pressure-to-cp-scale 2.0 \
  --require-models
```

For the manuscript's post-training hump inference-sensitivity analysis:

```bash
python nasa_hump/hump_validation.py \
  --data-dir nasa_hump \
  --mc-samples 50 \
  --pressure-to-cp-scale 2.0 \
  --require-models \
  --run-ablation \
  --ablation-dropout-rates 0.001 0.002 0.005 0.010 \
  --ablation-mc-samples 10 20 30 50
```

This hump analysis intentionally uses the same checkpoint trained at dropout 0.002 and changes the dropout probability only during stochastic inference. It is therefore an inference-sensitivity analysis, not a retraining-based ablation.

---

## Lightweight sanity check

The following commands are intended only to verify that the code can run, load data, train models, save checkpoints, and export CSV/figure outputs. They are not used to reproduce the manuscript numerical values.

### Static check

```bash
python verify_manuscript_config.py
```

### NASA hump quick run

```bash
rm -rf hump_results

python nasa_hump/hump_train.py --method all \
  --data-dir nasa_hump \
  --epochs 1 \
  --supervised-ratio 1.0 \
  --n-equation-points 200 \
  --data-loss-weight 10 \
  --equation-loss-weight 1e-5 \
  --cp-loss-weight 1 \
  --cp-source les \
  --cp-scale 2.0 \
  --seed 2025

python nasa_hump/hump_validation.py \
  --data-dir nasa_hump \
  --models-root hump_results/models \
  --results-root ./hump_results \
  --mc-samples 5
```

### Cylinder-wake quick run

If the cylinder data are placed in `cylinder_wake/`, run:

```bash
rm -rf benchmark_results

python cylinder_wake/benchmark_train.py --method all \
  --data-path ./cylinder_wake/2d_cylinder_Re3900_100x100_kw_sst.mat \
  --epochs 1 \
  --supervised-ratio 0.02 \
  --batch-fraction 0.02 \
  --n-equation-points 500 \
  --data-loss-weight 10 \
  --equation-loss-weight 1 \
  --seed 2025 \
  --disable-diagnostics

python cylinder_wake/benchmark_evaluate.py \
  --data-path ./cylinder_wake/2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 5
```

A successful lightweight run should generate files under:

```text
benchmark_results/
hump_results/
```

including model checkpoints, training logs, evaluation CSV files, and figures.

---

## Output directories

Typical outputs are written to:

```text
benchmark_results/models/
benchmark_results/training_logs/
benchmark_results/evaluation/

hump_results/models/
hump_results/training_logs/
hump_results/hump_evaluation/tables/
hump_results/hump_evaluation/figures/
```

These generated result folders are not required to be version-controlled unless explicitly needed for archival purposes.

---

## Parameter-count reporting

All four methods use the same 91,502-parameter neural network. The adaptive-weight baseline additionally optimizes scalar loss-weight variables. For complete transparency, output tables should distinguish:

```text
network parameters
additional adaptive loss parameters
total optimized parameters
```

This distinction avoids mixing the neural-network parameter count with trainable loss-weight variables.

---

## Notes on reproducibility

- The full 2,000-epoch training runs are computationally more expensive than the lightweight sanity checks.
- The lightweight commands verify code execution and output generation only.
- Exact numerical values may vary slightly with hardware, PyTorch version, CUDA behavior, random seed handling, and floating-point nondeterminism.
- The manuscript tables should be regenerated from the formal 2,000-epoch runs before claiming numerical reproduction.
