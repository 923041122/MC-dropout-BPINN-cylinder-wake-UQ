# MC-dropout B-PINN for Physics-Constrained Flow Reconstruction and UQ

This repository accompanies the revised manuscript:

**Uncertainty Quantification for Physics-Constrained Reconstruction of a High-Reynolds-Number Cylinder Wake Using an MC-Dropout Physics-Informed Neural Network**

The implementation is a physics-constrained reconstruction framework. The cylinder reference fields originate from a k-omega SST CFD dataset, but the PINN loss contains only the dimensionless two-dimensional incompressible momentum residuals as soft regularization; it does **not** contain the k or omega transport equations, SST blending functions, eddy-viscosity closure, or modeled Reynolds stresses.

## Formal manuscript settings

### Cylinder wake

- Input/output: `(x, y, t) -> (psi, p)`; `u = dpsi/dy`, `v = -dpsi/dx`
- Network: 10 hidden layers, 100 neurons per layer, Tanh
- Network trainable parameters: 91,502
- Adam initial learning rate: 0.001
- Epochs: 2,000
- Fixed supervised sampling ratio: 0.02
- Collocation points: 100,000
- Data/physics loss weights: 10 / 1
- Main MC-dropout training rate: 0.002
- Main uncertainty evaluation: 50 stochastic forward passes
- Weight decay baseline coefficient: 1e-5

The `supervised_ratio` selects a fixed 2% labelled subset once per run. `batch_fraction` is a separate memory-management parameter and does not change the labelled-data fraction.

### NASA 2D wall-mounted hump

- Same 10 x 100 network (91,502 network parameters)
- Epochs: 2,000
- Supervised LES velocity points: all 4,761 points (`supervised_ratio=1.0`)
- Momentum-residual collocation points: 10,000
- LES wall-Cp points: 428
- Velocity / physics / Cp loss weights: 10 / 1e-5 / 1
- Reynolds number: 935,892
- Pressure conversion: `Cp = 2p`
- MC-dropout training rate / inference samples: 0.002 / 50
- No fitted constant Cp offset is used by default

## Repository structure

```text
cylinder_wake/
  benchmark_config.py
  benchmark_train.py
  train_dropout_ablation.py
  benchmark_evaluate.py
  uncertainty_ablation.py
  wake_probe_spectrum.py
  benchmark_tools.py
  pinn_model.py
  bayesian_uncertainty_plot.py       # compatibility wrapper
  plot_dimensionless.py              # compatibility wrapper
  train_uv_modify.py                 # compatibility wrapper
nasa_hump/
  hump_train.py
  hump_validation.py
  benchmark_tools.py
  *.dat
verify_manuscript_config.py
```

## Data

Obtain the cylinder dataset `2d_cylinder_Re3900_100x100_kw_sst.mat` from the original public source and place it in the repository root (or pass `--data-path`):

https://github.com/Shengfeng233/PINN-for-turbulence

NASA hump files remain in `nasa_hump/` with these exact names:

- `LES_cp_nasahump2009.dat`
- `LES_meanfield_nasahump2009_tec.dat`
- `LES_statistics_profiles_nasahump2009.dat`
- `noflow_cf_exp.dat`
- `noflow_cp_exp.dat`
- `noflow_vel_and_turb_exp.dat`

## Reproduce the cylinder results

Train the four formal methods only:

```bash
python cylinder_wake/benchmark_train.py --method all \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --epochs 2000 --supervised-ratio 0.02 --batch-fraction 0.02 \
  --n-equation-points 100000 --data-loss-weight 10 \
  --equation-loss-weight 1 --seed 2025
```

`--method all` uses only Standard PINN, Weight Decay PINN, Adaptive-weight PINN, and MC-dropout B-PINN. The experimental Fourier-feature entry is not included in the formal run.

Generate global/local errors, pointwise absolute-error maps, uncertainty maps, calibration curves, and Tables 2-4 CSV outputs:

```bash
python cylinder_wake/benchmark_evaluate.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 50
```

Generate the wake-probe signal and FFT at `(x,y)=(1.500,0.000)`:

```bash
python cylinder_wake/wake_probe_spectrum.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --mc-samples 50
```

### Formal retraining-based dropout ablation

First independently train one checkpoint for every dropout rate:

```bash
python cylinder_wake/train_dropout_ablation.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --dropout-rates 0.002 0.005 0.010 0.020 0.050 \
  --epochs 2000 --supervised-ratio 0.02 --batch-fraction 0.02 \
  --n-equation-points 100000 --data-loss-weight 10 \
  --equation-loss-weight 1 --seed 2025
```

Then evaluate each rate-specific checkpoint at 5, 10, 20, 30, 50, and 100 MC samples:

```bash
python cylinder_wake/uncertainty_ablation.py \
  --data-path ./2d_cylinder_Re3900_100x100_kw_sst.mat \
  --checkpoint-dir benchmark_results/models/dropout_ablation \
  --dropout-rates 0.002 0.005 0.010 0.020 0.050 \
  --mc-samples 5 10 20 30 50 100 --seed 2025
```

`uncertainty_ablation.py` now fails if any rate-specific checkpoint is missing. It never substitutes a new inference probability into one shared checkpoint.

## Reproduce the NASA hump results

```bash
python nasa_hump/hump_train.py --method all \
  --data-dir nasa_hump --epochs 2000 --supervised-ratio 1.0 \
  --n-equation-points 10000 --data-loss-weight 10 \
  --equation-loss-weight 1e-5 --cp-loss-weight 1 \
  --cp-source les --cp-scale 2.0 --seed 2025
```

```bash
python nasa_hump/hump_validation.py \
  --data-dir nasa_hump --mc-samples 50 \
  --pressure-to-cp-scale 2.0 --require-models
```

For the manuscript's post-training hump inference-sensitivity analysis:

```bash
python nasa_hump/hump_validation.py \
  --data-dir nasa_hump --mc-samples 50 --pressure-to-cp-scale 2.0 \
  --require-models --run-ablation \
  --ablation-dropout-rates 0.001 0.002 0.005 0.010 \
  --ablation-mc-samples 10 20 30 50
```

This hump analysis intentionally uses the same checkpoint trained at dropout 0.002 and changes dropout probability only during stochastic inference; it is therefore an inference-sensitivity analysis, not a retraining-based ablation.

## Integrity check

```bash
python verify_manuscript_config.py
```

The script checks the published architecture count and all key default settings. Numerical tables must still be regenerated from the newly trained checkpoints; code alignment alone does not validate previously reported numerical values.

## Parameter-count reporting

All four methods use the same 91,502-parameter neural network. The adaptive-weight baseline additionally optimizes two scalar loss-weight variables for the cylinder case and three for the hump case. For complete transparency, output tables should distinguish **network parameters** from **additional adaptive loss parameters** and **total optimized parameters**.
