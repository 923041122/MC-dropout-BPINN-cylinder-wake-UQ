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
