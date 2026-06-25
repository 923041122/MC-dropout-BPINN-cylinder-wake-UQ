# MC-dropout B-PINN for Physics-Constrained Flow Reconstruction and Uncertainty Quantification

This repository contains the code and processed validation data used in the manuscript:

**Uncertainty Quantification for Physics-Constrained Reconstruction of a High-Reynolds-Number Cylinder Wake Using an MC-Dropout Physics-Informed Neural Network**

The repository provides implementations for physics-constrained flow-field reconstruction and uncertainty quantification using an MC-dropout Bayesian physics-informed neural network (MC-dropout B-PINN). Two separated-flow cases are considered:

1. A two-dimensional high-Reynolds-number cylinder wake at $Re = 3900$
2. A two-dimensional NASA wall-mounted hump separated-flow case

The code includes deterministic PINN baselines, MC-dropout uncertainty estimation, calibration analysis, error evaluation, and plotting scripts.

---

## Repository structure

```text
MC-dropout-BPINN-cylinder-wake-UQ/
├── README.md
├── requirements.txt
├── LICENSE
├── cylinder_wake/
│   ├── bayesian_uncertainty_plot.py
│   ├── benchmark_config.py
│   ├── benchmark_evaluate.py
│   ├── benchmark_tools.py
│   ├── benchmark_train.py
│   ├── learning_schedule.py
│   ├── pinn_model.py
│   ├── plot_dimensionless.py
│   ├── read_data.py
│   ├── train_uv_modify.py
│   └── uncertainty_ablation.py
└── nasa_hump/
    ├── benchmark_tools.py
    ├── hump_train.py
    ├── hump_validation.py
    ├── LES_cp_nasahump2009.dat
    ├── LES_meanfield_nasahump2009_tec.dat
    ├── LES_statistics_profiles_nasahump2009.dat
    ├── noflow_cf_exp.dat
    ├── noflow_cp_exp.dat
    └── noflow_vel_and_turb_exp.dat
```

---

## Main components

### 1. Cylinder-wake case

The `cylinder_wake/` folder contains the implementation for the two-dimensional cylinder-wake reconstruction case at $Re = 3900$.

The cylinder-wake reference data used in this study were obtained from the publicly available GitHub repository:

https://github.com/Shengfeng233/PINN-for-turbulence

The original repository provides the two-dimensional circular-cylinder wake dataset calculated using a $k$-$\omega$ SST model at $Re = 3900$. Users should obtain the original cylinder-wake data from the source repository and cite the original repository and associated references when using the data.

Due to redistribution and attribution considerations, the original cylinder-wake `.mat` data file is not redistributed in this repository. The present repository provides the scripts used for physics-constrained reconstruction, uncertainty estimation, calibration analysis, and post-processing.

Main files:

* `pinn_model.py`
  Defines the neural-network architecture and PINN model.

* `benchmark_config.py`
  Provides configuration settings for model training and evaluation.

* `benchmark_train.py`
  Trains deterministic PINN baselines.

* `train_uv_modify.py`
  Training script used for velocity and pressure reconstruction.

* `benchmark_evaluate.py`
  Computes reconstruction errors and benchmark metrics.

* `bayesian_uncertainty_plot.py`
  Generates MC-dropout uncertainty maps and related plots.

* `uncertainty_ablation.py`
  Performs uncertainty-related ablation analysis.

* `plot_dimensionless.py`
  Generates dimensionless flow-field and error plots.

* `read_data.py`
  Reads and preprocesses the cylinder-wake reference data.

* `learning_schedule.py`
  Defines the learning-rate schedule used during training.

* `benchmark_tools.py`
  Provides auxiliary functions for training, evaluation, and post-processing.

---

### 2. NASA wall-mounted hump case

The `nasa_hump/` folder contains the implementation and processed reference data for the two-dimensional NASA wall-mounted hump separated-flow case.

The NASA wall-mounted hump validation data were obtained from the NASA Turbulence Modeling Resource:

https://tmbwg.github.io/turbmodels/nasahump_val.html

Users should cite the NASA Turbulence Modeling Resource and the associated experimental references when using these data.

Main files:

* `hump_train.py`
  Trains the PINN models for the wall-mounted hump case.

* `hump_validation.py`
  Evaluates wall-pressure coefficient and velocity-profile predictions.

* `benchmark_tools.py`
  Provides auxiliary functions for the hump-case evaluation.

Reference data files:

* `LES_cp_nasahump2009.dat`
* `LES_meanfield_nasahump2009_tec.dat`
* `LES_statistics_profiles_nasahump2009.dat`
* `noflow_cf_exp.dat`
* `noflow_cp_exp.dat`
* `noflow_vel_and_turb_exp.dat`

These files contain processed LES and experimental reference data used for wall-pressure and velocity-profile validation.

---

## Models included

The repository includes the following PINN variants:

* Standard PINN
* Weight Decay PINN
* Adaptive-weight PINN
* MC-dropout B-PINN

The MC-dropout B-PINN uses dropout-based stochastic forward passes during inference to estimate conditional predictive uncertainty.

---

## Requirements

The code was developed in Python. The main required packages are:

```text
numpy
scipy
matplotlib
pandas
torch
scikit-learn
tqdm
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Usage

Example commands for the cylinder-wake case:

```bash
python cylinder_wake/benchmark_train.py
python cylinder_wake/benchmark_evaluate.py
python cylinder_wake/bayesian_uncertainty_plot.py
python cylinder_wake/uncertainty_ablation.py
```

Example commands for the NASA wall-mounted hump case:

```bash
python nasa_hump/hump_train.py
python nasa_hump/hump_validation.py
```

Depending on the local environment, file paths in the scripts may need to be adjusted to match the repository directory structure.

For the cylinder-wake case, users should first obtain the original `.mat` data file from:

https://github.com/Shengfeng233/PINN-for-turbulence

and place it in the local data path expected by the corresponding scripts.

---

## Data

### Cylinder-wake data

The cylinder-wake reference data are not redistributed in this repository. The data were obtained from the following publicly available source repository:

https://github.com/Shengfeng233/PINN-for-turbulence

Users should obtain the data from the original source and cite the original repository and associated references.

### NASA wall-mounted hump data

The processed LES and experimental reference data for the NASA wall-mounted hump case are provided in the `nasa_hump/` folder.

The original validation data source is the NASA Turbulence Modeling Resource:

https://tmbwg.github.io/turbmodels/nasahump_val.html

These data are used for wall-pressure coefficient comparison and velocity-profile validation.

---

## Notes on uncertainty interpretation

The MC-dropout uncertainty reported by this code should be interpreted as a conditional reconstruction-reliability indicator associated with the trained neural-network model, available reference data, training procedure, and dropout approximation.

It should not be interpreted as a complete representation of all physical uncertainty sources in the turbulent flow.

---

## Reproducibility

The main training settings follow those reported in the manuscript, including:

* Network size: 10 hidden layers with 100 neurons per layer
* Activation function: hyperbolic tangent
* Optimizer: Adam
* Data-loss weight: $\lambda_d = 10$
* Physics-loss weight: $\lambda_f = 1$
* Dropout rate for MC-dropout B-PINN: 0.002
* Number of Monte Carlo samples for uncertainty evaluation: 50

The same network size is used for the deterministic PINN baselines and the MC-dropout B-PINN.

---

## Data availability statement

The implementation used for the MC-dropout B-PINN, deterministic PINN baselines, uncertainty-calibration metrics, and plotting scripts is publicly available in this repository.

The cylinder-wake reference data were obtained from the publicly available GitHub repository:

https://github.com/Shengfeng233/PINN-for-turbulence

Due to redistribution and attribution considerations, the original cylinder-wake `.mat` data file is not redistributed in this repository and should be accessed from the original source.

The NASA wall-mounted hump validation data were obtained from the NASA Turbulence Modeling Resource:

https://tmbwg.github.io/turbmodels/nasahump_val.html

The processed NASA wall-mounted hump data used for validation are provided in the `nasa_hump/` folder.

---

## Citation

If you use this code or data, please cite the associated manuscript:

```text
Zhu, L. and Zhang, X.,
Uncertainty Quantification for Physics-Constrained Reconstruction of a High-Reynolds-Number Cylinder Wake Using an MC-Dropout Physics-Informed Neural Network.
```

Please also cite the original data sources when using the cylinder-wake dataset or the NASA wall-mounted hump validation data.

---

## License

This repository is released under the MIT License.

