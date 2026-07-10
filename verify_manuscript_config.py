"""Static integrity checks for manuscript/code configuration alignment."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cfg = import_from_path(
        "manuscript_benchmark_config",
        ROOT / "cylinder_wake" / "benchmark_config.py",
    )
    layers = cfg.LAYER_MAT_PSI
    params = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
    assert params == 91502, params
    assert cfg.REYNOLDS == 3900.0
    assert cfg.DEFAULT_TRAINING["epochs"] == 2000
    assert cfg.DEFAULT_TRAINING["supervised_ratio"] == 0.02
    assert cfg.DEFAULT_TRAINING["n_equation_points"] == 100000
    assert cfg.METHODS["bpinn_dropout"]["dropout_rate"] == 0.002
    assert cfg.DEFAULT_EVAL["mc_samples"] == 50
    assert cfg.DEFAULT_EVAL["dropout_rates"] == [0.002, 0.005, 0.01, 0.02, 0.05]
    assert cfg.DEFAULT_EVAL["mc_sample_grid"] == [5, 10, 20, 30, 50, 100]
    assert cfg.ACTIVE_METHODS == [
        "standard_pinn",
        "weight_decay_pinn",
        "adaptive_weight_pinn",
        "bpinn_dropout",
    ]

    ht = (ROOT / "nasa_hump" / "hump_train.py").read_text(encoding="utf-8")
    hv = (ROOT / "nasa_hump" / "hump_validation.py").read_text(encoding="utf-8")
    for token in [
        "default=10000",
        "default=1e-5",
        "default=1.0",
        "default=2.0",
        "RE_HUMP = 935_892.0",
    ]:
        assert token in ht, f"Missing hump training token: {token}"
    for token in ["default=50", "default=2.0", "--align-cp-offset"]:
        assert token in hv, f"Missing hump validation token: {token}"
    for filename in [
        "noflow_cp_exp.dat",
        "noflow_cf_exp.dat",
        "noflow_vel_and_turb_exp.dat",
    ]:
        assert filename in ht or filename in hv

    print("PASS: key code defaults match the revised manuscript configuration.")
    print("Network parameters:", params)
    print("Reminder: rerun training/evaluation before claiming that numerical tables are reproduced.")


if __name__ == "__main__":
    main()
