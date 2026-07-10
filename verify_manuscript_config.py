"""Static integrity checks for manuscript/code configuration alignment."""

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent


def import_from_path(name: str, path: Path):
    """Import a Python module from an explicit file path."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Python file: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_file(relative_path: str) -> Path:
    """Require a repository file to exist."""
    path = ROOT / relative_path
    assert path.exists(), f"Missing required file: {relative_path}"
    return path


def main():
    # ------------------------------------------------------------------
    # 1. Cylinder-wake configuration checks
    # ------------------------------------------------------------------
    cfg = import_from_path(
        "manuscript_benchmark_config",
        ROOT / "cylinder_wake" / "benchmark_config.py",
    )

    layers = cfg.LAYER_MAT_PSI
    params = sum(
        layers[i] * layers[i + 1] + layers[i + 1]
        for i in range(len(layers) - 1)
    )

    assert params == 91502, f"Unexpected parameter count: {params}"
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

    # ------------------------------------------------------------------
    # 2. learning_schedule module check
    # ------------------------------------------------------------------
    require_file("cylinder_wake/learning_schedule.py")

    old_typo_path = ROOT / "cylinder_wake" / "learning_schdule.py"
    assert not old_typo_path.exists(), (
        "Found old typo file: cylinder_wake/learning_schdule.py. "
        "Please remove it after renaming to learning_schedule.py."
    )

    scheduler_module = import_from_path(
        "learning_schedule",
        ROOT / "cylinder_wake" / "learning_schedule.py",
    )

    assert hasattr(scheduler_module, "ChainedScheduler"), (
        "learning_schedule.py does not define ChainedScheduler."
    )

    benchmark_train_text = (
        ROOT / "cylinder_wake" / "benchmark_train.py"
    ).read_text(encoding="utf-8")

    assert "from learning_schedule import ChainedScheduler" in benchmark_train_text, (
        "benchmark_train.py should import ChainedScheduler from learning_schedule."
    )

    # ------------------------------------------------------------------
    # 3. NASA hump script token checks
    # ------------------------------------------------------------------
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

    for token in [
        "default=50",
        "default=2.0",
        "--align-cp-offset",
    ]:
        assert token in hv, f"Missing hump validation token: {token}"

    # ------------------------------------------------------------------
    # 4. NASA hump data filename checks
    # ------------------------------------------------------------------
    # Actual repository filenames use ".exp.dat", not "_exp.dat".
    nasa_hump_required_files = [
        "LES_cp_nasahump2009.dat",
        "LES_meanfield_nasahump2009_tec.dat",
        "LES_statistics_profiles_nasahump2009.dat",
        "noflow_cp.exp.dat",
        "noflow_cf.exp.dat",
        "noflow_vel_and_turb.exp.dat",
    ]

    for filename in nasa_hump_required_files:
        require_file(f"nasa_hump/{filename}")

    # Fail clearly if outdated wrong names are still hard-coded in scripts.
    outdated_names = [
        "noflow_cp_exp.dat",
        "noflow_cf_exp.dat",
        "noflow_vel_and_turb_exp.dat",
    ]

    for wrong_name in outdated_names:
        assert wrong_name not in ht, (
            f"Outdated NASA filename in hump_train.py: {wrong_name}"
        )
        assert wrong_name not in hv, (
            f"Outdated NASA filename in hump_validation.py: {wrong_name}"
        )

    print("PASS: key code defaults match the revised manuscript configuration.")
    print("Network parameters:", params)
    print("NASA hump data filenames: OK")
    print("learning_schedule module: OK")
    print(
        "Reminder: rerun training/evaluation before claiming that numerical tables are reproduced."
    )


if __name__ == "__main__":
    main()
