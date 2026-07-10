"""Evaluate NASA wall-mounted hump models on an exact held-out velocity split.

Place this file in the repository's ``nasa_hump`` directory and run it from the
repository root. The script reconstructs the exact pandas split used by
``hump_train.py --supervised-ratio <ratio> --seed <seed>`` and evaluates only
the complementary (unseen) LES velocity points.

In addition to global error and uncertainty-calibration tables, this revised
version writes a combined pointwise CSV:

    heldout_evaluation/heldout_pointwise_predictions.csv

Each row corresponds to one held-out LES velocity point and contains the
reference values, every loaded model's pointwise predictions and absolute
errors, and MC-dropout B-PINN pointwise mean/std/95% intervals when available.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, MutableMapping

import numpy as np
import pandas as pd

from benchmark_tools import get_device
from hump_train import read_les_meanfield_tec
from hump_validation import (
    ACTIVE_HUMP_METHODS,
    HUMP_METHODS,
    load_hump_models,
    mc_dropout_stats,
    predict_on_points,
    uq_metric_rows,
)


METHOD_PREFIXES = {
    "standard_pinn": "standard",
    "weight_decay_pinn": "weight_decay",
    "adaptive_weight_pinn": "adaptive_weight",
    "bpinn_dropout": "bpinn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NASA hump models on held-out LES velocity points."
    )
    parser.add_argument("--data-dir", default="./nasa_hump")
    parser.add_argument("--models-root", default="./hump_results_heldout80/models")
    parser.add_argument("--results-root", default="./hump_results_heldout80")
    parser.add_argument("--methods", nargs="+", default=ACTIVE_HUMP_METHODS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument(
        "--uq-levels",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.95],
        help=(
            "Nominal central prediction-interval levels used for UQ metrics. "
            "The pointwise output always also includes 95%% intervals."
        ),
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.002,
        help=(
            "Dropout rate used to reconstruct the trained bpinn_dropout model "
            "during evaluation. This must match the rate used during training, "
            "because dropout probability is not stored in the state_dict."
        ),
    )
    parser.add_argument("--skip-uq", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be strictly between 0 and 1.")

    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1.")

    if args.mc_samples < 1:
        raise ValueError("--mc-samples must be at least 1.")

    if not (0.0 < args.dropout_rate < 1.0):
        raise ValueError("--dropout-rate must be strictly between 0 and 1.")

    bad_levels = [level for level in args.uq_levels if not (0.0 < level < 1.0)]
    if bad_levels:
        raise ValueError(f"All --uq-levels must be in (0, 1), got {bad_levels}")


def method_label(method: str) -> str:
    return HUMP_METHODS.get(method, {}).get("label", method)


def method_prefix(method: str) -> str:
    """Return the column prefix used in the combined pointwise CSV."""
    if method in METHOD_PREFIXES:
        return METHOD_PREFIXES[method]
    return (
        method.replace("_pinn", "")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


def metric_dict(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]

    if truth.size == 0:
        return {
            "relative_l2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "points": 0,
        }

    error = pred - truth
    denominator = np.linalg.norm(truth)
    relative_l2 = (
        np.linalg.norm(error) / denominator if denominator > 0 else np.nan
    )

    return {
        "relative_l2": float(relative_l2),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "points": int(truth.size),
    }


def make_exact_split(
    meanfield: Dict[str, object],
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstruct the exact supervised/held-out split used by hump_train.py.

    The original integer index is preserved before resetting the returned
    DataFrames. This is important because the pointwise prediction CSV must be
    traceable back to the original LES mean-field table.
    """
    flat = meanfield["flat"].copy()
    required_columns = ["x", "y", "u", "v"]
    missing = [col for col in required_columns if col not in flat.columns]
    if missing:
        raise KeyError(f"meanfield['flat'] is missing required columns: {missing}")

    # Keep t when available; otherwise add a constant pseudo-time column so the
    # output schema is stable.
    if "t" not in flat.columns:
        flat["t"] = 0.0

    full = flat[["x", "y", "t", "u", "v"]].copy()
    full = full.replace([np.inf, -np.inf], np.nan).dropna()
    full.insert(0, "original_index", full.index.astype(int))

    # This matches hump_train.build_supervised_uv_data().
    train_index = full.sample(frac=train_ratio, random_state=seed).index

    full["split"] = "heldout_test"
    full.loc[train_index, "split"] = "train"

    train_df = full.loc[train_index].sort_index().copy()
    test_df = full.drop(index=train_index).sort_index().copy()
    manifest = full.sort_index().reset_index(drop=True)

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        manifest,
    )


def prediction_array(
    preds: MutableMapping[str, np.ndarray],
    quantity: str,
    expected_points: int,
    method: str,
) -> np.ndarray:
    values = np.asarray(preds[quantity], dtype=float).reshape(-1)
    if values.size != expected_points:
        raise ValueError(
            f"{method} produced {values.size} {quantity}-predictions, "
            f"but {expected_points} held-out points were expected."
        )
    return values


def build_pointwise_table(test_df: pd.DataFrame) -> pd.DataFrame:
    """Create the base held-out pointwise table."""
    required = ["original_index", "x", "y", "t", "split", "u", "v"]
    missing = [col for col in required if col not in test_df.columns]
    if missing:
        raise KeyError(f"test_df is missing required columns: {missing}")

    return pd.DataFrame(
        {
            "original_index": test_df["original_index"].to_numpy(dtype=int),
            "x": test_df["x"].to_numpy(dtype=float),
            "y": test_df["y"].to_numpy(dtype=float),
            "t": test_df["t"].to_numpy(dtype=float),
            "split": test_df["split"].to_numpy(dtype=object),
            "u_ref": test_df["u"].to_numpy(dtype=float),
            "v_ref": test_df["v"].to_numpy(dtype=float),
        }
    )


def add_deterministic_pointwise_columns(
    pointwise_df: pd.DataFrame,
    method: str,
    preds: MutableMapping[str, np.ndarray],
) -> None:
    """Append deterministic prediction and pointwise absolute-error columns."""
    prefix = method_prefix(method)
    n_points = len(pointwise_df)

    for quantity in ("u", "v"):
        pred = prediction_array(
            preds=preds,
            quantity=quantity,
            expected_points=n_points,
            method=method,
        )
        ref = pointwise_df[f"{quantity}_ref"].to_numpy(dtype=float)
        pointwise_df[f"{prefix}_{quantity}_pred"] = pred
        pointwise_df[f"{prefix}_{quantity}_abs_error"] = np.abs(pred - ref)


def aligned_mc_samples(
    samples: np.ndarray,
    expected_points: int,
    expected_samples: int,
    quantity: str,
) -> np.ndarray:
    """Return MC samples with shape (mc_samples, points)."""
    arr = np.asarray(samples, dtype=float)

    if arr.ndim == 1:
        if arr.size != expected_points:
            raise ValueError(
                f"1D MC samples for {quantity} have length {arr.size}, "
                f"but {expected_points} held-out points were expected."
            )
        return arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(
            f"MC samples for {quantity} must be 1D or 2D, got shape {arr.shape}."
        )

    if arr.shape[1] == expected_points:
        return arr

    if arr.shape[0] == expected_points:
        return arr.T

    raise ValueError(
        f"Cannot align MC samples for {quantity}; got shape {arr.shape}, "
        f"expected one axis to equal {expected_points} points."
    )


def level_suffix(level: float) -> str:
    percent = level * 100.0
    rounded = round(percent)
    if np.isclose(percent, rounded):
        return str(int(rounded))
    return f"{level:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def central_interval_z(level: float) -> float:
    if not (0.0 < level < 1.0):
        raise ValueError(f"Interval level must be in (0, 1), got {level}.")
    return NormalDist().inv_cdf(0.5 + 0.5 * level)


def add_bpinn_mc_pointwise_columns(
    pointwise_df: pd.DataFrame,
    stats: MutableMapping[str, MutableMapping[str, np.ndarray]],
    mc_samples: int,
    interval_levels: Iterable[float],
) -> None:
    """Append MC-dropout B-PINN mean, std, error, and interval columns."""
    n_points = len(pointwise_df)

    for quantity in ("u", "v"):
        samples = aligned_mc_samples(
            samples=stats[quantity]["samples"],
            expected_points=n_points,
            expected_samples=mc_samples,
            quantity=quantity,
        )

        mean = np.mean(samples, axis=0)
        std = (
            np.std(samples, axis=0, ddof=1)
            if samples.shape[0] > 1
            else np.zeros(n_points, dtype=float)
        )
        ref = pointwise_df[f"{quantity}_ref"].to_numpy(dtype=float)

        pointwise_df[f"bpinn_{quantity}_mean"] = mean
        pointwise_df[f"bpinn_{quantity}_std"] = std
        pointwise_df[f"bpinn_{quantity}_abs_error"] = np.abs(mean - ref)

        for level in interval_levels:
            z = central_interval_z(float(level))
            suffix = level_suffix(float(level))
            pointwise_df[f"bpinn_{quantity}_lower_{suffix}"] = mean - z * std
            pointwise_df[f"bpinn_{quantity}_upper_{suffix}"] = mean + z * std


def add_bpinn_deterministic_fallback_columns(
    pointwise_df: pd.DataFrame,
    preds: MutableMapping[str, np.ndarray],
) -> None:
    """Use deterministic B-PINN output as mean when UQ is skipped/unavailable."""
    n_points = len(pointwise_df)

    for quantity in ("u", "v"):
        mean = prediction_array(
            preds=preds,
            quantity=quantity,
            expected_points=n_points,
            method="bpinn_dropout",
        )
        ref = pointwise_df[f"{quantity}_ref"].to_numpy(dtype=float)
        pointwise_df[f"bpinn_{quantity}_mean"] = mean
        pointwise_df[f"bpinn_{quantity}_std"] = np.nan
        pointwise_df[f"bpinn_{quantity}_abs_error"] = np.abs(mean - ref)
        pointwise_df[f"bpinn_{quantity}_lower_95"] = np.nan
        pointwise_df[f"bpinn_{quantity}_upper_95"] = np.nan


def main() -> None:
    args = parse_args()
    validate_args(args)

    data_dir = Path(args.data_dir)
    models_root = Path(args.models_root)
    output_dir = Path(args.results_root) / "heldout_evaluation"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    meanfield_path = data_dir / "LES_meanfield_nasahump2009_tec.dat"
    if not meanfield_path.exists():
        raise FileNotFoundError(
            f"Missing NASA hump mean-field file: {meanfield_path}"
        )

    meanfield = read_les_meanfield_tec(meanfield_path)
    train_df, test_df, manifest = make_exact_split(
        meanfield=meanfield,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    manifest.to_csv(output_dir / "heldout_split_manifest.csv", index=False)
    train_df.to_csv(output_dir / "heldout_train_points.csv", index=False)
    test_df.to_csv(output_dir / "heldout_test_points.csv", index=False)

    print(f"Total valid LES velocity points: {len(train_df) + len(test_df)}")
    print(f"Training points: {len(train_df)}")
    print(f"Held-out test points: {len(test_df)}")
    print(f"Split seed: {args.seed}")

    # Dropout probability is part of the model architecture and is not stored
    # in a state_dict. Set it before constructing/loading the B-PINN model.
    if "bpinn_dropout" in HUMP_METHODS:
        HUMP_METHODS["bpinn_dropout"]["dropout_rate"] = float(args.dropout_rate)

    device = get_device()
    models = load_hump_models(
        methods=args.methods,
        models_root=models_root,
        device=device,
        require_models=args.require_models,
    )
    if not models:
        raise RuntimeError(f"No model checkpoints loaded from {models_root}")

    x_test = test_df["x"].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)
    pointwise_df = build_pointwise_table(test_df)

    metric_rows: List[Dict[str, object]] = []
    timing_rows: List[Dict[str, object]] = []
    deterministic_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for method, model in models.items():
        # Warm-up call is excluded from timing.
        predict_on_points(
            model=model,
            x=x_test,
            y=y_test,
            device=device,
            batch_size=args.batch_size,
            train_mode=False,
        )

        repeated_times: List[float] = []
        preds = None

        for _ in range(args.timing_repeats):
            preds, elapsed = predict_on_points(
                model=model,
                x=x_test,
                y=y_test,
                device=device,
                batch_size=args.batch_size,
                train_mode=False,
            )
            repeated_times.append(float(elapsed))

        assert preds is not None
        deterministic_predictions[method] = {
            "u": prediction_array(preds, "u", len(test_df), method),
            "v": prediction_array(preds, "v", len(test_df), method),
        }

        timing_rows.append(
            {
                "method": method,
                "label": method_label(method),
                "points": len(test_df),
                "timing_repeats": args.timing_repeats,
                "inference_time_mean_seconds": float(
                    np.mean(repeated_times)
                ),
                "inference_time_median_seconds": float(
                    np.median(repeated_times)
                ),
                "inference_time_std_seconds": float(
                    np.std(repeated_times, ddof=0)
                ),
                "dropout_rate": (
                    float(args.dropout_rate)
                    if method == "bpinn_dropout"
                    else np.nan
                ),
            }
        )

        prediction_table = test_df.copy()

        for quantity in ("u", "v"):
            truth = test_df[quantity].to_numpy(dtype=float)
            pred = deterministic_predictions[method][quantity]
            values = metric_dict(pred, truth)
            values.update(
                {
                    "method": method,
                    "label": method_label(method),
                    "variable": quantity,
                    "split": "heldout_test",
                    "train_ratio": args.train_ratio,
                    "seed": args.seed,
                    "dropout_rate": (
                        float(args.dropout_rate)
                        if method == "bpinn_dropout"
                        else np.nan
                    ),
                }
            )
            metric_rows.append(values)
            prediction_table[f"{quantity}_pred"] = pred
            prediction_table[f"{quantity}_abs_error"] = np.abs(pred - truth)

        prediction_table.to_csv(
            predictions_dir / f"{method}_heldout_predictions.csv",
            index=False,
        )

        # Add deterministic methods directly to the combined pointwise table.
        # B-PINN is added below using MC-dropout mean/std when UQ is enabled.
        if method != "bpinn_dropout":
            add_deterministic_pointwise_columns(
                pointwise_df=pointwise_df,
                method=method,
                preds=deterministic_predictions[method],
            )

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "heldout_global_uv_errors.csv",
        index=False,
    )
    pd.DataFrame(timing_rows).to_csv(
        output_dir / "heldout_inference_timing.csv",
        index=False,
    )

    bpinn_pointwise_added = False
    if not args.skip_uq and "bpinn_dropout" in models:
        model = models["bpinn_dropout"]
        stats = mc_dropout_stats(
            model=model,
            x=x_test,
            y=y_test,
            device=device,
            samples=args.mc_samples,
            batch_size=args.batch_size,
        )

        uq_rows: List[Dict[str, object]] = []

        for quantity in ("u", "v"):
            rows = uq_metric_rows(
                method="bpinn_dropout",
                quantity=quantity,
                truth=test_df[quantity].to_numpy(dtype=float),
                samples=stats[quantity]["samples"],
                levels=args.uq_levels,
                mc_samples=args.mc_samples,
                mc_inference_time_seconds=stats[quantity]["time"],
                region_name="heldout_LES_velocity_points",
            )
            for row in rows:
                row.update(
                    {
                        "label": method_label("bpinn_dropout"),
                        "split": "heldout_test",
                        "train_ratio": args.train_ratio,
                        "seed": args.seed,
                        "dropout_rate": float(args.dropout_rate),
                    }
                )
            uq_rows.extend(rows)

        pd.DataFrame(uq_rows).to_csv(
            output_dir / "heldout_uncertainty_calibration_metrics.csv",
            index=False,
        )

        # Always include 95% intervals in the pointwise file, even when the
        # command-line --uq-levels omit 0.95.
        interval_levels = sorted({float(level) for level in args.uq_levels} | {0.95})
        add_bpinn_mc_pointwise_columns(
            pointwise_df=pointwise_df,
            stats=stats,
            mc_samples=args.mc_samples,
            interval_levels=interval_levels,
        )
        bpinn_pointwise_added = True

    if (
        not bpinn_pointwise_added
        and "bpinn_dropout" in deterministic_predictions
    ):
        add_bpinn_deterministic_fallback_columns(
            pointwise_df=pointwise_df,
            preds=deterministic_predictions["bpinn_dropout"],
        )

    pointwise_path = output_dir / "heldout_pointwise_predictions.csv"
    pointwise_df.to_csv(pointwise_path, index=False)

    print(
        "Held-out evaluation finished. "
        f"Outputs: {output_dir.resolve()}"
    )
    print(f"Combined pointwise predictions: {pointwise_path.resolve()}")


if __name__ == "__main__":
    main()
