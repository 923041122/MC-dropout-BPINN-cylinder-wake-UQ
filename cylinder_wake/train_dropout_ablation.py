"""Independently retrain one MC-dropout B-PINN for each cylinder-wake dropout rate.

This script implements the formal retraining-based ablation stated in the
revised manuscript. It never reuses one checkpoint across different training
dropout rates. All rates use the same architecture, labelled-data fraction,
collocation count, loss weights, optimizer settings, epoch count, and seed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_RATES = [0.002, 0.005, 0.010, 0.020, 0.050]

def rate_tag(rate: float) -> str:
    return f"{rate:.6f}".rstrip("0").rstrip(".").replace(".", "p")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train independent cylinder-wake dropout-ablation checkpoints.")
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--dropout-rates", nargs="+", type=float, default=DEFAULT_RATES)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--supervised-ratio", type=float, default=0.02)
    p.add_argument("--batch-fraction", type=float, default=0.02)
    p.add_argument("--n-equation-points", type=int, default=100000)
    p.add_argument("--data-loss-weight", type=float, default=10.0)
    p.add_argument("--equation-loss-weight", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("benchmark_results/models/dropout_ablation"))
    p.add_argument("--overwrite", action="store_true", help="Delete an existing rate-specific checkpoint and retrain from scratch.")
    p.add_argument("--resume", action="store_true", help="Resume each rate from its own rate-specific checkpoint.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().with_name("benchmark_train.py")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for rate in args.dropout_rates:
        if not 0.0 < rate < 1.0:
            raise ValueError(f"Invalid dropout rate: {rate}")
        tag = rate_tag(rate)
        checkpoint = args.checkpoint_dir / f"bpinn_dropout_dr_{tag}.pth"
        if checkpoint.exists() and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"{checkpoint} already exists. Use --overwrite for an independent fresh retraining "
                "or --resume to continue this same rate-specific run."
            )
        if checkpoint.exists() and args.overwrite:
            checkpoint.unlink()

        cmd = [
            sys.executable, str(script),
            "--method", "bpinn_dropout",
            "--data-path", str(args.data_path),
            "--epochs", str(args.epochs),
            "--supervised-ratio", str(args.supervised_ratio),
            "--batch-fraction", str(args.batch_fraction),
            "--n-equation-points", str(args.n_equation_points),
            "--data-loss-weight", str(args.data_loss_weight),
            "--equation-loss-weight", str(args.equation_loss_weight),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--dropout-rate", str(rate),
            "--checkpoint", str(checkpoint),
            "--run-name", f"bpinn_dropout_dr_{tag}",
        ]
        if args.resume:
            cmd.append("--resume")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("All independently retrained dropout-rate checkpoints are complete.")

if __name__ == "__main__":
    main()
