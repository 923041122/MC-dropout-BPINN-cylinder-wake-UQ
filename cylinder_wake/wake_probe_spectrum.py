"""Reproduce the cylinder-wake probe signal and FFT spectrum used in Fig. 2.

Default probe: (x, y) = (1.500, 0.000)
Quantity: transverse velocity v(t)
Prediction: MC-dropout predictive mean from 50 stochastic forward passes
Spectrum: mean-removed one-sided FFT amplitude spectrum
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_config import LAYER_MAT_PSI, METHODS
from benchmark_evaluate import load_data_stack
from benchmark_tools import build_model, get_device, predict_uvp_numpy, safe_load_state

def parse_args():
    p = argparse.ArgumentParser(description="Cylinder-wake probe and FFT comparison.")
    p.add_argument("--data-path", default="./2d_cylinder_Re3900_100x100_kw_sst.mat")
    p.add_argument("--checkpoint", default=str(METHODS["bpinn_dropout"]["checkpoint"]))
    p.add_argument("--probe-x", type=float, default=1.5)
    p.add_argument("--probe-y", type=float, default=0.0)
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("benchmark_results/evaluation/wake_probe"))
    return p.parse_args()

def spectrum(time, signal):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    order = np.argsort(time)
    time, signal = time[order], signal[order]
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time values must be strictly increasing.")
    mean_dt = float(dt.mean())
    if not np.allclose(dt, mean_dt, rtol=1e-4, atol=1e-10):
        raise ValueError("FFT requires uniformly sampled time values.")
    centered = signal - signal.mean()
    amp = np.abs(np.fft.rfft(centered)) * (2.0 / centered.size)
    freq = np.fft.rfftfreq(centered.size, d=mean_dt)
    if amp.size:
        amp[0] *= 0.5
    return freq, amp

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_data_stack(args.data_path)
    x_all, y_all, t_all = data[:, 0], data[:, 1], data[:, 2]
    xy = np.column_stack([x_all, y_all])
    target = np.array([args.probe_x, args.probe_y])
    unique_xy, first_idx = np.unique(xy, axis=0, return_index=True)
    nearest_xy = unique_xy[np.argmin(np.linalg.norm(unique_xy - target, axis=1))]
    mask = np.isclose(x_all, nearest_xy[0]) & np.isclose(y_all, nearest_xy[1])
    probe = data[mask]
    probe = probe[np.argsort(probe[:, 2])]
    if len(probe) < 4:
        raise ValueError("Too few time samples at the selected probe.")

    cfg = dict(METHODS["bpinn_dropout"])
    model = build_model(cfg, LAYER_MAT_PSI).to(get_device())
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(safe_load_state(checkpoint, get_device()), strict=True)
    model.train()  # keep dropout active for MC inference

    x = probe[:, 0:1]
    y = probe[:, 1:2]
    time = probe[:, 2:3]
    v_samples = []
    for _ in range(args.mc_samples):
        pred, _ = predict_uvp_numpy(
            model, x, y, time, get_device(), batch_size=args.batch_size, eval_mode=False
        )
        v_samples.append(pred["v"].reshape(-1))
    v_mean = np.mean(np.stack(v_samples, axis=0), axis=0)
    v_ref = probe[:, 4]

    f_ref, a_ref = spectrum(time.reshape(-1), v_ref)
    f_pred, a_pred = spectrum(time.reshape(-1), v_mean)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "time": time.reshape(-1), "v_reference": v_ref, "v_mc_mean": v_mean,
        "probe_x_requested": args.probe_x, "probe_y_requested": args.probe_y,
        "probe_x_used": nearest_xy[0], "probe_y_used": nearest_xy[1],
        "mc_samples": args.mc_samples,
    }).to_csv(args.output_dir / "wake_probe_signal.csv", index=False)
    pd.DataFrame({
        "frequency": f_ref, "reference_amplitude": a_ref, "prediction_amplitude": a_pred,
    }).to_csv(args.output_dir / "wake_probe_spectrum.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(time, v_ref, label="Reference")
    axes[0].plot(time, v_mean, label="MC-dropout predictive mean")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("v")
    axes[0].set_title(f"Wake probe at ({nearest_xy[0]:.3f}, {nearest_xy[1]:.3f})")
    axes[0].legend()
    axes[1].plot(f_ref[1:], a_ref[1:], label="Reference")
    axes[1].plot(f_pred[1:], a_pred[1:], label="MC-dropout predictive mean")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("One-sided FFT spectrum")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "wake_probe_fft.png", dpi=300)
    fig.savefig(args.output_dir / "wake_probe_fft.pdf")
    plt.close(fig)
    print(f"Outputs saved to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
