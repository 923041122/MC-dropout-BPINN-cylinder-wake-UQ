"""
benchmark_tools.py

Minimal standalone tools for NASA 2D wall-mounted hump PINN training/validation.

Put this file in the same folder as:
    hump_train.py
    hump_validation.py

This version provides:
    build_model
    count_parameters
    get_device
    safe_load_state
    predict_uvp_numpy

The model is a psi-p PINN:
    input : x, y, t
    output: psi, p
    u = d psi / d y
    v = - d psi / d x
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PINN_Net(nn.Module):
    """
    Psi-p PINN.

    Input:
        x, y, t

    Output:
        psi, p

    Velocity:
        u = d psi / d y
        v = - d psi / d x
    """

    def __init__(self, layer_mat, dropout_rate: float = 0.0):
        super().__init__()

        self.layer_mat = list(layer_mat)
        self.dropout_rate = float(dropout_rate)

        layers = []

        for i in range(len(self.layer_mat) - 2):
            layers.append(nn.Linear(self.layer_mat[i], self.layer_mat[i + 1]))
            layers.append(nn.Tanh())

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

        layers.append(nn.Linear(self.layer_mat[-2], self.layer_mat[-1]))

        self.base = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.base:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.base(inputs)

    def predict_fields(self, x, y, t, create_graph: bool = True):
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        if not t.requires_grad:
            t.requires_grad_(True)

        out = self.forward(x, y, t)

        psi = out[:, 0:1]
        p = out[:, 1:2]

        u = torch.autograd.grad(
            psi.sum(),
            y,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        v = -torch.autograd.grad(
            psi.sum(),
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        return u, v, p

    def predict_fields_safe(
        self,
        x,
        y,
        t,
        create_graph: bool = False,
        train_mode: bool = False,
    ):
        was_training = self.training

        if train_mode:
            self.train()
        else:
            self.eval()

        x_eval = x.detach().clone().requires_grad_(True)
        y_eval = y.detach().clone().requires_grad_(True)
        t_eval = t.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            u_pred, v_pred, p_pred = self.predict_fields(
                x_eval,
                y_eval,
                t_eval,
                create_graph=create_graph,
            )

        if was_training:
            self.train()
        else:
            self.eval()

        return u_pred.detach(), v_pred.detach(), p_pred.detach()


def build_model(cfg: Dict, layer_mat):
    model_type = cfg.get("model_type", "psi")
    dropout_rate = float(cfg.get("dropout_rate", 0.0))

    if model_type != "psi":
        raise ValueError(
            f"This standalone benchmark_tools.py only supports model_type='psi'. "
            f"Got model_type={model_type!r}."
        )

    return PINN_Net(layer_mat=layer_mat, dropout_rate=dropout_rate)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_uvp_numpy(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    t_np: np.ndarray,
    device: torch.device,
    batch_size: int = 20000,
    eval_mode: bool = True,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Batched prediction for u, v, p.

    This function keeps gradients enabled because u and v are derivatives of psi.
    """

    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1, 1)
    t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)

    n = x_np.shape[0]

    all_u = []
    all_v = []
    all_p = []

    was_training = model.training

    if eval_mode:
        model.eval()
    else:
        model.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)

        x = torch.tensor(
            x_np[start_idx:end_idx],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        y = torch.tensor(
            y_np[start_idx:end_idx],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        t = torch.tensor(
            t_np[start_idx:end_idx],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.enable_grad():
            u, v, p = model.predict_fields(
                x,
                y,
                t,
                create_graph=False,
            )

        all_u.append(_to_numpy(u))
        all_v.append(_to_numpy(v))
        all_p.append(_to_numpy(p))

        del x, y, t, u, v, p

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if was_training:
        model.train()
    else:
        model.eval()

    preds = {
        "u": np.concatenate(all_u, axis=0),
        "v": np.concatenate(all_v, axis=0),
        "p": np.concatenate(all_p, axis=0),
    }

    return preds, elapsed
