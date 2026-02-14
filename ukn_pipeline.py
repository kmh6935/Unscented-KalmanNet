"""
ukn_pipeline.py

End-to-end toy pipeline for Unscented KalmanNet (UKN):
- Nonlinear measurement model (2D)
- GPU auto-detection (CUDA/MPS/CPU)
- Dataset generation & saving (train/test)
- Training & testing from saved datasets
- Saving histories and test predictions
- Plotting results (training curve, 99% error bands, example trajectories)

Usage:
  python ukn_pipeline.py generate --out_dir data --train_N 20000 --test_N 4000 --T 200
  python ukn_pipeline.py train --data_dir data --results_dir results/run1 --epochs 50 --batch_size 128 --lr 1e-3
  python ukn_pipeline.py plot --results_dir results/run1 --example_idx 0
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Device helper
# -----------------------------
def get_best_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# Utility: SPD helpers
# -----------------------------
def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def project_spd(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project a symmetric matrix to SPD via eigenvalue clamping (batched)."""
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

def safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 6) -> torch.Tensor:
    """
    Batched safe Cholesky: tries A + jitter*I, increases jitter if needed.
    Falls back to SPD projection if still failing.
    """
    A = symmetrize(A)
    n = A.shape[-1]
    if A.dim() == 2:
        I = torch.eye(n, device=A.device, dtype=A.dtype)
    else:
        B = A.shape[0]
        I = torch.eye(n, device=A.device, dtype=A.dtype).expand(B, -1, -1)

    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(A + (jitter * (10 ** i)) * I)
        except RuntimeError:
            continue

    A_spd = project_spd(A, eps=jitter)
    return torch.linalg.cholesky(A_spd)

# -----------------------------
# Unscented transform params
# -----------------------------
@dataclass
class UTParams:
    alpha: float = 0.8
    beta: float = 2.0
    kappa: float = 0.0

def unscented_weights(n: int, params: UTParams, device=None, dtype=None):
    """
    Returns:
      wm: (2n+1,)
      wc: (2n+1,)
      sqrt_c: scalar float = sqrt(n + lambda)
    """
    alpha, beta, kappa = params.alpha, params.beta, params.kappa
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    wm = torch.full((2*n+1,), 1.0/(2.0*c), device=device, dtype=dtype)
    wc = torch.full((2*n+1,), 1.0/(2.0*c), device=device, dtype=dtype)
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - alpha**2 + beta)
    sqrt_c = math.sqrt(c)
    return wm, wc, sqrt_c

def sigma_points(mean: torch.Tensor, cov: torch.Tensor, sqrt_c: float, jitter: float = 1e-6):
    """
    mean: (B,n), cov: (B,n,n)
    returns sigma: (B,2n+1,n)
    """
    B, n = mean.shape
    L = safe_cholesky(cov, jitter=jitter)     # (B,n,n) lower
    scaled = sqrt_c * L                       # (B,n,n)
    A = scaled.transpose(1, 2)                # (B,n,n) columns as rows
    mean_exp = mean.unsqueeze(1)              # (B,1,n)
    sig0 = mean_exp
    sig_plus = mean_exp + A
    sig_minus = mean_exp - A
    return torch.cat([sig0, sig_plus, sig_minus], dim=1)

# -----------------------------
# Toy model: Duffing oscillator
# z = [p, v, k, alpha], theta only in f
# Measurement: nonlinear 2D
# -----------------------------
@dataclass
class ToyConfig:
    dt: float = 0.05
    c: float = 0.25
    # measurement nonlinearity coefficients: y1 = p + ap*p^3, y2 = v + av*v^3
    ap: float = 0.1
    av: float = 0.1

    # True process noise for simulation (x only)
    Qx_p: float = 1e-5
    Qx_v: float = 1e-4

    # Measurement noise (additive in measurement space)
    sigma_meas: float = 0.05
    outlier: bool = True
    p_out: float = 0.05
    outlier_scale: float = 25.0  # Rout = scale * R0

    # Input
    u_refresh: int = 5
    u_min: float = -1.0
    u_max: float = 1.0

    # Parameter distribution (true)
    k_min: float = 0.8
    k_max: float = 1.2
    alpha_min: float = 0.2
    alpha_max: float = 0.6

    # Initial state distribution (true)
    p0_mean: float = 1.0
    p0_std: float = 0.2
    v0_mean: float = 0.0
    v0_std: float = 0.2

def f_duffing(z: torch.Tensor, u: torch.Tensor, dt: float, c: float) -> torch.Tensor:
    """
    z: (...,4), u: broadcastable to leading dims
    """
    p = z[..., 0]
    v = z[..., 1]
    k = z[..., 2]
    alpha = z[..., 3]

    p_next = p + dt * v
    v_next = v + dt * (-c * v - k * p - alpha * (p**3) + u)

    # parameters deterministic part stays constant (random walk handled by Q in filter)
    return torch.stack([p_next, v_next, k, alpha], dim=-1)

def h_nonlinear(z: torch.Tensor, ap: float, av: float) -> torch.Tensor:
    """
    Nonlinear measurement (2D):
      y1 = p + ap*p^3
      y2 = v + av*v^3
    z: (...,4)
    returns (...,2)
    """
    p = z[..., 0]
    v = z[..., 1]
    y1 = p + ap * (p**3)
    y2 = v + av * (v**3)
    return torch.stack([y1, y2], dim=-1)

def invert_cubic_measurement(y: torch.Tensor, a: float, n_iter: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """
    Solve x + a*x^3 = y for x using Newton iterations (elementwise).
    Works well for small a and moderate y.
    y: (...,)
    """
    x = y.clone()
    for _ in range(n_iter):
        f = x + a * (x**3) - y
        fp = 1.0 + 3.0 * a * (x**2)
        x = x - f / (fp + eps)
    return x

def approx_inverse_h(y: torch.Tensor, ap: float, av: float) -> torch.Tensor:
    """
    Approx inverse for initial prior: x0 ~ h^{-1}(y0).
    y: (...,2)
    """
    p_est = invert_cubic_measurement(y[..., 0], ap)
    v_est = invert_cubic_measurement(y[..., 1], av)
    return torch.stack([p_est, v_est], dim=-1)

# -----------------------------
# UKF core (predict + measurement stats)
# -----------------------------
def ukf_predict(z: torch.Tensor, P: torch.Tensor, u_prev: torch.Tensor, Q: torch.Tensor,
                wm: torch.Tensor, wc: torch.Tensor, sqrt_c: float,
                dt: float, c: float,
                jitter: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    z: (B,n), P: (B,n,n), u_prev: (B,)
    Q: (n,n) or (B,n,n)
    returns z_pred, P_pred, X_sigma (predicted sigma points)
    """
    B, n = z.shape
    sig = sigma_points(z, P, sqrt_c, jitter=jitter)  # (B,S,n)

    u_b = u_prev.view(B, 1)                          # (B,1) broadcast to sigma dim
    X_sigma = f_duffing(sig, u_b, dt=dt, c=c)        # (B,S,n)

    z_pred = torch.sum(X_sigma * wm.view(1, -1, 1), dim=1)  # (B,n)

    dX = X_sigma - z_pred.unsqueeze(1)                      # (B,S,n)
    P_pred = torch.einsum('bsi,bsj,s->bij', dX, dX, wc)     # (B,n,n)

    if Q.dim() == 2:
        P_pred = P_pred + Q.unsqueeze(0)
    else:
        P_pred = P_pred + Q

    P_pred = symmetrize(P_pred) + jitter * torch.eye(n, device=z.device, dtype=z.dtype).unsqueeze(0)
    return z_pred, P_pred, X_sigma

def ukf_measurement_stats(z_pred: torch.Tensor, P_pred: torch.Tensor, X_sigma: torch.Tensor,
                          R: torch.Tensor, wm: torch.Tensor, wc: torch.Tensor,
                          ap: float, av: float,
                          jitter: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build y_pred, S, P_zy from predicted sigma points.
    """
    B, S, n = X_sigma.shape
    ny = R.shape[-1]

    Y_sigma = h_nonlinear(X_sigma, ap=ap, av=av)             # (B,S,ny)
    y_pred = torch.sum(Y_sigma * wm.view(1, -1, 1), dim=1)   # (B,ny)

    dY = Y_sigma - y_pred.unsqueeze(1)                       # (B,S,ny)
    dX = X_sigma - z_pred.unsqueeze(1)                       # (B,S,n)

    S_cov = torch.einsum('bsi,bsj,s->bij', dY, dY, wc)       # (B,ny,ny)
    if R.dim() == 2:
        S_cov = S_cov + R.unsqueeze(0)
    else:
        S_cov = S_cov + R

    S_cov = symmetrize(S_cov) + jitter * torch.eye(ny, device=z_pred.device, dtype=z_pred.dtype).unsqueeze(0)
    P_zy = torch.einsum('bsi,bsj,s->bij', dX, dY, wc)        # (B,n,ny)
    return y_pred, S_cov, P_zy

def compute_K_ukf(P_zy: torch.Tensor, S: torch.Tensor, jitter: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K = P_zy S^{-1} using Cholesky solve.
    """
    cholS = safe_cholesky(S, jitter=jitter)  # (B,ny,ny)
    # Solve: S X^T = P_zy^T  => X = P_zy S^{-1}
    K_T = torch.cholesky_solve(P_zy.transpose(-1, -2), cholS)  # (B,ny,n)
    K = K_T.transpose(-1, -2)  # (B,n,ny)
    return K, cholS

def generalized_joseph(P_pred: torch.Tensor, K: torch.Tensor, S: torch.Tensor, P_zy: torch.Tensor,
                       jitter: float = 1e-6) -> torch.Tensor:
    """
    Generalized Joseph update (safe for non-optimal learned K):
      P_post = P_pred - K P_yz - P_zy K^T + K S K^T
    """
    P_yz = P_zy.transpose(-1, -2)  # (B,ny,n)
    P_post = P_pred - (K @ P_yz) - (P_zy @ K.transpose(-1, -2)) + (K @ S @ K.transpose(-1, -2))
    P_post = symmetrize(P_post) + jitter * torch.eye(P_pred.shape[-1], device=P_pred.device, dtype=P_pred.dtype).unsqueeze(0)
    return P_post

# -----------------------------
# GainNet: Encoder -> GRU -> heads (ΔK + gates)
# -----------------------------
class GainNet(nn.Module):
    def __init__(self,
                 d_in: int = 20,
                 hidden: int = 32,
                 embed: int = 32,
                 deltaK_scale: float = 0.1,
                 rho_theta_max: float = 0.3,
                 init_rho_bias: Tuple[float, float] = (-3.0, -5.0)):
        super().__init__()
        self.deltaK_scale = deltaK_scale
        self.rho_theta_max = rho_theta_max

        self.ln = nn.LayerNorm(d_in)
        self.enc = nn.Sequential(
            nn.Linear(d_in, embed), nn.ReLU(),
            nn.Linear(embed, embed), nn.ReLU(),
        )
        self.gru = nn.GRUCell(embed, hidden)

        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        self.dk_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 8),  # 4x2
        )
        self.gate_head = nn.Linear(hidden, 2)

        # init to start near UKF
        nn.init.zeros_(self.dk_head[-1].weight)
        nn.init.zeros_(self.dk_head[-1].bias)

        nn.init.zeros_(self.gate_head.weight)
        self.gate_head.bias.data[:] = torch.tensor(init_rho_bias, dtype=self.gate_head.bias.dtype)

    def forward(self, feat: torch.Tensor, h: torch.Tensor, K_ukf: torch.Tensor):
        """
        feat: (B,20)
        h: (B,hidden)
        K_ukf: (B,4,2) for scaling
        """
        x = self.ln(feat)
        x = self.enc(x)
        h_new = self.gru(x, h)

        t = self.trunk(h_new)

        dk_raw = self.dk_head(t)           # (B,8)
        gate_raw = self.gate_head(t)       # (B,2)

        rho_x = torch.sigmoid(gate_raw[:, 0:1])                     # (B,1)
        rho_theta = self.rho_theta_max * torch.sigmoid(gate_raw[:, 1:2])

        # scale ΔK relative to ||K_ukf|| to prevent blow-up
        scale = self.deltaK_scale * torch.linalg.norm(K_ukf.reshape(K_ukf.shape[0], -1),
                                                      dim=-1, keepdim=True)  # (B,1)
        deltaK = (scale * torch.tanh(dk_raw)).view(-1, 4, 2)

        return deltaK, rho_x, rho_theta, h_new

# -----------------------------
# UKN filter: UKF backbone + GainNet correction
# Feature: Core-20
# -----------------------------
class UKNFilter(nn.Module):
    def __init__(self,
                 toy: ToyConfig,
                 ut_params: UTParams = UTParams(alpha=0.8, beta=2.0, kappa=0.0),
                 Q: Optional[torch.Tensor] = None,
                 R: Optional[torch.Tensor] = None,
                 jitter: float = 1e-6,
                 feature_eps: float = 1e-8,
                 hidden: int = 32,
                 deltaK_scale: float = 0.1,
                 rho_theta_max: float = 0.3):
        super().__init__()
        self.toy = toy
        self.jitter = jitter
        self.feature_eps = feature_eps

        self.n_z = 4
        self.n_y = 2

        if Q is None:
            # Filter Q: includes parameter random-walk
            Q = torch.diag(torch.tensor([1e-5, 1e-4, 1e-6, 1e-6], dtype=torch.float32))
        if R is None:
            sigma = toy.sigma_meas
            R = torch.diag(torch.tensor([sigma**2, sigma**2], dtype=torch.float32))

        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

        wm, wc, sqrt_c = unscented_weights(self.n_z, ut_params, device=Q.device, dtype=Q.dtype)
        self.register_buffer("wm", wm)
        self.register_buffer("wc", wc)
        self.sqrt_c = sqrt_c

        self.hidden = hidden
        self.gain_net = GainNet(d_in=20, hidden=hidden, embed=32,
                                deltaK_scale=deltaK_scale,
                                rho_theta_max=rho_theta_max)

    def _build_features(self,
                        nu: torch.Tensor,
                        cholS: torch.Tensor,
                        z_pred: torch.Tensor,
                        P_pred: torch.Tensor,
                        P_zy: torch.Tensor,
                        K_ukf: torch.Tensor,
                        z_prev_ref: torch.Tensor) -> torch.Tensor:
        """
        Core-20:
          [tilde_nu(2), NIS(1), logdetS(1), delta_z_pred(4),
           logdiagP(4), vec(C_theta_y)(4), delta_z_ukf(4)] => 20
        """
        B = nu.shape[0]

        # whitened innovation: tilde_nu = L_S^{-1} nu
        tilde_nu = torch.linalg.solve_triangular(cholS, nu.unsqueeze(-1), upper=False).squeeze(-1)  # (B,2)

        nis = torch.sum(tilde_nu**2, dim=-1, keepdim=True)  # (B,1)

        logdetS = 2.0 * torch.sum(
            torch.log(torch.diagonal(cholS, dim1=-2, dim2=-1) + self.feature_eps),
            dim=-1, keepdim=True
        )  # (B,1)

        delta_z_pred = z_pred - z_prev_ref  # (B,4)

        logdiagP = torch.log(torch.diagonal(P_pred, dim1=-2, dim2=-1) + self.feature_eps)  # (B,4)

        # whitened cross-cov for theta-y:
        # C_theta_y = L_theta^{-1} P_theta_y L_S^{-T}
        P_theta_y = P_zy[:, 2:4, :]                # (B,2,2)
        P_theta_theta = P_pred[:, 2:4, 2:4]        # (B,2,2)

        L_theta = safe_cholesky(P_theta_theta, jitter=self.jitter)   # (B,2,2)
        C_temp = torch.linalg.solve_triangular(L_theta, P_theta_y, upper=False)  # (B,2,2)

        # inv(L_S)
        I = torch.eye(self.n_y, device=nu.device, dtype=nu.dtype).expand(B, -1, -1)
        inv_LS = torch.linalg.solve_triangular(cholS, I, upper=False)  # (B,2,2)
        C_theta_y = C_temp @ inv_LS.transpose(-1, -2)                  # (B,2,2)
        vecC = C_theta_y.reshape(B, -1)                                # (B,4)

        delta_z_ukf = (K_ukf @ nu.unsqueeze(-1)).squeeze(-1)           # (B,4)

        feat = torch.cat([tilde_nu, nis, logdetS, delta_z_pred, logdiagP, vecC, delta_z_ukf], dim=-1)
        return feat  # (B,20)

    def forward(self,
                y: torch.Tensor,      # (B,T,2)
                u: torch.Tensor,      # (B,T)
                z0: torch.Tensor,     # (B,4)
                P0: torch.Tensor,     # (B,4,4)
                return_debug: bool = False):
        B, T, _ = y.shape
        device = y.device
        dtype = y.dtype

        z = z0
        P = P0
        h = torch.zeros(B, self.hidden, device=device, dtype=dtype)

        z_hist = []
        debug = []
        z_prev = z.clone()

        for t in range(T):
            if t == 0:
                z_pred, P_pred = z, P
                sig = sigma_points(z_pred, P_pred, self.sqrt_c, jitter=self.jitter)
                X_sigma = sig
                z_prev_ref = z_pred
            else:
                u_prev = u[:, t-1]
                z_pred, P_pred, X_sigma = ukf_predict(z, P, u_prev, self.Q,
                                                      self.wm, self.wc, self.sqrt_c,
                                                      dt=self.toy.dt, c=self.toy.c,
                                                      jitter=self.jitter)
                z_prev_ref = z_prev

            y_pred, S, P_zy = ukf_measurement_stats(z_pred, P_pred, X_sigma, self.R,
                                                    self.wm, self.wc,
                                                    ap=self.toy.ap, av=self.toy.av,
                                                    jitter=self.jitter)

            nu = y[:, t, :] - y_pred
            K_ukf, cholS = compute_K_ukf(P_zy, S, jitter=self.jitter)

            feat = self._build_features(nu, cholS, z_pred, P_pred, P_zy, K_ukf, z_prev_ref)

            deltaK, rho_x, rho_theta, h = self.gain_net(feat, h, K_ukf)

            g_row = torch.cat([rho_x, rho_x, rho_theta, rho_theta], dim=-1)  # (B,4)
            K = K_ukf + deltaK * g_row.unsqueeze(-1)                         # (B,4,2)

            z_post = z_pred + (K @ nu.unsqueeze(-1)).squeeze(-1)             # (B,4)
            P_post = generalized_joseph(P_pred, K, S, P_zy, jitter=self.jitter)

            z_hist.append(z_post)

            if return_debug:
                debug.append({
                    "rho_x": rho_x.detach().cpu(),
                    "rho_theta": rho_theta.detach().cpu(),
                })

            z_prev = z_post
            z, P = z_post, P_post

        z_filt = torch.stack(z_hist, dim=1)
        if return_debug:
            return z_filt, debug
        return z_filt

# -----------------------------
# Baseline: Augmented UKF (no NN)
# -----------------------------
class AugmentedUKF(nn.Module):
    def __init__(self,
                 toy: ToyConfig,
                 ut_params: UTParams = UTParams(alpha=0.8, beta=2.0, kappa=0.0),
                 Q: Optional[torch.Tensor] = None,
                 R: Optional[torch.Tensor] = None,
                 jitter: float = 1e-6):
        super().__init__()
        self.toy = toy
        self.jitter = jitter
        self.n_z = 4
        self.n_y = 2

        if Q is None:
            Q = torch.diag(torch.tensor([1e-5, 1e-4, 1e-6, 1e-6], dtype=torch.float32))
        if R is None:
            sigma = toy.sigma_meas
            R = torch.diag(torch.tensor([sigma**2, sigma**2], dtype=torch.float32))

        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

        wm, wc, sqrt_c = unscented_weights(self.n_z, ut_params, device=Q.device, dtype=Q.dtype)
        self.register_buffer("wm", wm)
        self.register_buffer("wc", wc)
        self.sqrt_c = sqrt_c

    def forward(self, y: torch.Tensor, u: torch.Tensor, z0: torch.Tensor, P0: torch.Tensor):
        B, T, _ = y.shape
        z = z0
        P = P0

        z_hist = []
        for t in range(T):
            if t == 0:
                z_pred, P_pred = z, P
                sig = sigma_points(z_pred, P_pred, self.sqrt_c, jitter=self.jitter)
                X_sigma = sig
            else:
                u_prev = u[:, t-1]
                z_pred, P_pred, X_sigma = ukf_predict(z, P, u_prev, self.Q,
                                                      self.wm, self.wc, self.sqrt_c,
                                                      dt=self.toy.dt, c=self.toy.c,
                                                      jitter=self.jitter)

            y_pred, S, P_zy = ukf_measurement_stats(z_pred, P_pred, X_sigma, self.R,
                                                    self.wm, self.wc,
                                                    ap=self.toy.ap, av=self.toy.av,
                                                    jitter=self.jitter)
            nu = y[:, t, :] - y_pred
            K_ukf, _ = compute_K_ukf(P_zy, S, jitter=self.jitter)

            z_post = z_pred + (K_ukf @ nu.unsqueeze(-1)).squeeze(-1)
            P_post = generalized_joseph(P_pred, K_ukf, S, P_zy, jitter=self.jitter)

            z_hist.append(z_post)
            z, P = z_post, P_post

        return torch.stack(z_hist, dim=1)

# -----------------------------
# Dataset I/O
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, pt_path: str):
        blob = torch.load(pt_path, map_location="cpu")
        self.x = blob["x"]          # (N,T,2)
        self.y = blob["y"]          # (N,T,2)
        self.theta = blob["theta"]  # (N,2)
        self.u = blob["u"]          # (N,T)
        self.config = blob["config"]
        self.meta = blob.get("meta", {})

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "theta": self.theta[idx],
            "u": self.u[idx],
        }

def simulate_duffing_dataset(N: int, T: int, cfg: ToyConfig, seed: int) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)

    Qx = np.diag([cfg.Qx_p, cfg.Qx_v]).astype(np.float32)

    sigma = cfg.sigma_meas
    R0 = np.diag([sigma**2, sigma**2]).astype(np.float32)
    p_out = cfg.p_out
    Rout = (cfg.outlier_scale * R0).astype(np.float32)

    x = np.zeros((N, T, 2), dtype=np.float32)
    y = np.zeros((N, T, 2), dtype=np.float32)
    u = np.zeros((N, T), dtype=np.float32)
    theta = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = rng.uniform(cfg.k_min, cfg.k_max)
        alpha = rng.uniform(cfg.alpha_min, cfg.alpha_max)
        theta[n] = [k, alpha]

        # input
        for t in range(T):
            if t % cfg.u_refresh == 0:
                u[n, t] = rng.uniform(cfg.u_min, cfg.u_max)
            else:
                u[n, t] = u[n, t-1]

        # initial
        x[n, 0, 0] = rng.normal(cfg.p0_mean, cfg.p0_std)
        x[n, 0, 1] = rng.normal(cfg.v0_mean, cfg.v0_std)

        # simulate dynamics (Euler)
        for t in range(T-1):
            p, v = x[n, t]
            w = rng.multivariate_normal(np.zeros(2), Qx)
            p_next = p + cfg.dt * v + w[0]
            v_next = v + cfg.dt * (-cfg.c*v - k*p - alpha*(p**3) + u[n, t]) + w[1]
            x[n, t+1] = [p_next, v_next]

        # measurement: y = h(x) + noise in measurement space
        for t in range(T):
            p, v = x[n, t]
            y_clean = np.array([p + cfg.ap * (p**3), v + cfg.av * (v**3)], dtype=np.float32)

            if cfg.outlier and (rng.uniform() < p_out):
                v_meas = rng.multivariate_normal(np.zeros(2), Rout)
            else:
                v_meas = rng.multivariate_normal(np.zeros(2), R0)
            y[n, t] = y_clean + v_meas.astype(np.float32)

    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "theta": torch.from_numpy(theta),
        "u": torch.from_numpy(u),
    }

def save_dataset(out_path: str, data: Dict[str, torch.Tensor], cfg: ToyConfig, meta: Dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    blob = {
        "x": data["x"].contiguous(),
        "y": data["y"].contiguous(),
        "theta": data["theta"].contiguous(),
        "u": data["u"].contiguous(),
        "config": cfg.__dict__,
        "meta": meta,
    }
    torch.save(blob, out_path)

# -----------------------------
# Priors
# -----------------------------
def make_initial_prior(y0: torch.Tensor,
                       toy: ToyConfig,
                       theta_prior=(1.0, 0.4),
                       P0_diag=(0.2**2, 0.2**2, 0.5**2, 0.5**2)):
    """
    Build z0,P0 for batch from first measurement y0 (B,2) using approx inverse h^{-1}.
    """
    B = y0.shape[0]
    z0 = torch.zeros((B, 4), dtype=y0.dtype, device=y0.device)

    # approx inverse of nonlinear measurement to get x0
    x0 = approx_inverse_h(y0, ap=toy.ap, av=toy.av)
    z0[:, 0:2] = x0

    z0[:, 2] = float(theta_prior[0])
    z0[:, 3] = float(theta_prior[1])

    P0 = torch.diag(torch.tensor(P0_diag, dtype=y0.dtype, device=y0.device)).unsqueeze(0).repeat(B, 1, 1)
    return z0, P0

# -----------------------------
# Training / evaluation
# -----------------------------
def batch_loss(z_hat: torch.Tensor, x_true: torch.Tensor, theta_true: torch.Tensor,
               lambda_theta: float = 1.0, lambda_smooth: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    z_hat: (B,T,4)
    x_true: (B,T,2)
    theta_true: (B,2) (constant per seq)
    """
    x_hat = z_hat[..., 0:2]
    th_hat = z_hat[..., 2:4]
    th_true_seq = theta_true[:, None, :].expand_as(th_hat)

    loss_x = F.mse_loss(x_hat, x_true)
    loss_th = F.mse_loss(th_hat, th_true_seq)

    # encourage constant theta
    dth = th_hat[:, 1:, :] - th_hat[:, :-1, :]
    loss_smooth = F.mse_loss(dth, torch.zeros_like(dth))

    loss = loss_x + lambda_theta * loss_th + lambda_smooth * loss_smooth

    metrics = {
        "loss_x": float(loss_x.detach().cpu()),
        "loss_th": float(loss_th.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_total": float(loss.detach().cpu()),
    }
    return loss, metrics

@torch.no_grad()
def eval_model(model: nn.Module, dataloader: DataLoader, toy: ToyConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    total_x = 0.0
    total_th = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u = batch["u"].to(device)
        theta = batch["theta"].to(device)

        z0, P0 = make_initial_prior(y[:, 0, :], toy=toy)
        z_hat = model(y, u, z0, P0)  # (B,T,4)

        x_hat = z_hat[..., 0:2]
        th_hat = z_hat[..., 2:4]
        th_true_seq = theta[:, None, :].expand_as(th_hat)

        loss_x = F.mse_loss(x_hat, x)
        loss_th = F.mse_loss(th_hat, th_true_seq)
        loss = loss_x + loss_th

        total += float(loss.cpu())
        total_x += float(loss_x.cpu())
        total_th += float(loss_th.cpu())
        n_batches += 1

    return {
        "mse_total": total / max(1, n_batches),
        "mse_x": total_x / max(1, n_batches),
        "mse_theta": total_th / max(1, n_batches),
    }

@torch.no_grad()
def run_full_prediction(model: nn.Module, dataset: SequenceDataset, toy: ToyConfig, device: torch.device,
                        batch_size: int = 256) -> torch.Tensor:
    """
    Returns z_hat for full dataset: (N,T,4) on CPU.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outs = []
    for batch in loader:
        y = batch["y"].to(device)
        u = batch["u"].to(device)
        z0, P0 = make_initial_prior(y[:, 0, :], toy=toy)
        z_hat = model(y, u, z0, P0)
        outs.append(z_hat.detach().cpu())
    return torch.cat(outs, dim=0)

def train_command(args):
    device = get_best_device(force_cpu=args.force_cpu)
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")

    train_path = os.path.join(args.data_dir, "train.pt")
    test_path = os.path.join(args.data_dir, "test.pt")

    train_ds = SequenceDataset(train_path)
    test_ds = SequenceDataset(test_path)

    toy = ToyConfig(**train_ds.config)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    # models
    ukn = UKNFilter(toy=toy,
                    hidden=args.hidden,
                    deltaK_scale=args.deltaK_scale,
                    rho_theta_max=args.rho_theta_max).to(device)
    ukf = AugmentedUKF(toy=toy).to(device)

    opt = torch.optim.Adam(ukn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_loss_x": [],
        "train_loss_th": [],
        "train_loss_smooth": [],
        "train_dom": [],
        "test_mse_total": [],
        "test_mse_x": [],
        "test_mse_theta": [],
    }

    best = float("inf")
    os.makedirs(args.results_dir, exist_ok=True)
    cfg_out = {
        "train_args": vars(args),
        "toy_config": toy.__dict__,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        ukn.train()

        epoch_loss = 0.0
        epoch_x = 0.0
        epoch_th = 0.0
        epoch_smooth = 0.0
        epoch_dom = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            u = batch["u"].to(device, non_blocking=True)
            theta = batch["theta"].to(device, non_blocking=True)

            z0, P0 = make_initial_prior(y[:, 0, :], toy=toy)
            z_hat = ukn(y, u, z0, P0)

            loss_ukn, m = batch_loss(z_hat, x, theta,
                                     lambda_theta=args.lambda_theta,
                                     lambda_smooth=args.lambda_smooth)

            # dominance loss vs UKF (optional)
            dom = torch.tensor(0.0, device=device)
            if args.use_dom_loss:
                with torch.no_grad():
                    z_ukf = ukf(y, u, z0, P0)
                    x_ukf = z_ukf[..., 0:2]
                    th_ukf = z_ukf[..., 2:4]
                    th_true_seq = theta[:, None, :].expand_as(th_ukf)
                    loss_ukf = F.mse_loss(x_ukf, x) + args.lambda_theta * F.mse_loss(th_ukf, th_true_seq)

                dom = F.relu(loss_ukn - loss_ukf + args.dom_margin)
                loss = loss_ukn + args.lambda_dom * dom
            else:
                loss = loss_ukn

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ukn.parameters(), max_norm=args.grad_clip)
            opt.step()

            epoch_loss += float(loss_ukn.detach().cpu())
            epoch_x += m["loss_x"]
            epoch_th += m["loss_th"]
            epoch_smooth += m["loss_smooth"]
            epoch_dom += float(dom.detach().cpu())
            n_batches += 1

        # epoch averages
        epoch_loss /= max(1, n_batches)
        epoch_x /= max(1, n_batches)
        epoch_th /= max(1, n_batches)
        epoch_smooth /= max(1, n_batches)
        epoch_dom /= max(1, n_batches)

        # test metrics
        test_m = eval_model(ukn, test_loader, toy=toy, device=device)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["train_loss_x"].append(epoch_x)
        history["train_loss_th"].append(epoch_th)
        history["train_loss_smooth"].append(epoch_smooth)
        history["train_dom"].append(epoch_dom)
        history["test_mse_total"].append(test_m["mse_total"])
        history["test_mse_x"].append(test_m["mse_x"])
        history["test_mse_theta"].append(test_m["mse_theta"])

        print(f"[Epoch {epoch:03d}] train={epoch_loss:.6f} (x={epoch_x:.6f}, th={epoch_th:.6f}, sm={epoch_smooth:.6f}) "
              f"dom={epoch_dom:.6f} | test={test_m['mse_total']:.6f}")

        # save best
        if test_m["mse_total"] < best:
            best = test_m["mse_total"]
            torch.save({"model": ukn.state_dict(), "epoch": epoch, "best": best},
                       os.path.join(args.results_dir, "model_best.pt"))

        # periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({"model": ukn.state_dict(), "epoch": epoch},
                       os.path.join(args.results_dir, f"model_epoch_{epoch:03d}.pt"))

    # save last
    torch.save({"model": ukn.state_dict(), "epoch": args.epochs},
               os.path.join(args.results_dir, "model_last.pt"))
    torch.save(history, os.path.join(args.results_dir, "history.pt"))

    # Load best for final eval and save full predictions
    best_ckpt = torch.load(os.path.join(args.results_dir, "model_best.pt"), map_location="cpu")
    ukn.load_state_dict(best_ckpt["model"])
    ukn.to(device)

    print("[Final] Running full test predictions (UKN + UKF) ...")
    z_hat_test = run_full_prediction(ukn, test_ds, toy=toy, device=device, batch_size=args.eval_batch_size)
    z_ukf_test = run_full_prediction(ukf, test_ds, toy=toy, device=device, batch_size=args.eval_batch_size)

    # Save outputs for plotting
    out_blob = {
        "z_hat_test": z_hat_test,   # (N,T,4) CPU
        "z_ukf_test": z_ukf_test,   # (N,T,4) CPU
        "x_test": test_ds.x,        # (N,T,2) CPU
        "y_test": test_ds.y,        # (N,T,2) CPU
        "theta_test": test_ds.theta,# (N,2) CPU
        "u_test": test_ds.u,        # (N,T) CPU
        "toy_config": toy.__dict__,
        "best_epoch": int(best_ckpt["epoch"]),
        "best_mse_total": float(best_ckpt["best"]),
    }
    torch.save(out_blob, os.path.join(args.results_dir, "test_outputs.pt"))
    print(f"[Saved] {args.results_dir}/history.pt, test_outputs.pt, model_best.pt")

def generate_command(args):
    cfg = ToyConfig(
        dt=args.dt, c=args.c,
        ap=args.ap, av=args.av,
        Qx_p=args.Qx_p, Qx_v=args.Qx_v,
        sigma_meas=args.sigma_meas,
        outlier=bool(args.outlier),
        p_out=args.p_out,
        outlier_scale=args.outlier_scale,
        u_refresh=args.u_refresh,
        u_min=args.u_min, u_max=args.u_max,
        k_min=args.k_min, k_max=args.k_max,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        p0_mean=args.p0_mean, p0_std=args.p0_std,
        v0_mean=args.v0_mean, v0_std=args.v0_std,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Generate] train N={args.train_N}, T={args.T} -> {args.out_dir}/train.pt")
    train_data = simulate_duffing_dataset(N=args.train_N, T=args.T, cfg=cfg, seed=args.seed)
    save_dataset(os.path.join(args.out_dir, "train.pt"), train_data, cfg, meta={"split": "train", "seed": args.seed})

    print(f"[Generate] test  N={args.test_N},  T={args.T} -> {args.out_dir}/test.pt")
    test_data = simulate_duffing_dataset(N=args.test_N, T=args.T, cfg=cfg, seed=args.seed + 999)
    save_dataset(os.path.join(args.out_dir, "test.pt"), test_data, cfg, meta={"split": "test", "seed": args.seed + 999})

    print("[Done] Dataset saved.")

def plot_command(args):
    import matplotlib.pyplot as plt

    hist_path = os.path.join(args.results_dir, "history.pt")
    out_path = os.path.join(args.results_dir, "test_outputs.pt")
    if not os.path.exists(hist_path) or not os.path.exists(out_path):
        raise FileNotFoundError("history.pt 또는 test_outputs.pt 가 없습니다. train 커맨드를 먼저 실행하세요.")

    history = torch.load(hist_path, map_location="cpu")
    blob = torch.load(out_path, map_location="cpu")

    z_hat = blob["z_hat_test"]   # (N,T,4)
    z_ukf = blob["z_ukf_test"]   # (N,T,4)
    x = blob["x_test"]           # (N,T,2)
    theta = blob["theta_test"]   # (N,2)

    N, T, _ = z_hat.shape
    t_axis = np.arange(T)

    # 1) Training curve
    fig = plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train (UKN)")
    plt.plot(history["epoch"], history["test_mse_total"], label="test mse (UKN)")
    plt.xlabel("epoch")
    plt.ylabel("loss / mse")
    plt.title("Training curve")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    # Helper: compute mean + 99% band over episodes at each time
    def mean_band(err: np.ndarray, q_low=0.005, q_high=0.995):
        mean = np.mean(err, axis=0)
        lo = np.quantile(err, q_low, axis=0)
        hi = np.quantile(err, q_high, axis=0)
        return mean, lo, hi

    # 2) State error norm + 99% band (UKN vs UKF)
    err_x_ukn = (z_hat[..., 0:2] - x).numpy()
    err_x_ukf = (z_ukf[..., 0:2] - x).numpy()

    errnorm_ukn = np.linalg.norm(err_x_ukn, axis=-1)  # (N,T)
    errnorm_ukf = np.linalg.norm(err_x_ukf, axis=-1)

    mean_u, lo_u, hi_u = mean_band(errnorm_ukn)
    mean_f, lo_f, hi_f = mean_band(errnorm_ukf)

    fig = plt.figure()
    plt.plot(t_axis, mean_u, label="UKN mean |err_x|")
    plt.fill_between(t_axis, lo_u, hi_u, alpha=0.2, label="UKN 99% band")

    plt.plot(t_axis, mean_f, label="UKF mean |err_x|")
    plt.fill_between(t_axis, lo_f, hi_f, alpha=0.2, label="UKF 99% band")

    plt.xlabel("time step")
    plt.ylabel("state error norm")
    plt.title("State error with 99% band (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "error_band_state.png"), dpi=150)
    plt.close(fig)

    # 3) Parameter error norm + 99% band (UKN vs UKF)
    th_hat = z_hat[..., 2:4].numpy()                   # (N,T,2)
    th_ukf = z_ukf[..., 2:4].numpy()
    th_true = theta.numpy()[:, None, :]                # (N,1,2)

    err_th_ukn = np.linalg.norm(th_hat - th_true, axis=-1)  # (N,T)
    err_th_ukf = np.linalg.norm(th_ukf - th_true, axis=-1)

    mean_u, lo_u, hi_u = mean_band(err_th_ukn)
    mean_f, lo_f, hi_f = mean_band(err_th_ukf)

    fig = plt.figure()
    plt.plot(t_axis, mean_u, label="UKN mean |err_theta|")
    plt.fill_between(t_axis, lo_u, hi_u, alpha=0.2, label="UKN 99% band")

    plt.plot(t_axis, mean_f, label="UKF mean |err_theta|")
    plt.fill_between(t_axis, lo_f, hi_f, alpha=0.2, label="UKF 99% band")

    plt.xlabel("time step")
    plt.ylabel("parameter error norm")
    plt.title("Parameter error with 99% band (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "error_band_theta.png"), dpi=150)
    plt.close(fig)

    # Print summary RMSE
    rmse_x_ukn = float(np.sqrt(np.mean(err_x_ukn**2)))
    rmse_x_ukf = float(np.sqrt(np.mean(err_x_ukf**2)))
    rmse_th_ukn = float(np.sqrt(np.mean((th_hat - th_true)**2)))
    rmse_th_ukf = float(np.sqrt(np.mean((th_ukf - th_true)**2)))
    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write(f"RMSE_x  UKN={rmse_x_ukn:.6f}  UKF={rmse_x_ukf:.6f}\n")
        f.write(f"RMSE_th UKN={rmse_th_ukn:.6f}  UKF={rmse_th_ukf:.6f}\n")
    print(f"[Summary] RMSE_x  UKN={rmse_x_ukn:.6f} | UKF={rmse_x_ukf:.6f}")
    print(f"[Summary] RMSE_th UKN={rmse_th_ukn:.6f} | UKF={rmse_th_ukf:.6f}")
    print(f"[Saved] {args.results_dir}/summary.txt")

    # 4) One test example: predicted vs GT
    idx = int(args.example_idx)
    if idx < 0 or idx >= N:
        raise ValueError(f"example_idx must be in [0, {N-1}]")

    zhat_i = z_hat[idx].numpy()  # (T,4)
    zukf_i = z_ukf[idx].numpy()
    x_i = x[idx].numpy()
    th_i = theta[idx].numpy()

    # p
    fig = plt.figure()
    plt.plot(t_axis, x_i[:, 0], label="GT p")
    plt.plot(t_axis, zhat_i[:, 0], label="UKN p")
    plt.plot(t_axis, zukf_i[:, 0], label="UKF p")
    plt.xlabel("time step")
    plt.ylabel("p")
    plt.title(f"Example {idx}: position")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, f"example_{idx:04d}_p.png"), dpi=150)
    plt.close(fig)

    # v
    fig = plt.figure()
    plt.plot(t_axis, x_i[:, 1], label="GT v")
    plt.plot(t_axis, zhat_i[:, 1], label="UKN v")
    plt.plot(t_axis, zukf_i[:, 1], label="UKF v")
    plt.xlabel("time step")
    plt.ylabel("v")
    plt.title(f"Example {idx}: velocity")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, f"example_{idx:04d}_v.png"), dpi=150)
    plt.close(fig)

    # k
    fig = plt.figure()
    plt.plot(t_axis, np.full_like(t_axis, th_i[0], dtype=float), label="GT k")
    plt.plot(t_axis, zhat_i[:, 2], label="UKN k")
    plt.plot(t_axis, zukf_i[:, 2], label="UKF k")
    plt.xlabel("time step")
    plt.ylabel("k")
    plt.title(f"Example {idx}: parameter k")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, f"example_{idx:04d}_k.png"), dpi=150)
    plt.close(fig)

    # alpha
    fig = plt.figure()
    plt.plot(t_axis, np.full_like(t_axis, th_i[1], dtype=float), label="GT alpha")
    plt.plot(t_axis, zhat_i[:, 3], label="UKN alpha")
    plt.plot(t_axis, zukf_i[:, 3], label="UKF alpha")
    plt.xlabel("time step")
    plt.ylabel("alpha")
    plt.title(f"Example {idx}: parameter alpha")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, f"example_{idx:04d}_alpha.png"), dpi=150)
    plt.close(fig)

    print(f"[Saved] Plots to {args.results_dir}/ (training_curve.png, error_band_*.png, example_*.png)")

# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # generate
    g = sub.add_parser("generate")
    g.add_argument("--out_dir", type=str, default="data")
    g.add_argument("--train_N", type=int, default=20000)
    g.add_argument("--test_N", type=int, default=4000)
    g.add_argument("--T", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)

    # toy cfg overrides
    g.add_argument("--dt", type=float, default=0.05)
    g.add_argument("--c", type=float, default=0.25)
    g.add_argument("--ap", type=float, default=0.1)
    g.add_argument("--av", type=float, default=0.1)
    g.add_argument("--Qx_p", type=float, default=1e-5)
    g.add_argument("--Qx_v", type=float, default=1e-4)
    g.add_argument("--sigma_meas", type=float, default=0.05)
    g.add_argument("--outlier", type=int, default=1)
    g.add_argument("--p_out", type=float, default=0.05)
    g.add_argument("--outlier_scale", type=float, default=25.0)
    g.add_argument("--u_refresh", type=int, default=5)
    g.add_argument("--u_min", type=float, default=-1.0)
    g.add_argument("--u_max", type=float, default=1.0)
    g.add_argument("--k_min", type=float, default=0.8)
    g.add_argument("--k_max", type=float, default=1.2)
    g.add_argument("--alpha_min", type=float, default=0.2)
    g.add_argument("--alpha_max", type=float, default=0.6)
    g.add_argument("--p0_mean", type=float, default=1.0)
    g.add_argument("--p0_std", type=float, default=0.2)
    g.add_argument("--v0_mean", type=float, default=0.0)
    g.add_argument("--v0_std", type=float, default=0.2)

    # train
    t = sub.add_parser("train")
    t.add_argument("--data_dir", type=str, default="data")
    t.add_argument("--results_dir", type=str, default=None)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--eval_batch_size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--deltaK_scale", type=float, default=0.1)
    t.add_argument("--rho_theta_max", type=float, default=0.3)
    t.add_argument("--lambda_theta", type=float, default=1.0)
    t.add_argument("--lambda_smooth", type=float, default=0.1)
    t.add_argument("--grad_clip", type=float, default=5.0)

    t.add_argument("--use_dom_loss", type=int, default=1)
    t.add_argument("--lambda_dom", type=float, default=1.0)
    t.add_argument("--dom_margin", type=float, default=0.0)
    t.add_argument("--save_every", type=int, default=10)

    t.add_argument("--force_cpu", action="store_true")

    # plot
    pl = sub.add_parser("plot")
    pl.add_argument("--results_dir", type=str, required=True)
    pl.add_argument("--example_idx", type=int, default=0)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "generate":
        generate_command(args)
    elif args.cmd == "train":
        if args.results_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.results_dir = os.path.join("results", f"run_{ts}")
        train_command(args)
    elif args.cmd == "plot":
        plot_command(args)
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
