# ukn_duffing_toy.py
# PyTorch implementation of Unscented KalmanNet (UKN) for the Duffing toy problem.
# nx=2, ny=2, ntheta=2, theta only in f.

import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility: SPD helpers
# -----------------------------
def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def project_spd(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project a symmetric matrix to SPD via eigenvalue clamping."""
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

def safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5) -> torch.Tensor:
    """
    Batched safe Cholesky: tries A + jitter*I, increases jitter if needed.
    Falls back to SPD projection if still failing.
    """
    A = symmetrize(A)

    if A.dim() == 2:
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    else:
        B = A.shape[0]
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).expand(B, -1, -1)

    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(A + (jitter * (10**i)) * I)
        except RuntimeError:
            continue

    A_spd = project_spd(A, eps=jitter)
    return torch.linalg.cholesky(A_spd)


# -----------------------------
# Unscented transform (weights + sigma points)
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
    A = scaled.transpose(1, 2)                # (B,n,n) where A[:,i,:] = column i of scaled

    mean_exp = mean.unsqueeze(1)              # (B,1,n)
    sig0 = mean_exp
    sig_plus = mean_exp + A
    sig_minus = mean_exp - A
    return torch.cat([sig0, sig_plus, sig_minus], dim=1)


# -----------------------------
# Toy dynamics: Duffing oscillator
# z = [p, v, k, alpha] where theta=[k,alpha] only in f
# y = [p, v] + noise
# -----------------------------
def duffing_f(z: torch.Tensor, u: torch.Tensor, dt: float = 0.05, c: float = 0.25):
    """
    z: (...,4), u: broadcastable to (...)
    returns z_next with same leading dims.
    """
    p = z[..., 0]
    v = z[..., 1]
    k = z[..., 2]
    alpha = z[..., 3]

    p_next = p + dt * v
    v_next = v + dt * (-c * v - k * p - alpha * (p**3) + u)

    # theta random walk is handled via Q_theta in covariance; deterministic part keeps theta
    k_next = k
    alpha_next = alpha

    return torch.stack([p_next, v_next, k_next, alpha_next], dim=-1)

def h_identity_x(z: torch.Tensor):
    """Measurement model: y = x + noise, where x=[p,v]."""
    return z[..., 0:2]


# -----------------------------
# UKF core ops (predict + measurement stats)
# -----------------------------
def ukf_predict(z: torch.Tensor, P: torch.Tensor, u_prev: torch.Tensor, Q: torch.Tensor,
                wm: torch.Tensor, wc: torch.Tensor, sqrt_c: float,
                dt: float = 0.05, c: float = 0.25,
                jitter: float = 1e-6):
    """
    z: (B,n), P: (B,n,n), u_prev: (B,)
    Q: (n,n) or (B,n,n)
    returns z_pred, P_pred, X_sigma (predicted sigma points)
    """
    B, n = z.shape
    sig = sigma_points(z, P, sqrt_c, jitter=jitter)  # (B,S,n)

    u_b = u_prev.view(B, 1)                          # (B,1) broadcast to sigma dim
    X_sigma = duffing_f(sig, u_b, dt=dt, c=c)        # (B,S,n)

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
                          jitter: float = 1e-6):
    """
    Build y_pred, S, P_zy from predicted sigma points.
    """
    B, S, n = X_sigma.shape
    ny = R.shape[-1]

    Y_sigma = h_identity_x(X_sigma)                        # (B,S,ny)
    y_pred = torch.sum(Y_sigma * wm.view(1, -1, 1), dim=1) # (B,ny)

    dY = Y_sigma - y_pred.unsqueeze(1)                     # (B,S,ny)
    dX = X_sigma - z_pred.unsqueeze(1)                     # (B,S,n)

    S_cov = torch.einsum('bsi,bsj,s->bij', dY, dY, wc)      # (B,ny,ny)
    if R.dim() == 2:
        S_cov = S_cov + R.unsqueeze(0)
    else:
        S_cov = S_cov + R

    S_cov = symmetrize(S_cov) + jitter * torch.eye(ny, device=z_pred.device, dtype=z_pred.dtype).unsqueeze(0)
    P_zy = torch.einsum('bsi,bsj,s->bij', dX, dY, wc)       # (B,n,ny)
    return y_pred, S_cov, P_zy

def compute_K_ukf(P_zy: torch.Tensor, S: torch.Tensor, jitter: float = 1e-6):
    """
    K = P_zy S^{-1} using Cholesky solve.
    """
    cholS = safe_cholesky(S, jitter=jitter)  # (B,ny,ny)
    K_T = torch.cholesky_solve(P_zy.transpose(-1, -2), cholS)  # (B,ny,n)
    K = K_T.transpose(-1, -2)  # (B,n,ny)
    return K, cholS

def generalized_joseph(P_pred: torch.Tensor, K: torch.Tensor, S: torch.Tensor, P_zy: torch.Tensor,
                       jitter: float = 1e-6):
    """
    Generalized Joseph update (safe for non-optimal learned K):
      P_post = P_pred - K P_yz - P_zy K^T + K S K^T
    """
    P_yz = P_zy.transpose(-1, -2)                     # (B,ny,n)
    P_post = P_pred - (K @ P_yz) - (P_zy @ K.transpose(-1, -2)) + (K @ S @ K.transpose(-1, -2))
    P_post = symmetrize(P_post) + jitter * torch.eye(P_pred.shape[-1], device=P_pred.device, dtype=P_pred.dtype).unsqueeze(0)
    return P_post


# -----------------------------
# GainNet (Encoder-MLP -> GRUCell -> heads)
# Outputs: deltaK(4x2), rho_x, rho_theta
# -----------------------------
class GainNet(nn.Module):
    def __init__(self,
                 d_in: int = 20,
                 hidden: int = 32,
                 embed: int = 32,
                 deltaK_scale: float = 0.1,
                 rho_theta_max: float = 0.3,
                 init_rho_bias=(-3.0, -5.0)):
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

        # Init: start close to UKF (ΔK ~ 0, rho ~ small)
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

        dk_raw = self.dk_head(t)          # (B,8)
        gate_raw = self.gate_head(t)      # (B,2)

        rho_x = torch.sigmoid(gate_raw[:, 0:1])                       # (B,1)
        rho_theta = self.rho_theta_max * torch.sigmoid(gate_raw[:, 1:2])

        # ΔK scaled relative to ||K_ukf|| to prevent blow-up
        scale = self.deltaK_scale * torch.linalg.norm(K_ukf.reshape(K_ukf.shape[0], -1),
                                                      dim=-1, keepdim=True)  # (B,1)
        deltaK = (scale * torch.tanh(dk_raw)).view(-1, 4, 2)

        return deltaK, rho_x, rho_theta, h_new


# -----------------------------
# UKN filter (Plan B): UKF backbone + GainNet correction
# Feature: Core-20
# -----------------------------
class UKNFilter(nn.Module):
    def __init__(self,
                 dt: float = 0.05,
                 c: float = 0.25,
                 ut_params: UTParams = UTParams(alpha=0.8, beta=2.0, kappa=0.0),
                 Q: torch.Tensor = None,
                 R: torch.Tensor = None,
                 jitter: float = 1e-6,
                 feature_eps: float = 1e-8,
                 hidden: int = 32,
                 deltaK_scale: float = 0.1,
                 rho_theta_max: float = 0.3):
        super().__init__()
        self.dt = dt
        self.c = c
        self.jitter = jitter
        self.feature_eps = feature_eps

        self.n_z = 4
        self.n_y = 2

        if Q is None:
            # Q = diag([Qp, Qv, Qk, Qalpha])
            Q = torch.diag(torch.tensor([1e-5, 1e-4, 1e-6, 1e-6], dtype=torch.float32))
        if R is None:
            R = torch.diag(torch.tensor([0.05**2, 0.05**2], dtype=torch.float32))

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
        Core-20 features:
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

        I = torch.eye(self.n_y, device=nu.device, dtype=nu.dtype).expand(B, -1, -1)
        inv_LS = torch.linalg.solve_triangular(cholS, I, upper=False)            # (B,2,2) = L_S^{-1}

        C_theta_y = C_temp @ inv_LS.transpose(-1, -2)                            # (B,2,2)
        vecC = C_theta_y.reshape(B, -1)                                          # (B,4)

        delta_z_ukf = (K_ukf @ nu.unsqueeze(-1)).squeeze(-1)                     # (B,4)

        feat = torch.cat([tilde_nu, nis, logdetS, delta_z_pred, logdiagP, vecC, delta_z_ukf], dim=-1)
        return feat  # (B,20)

    def forward(self,
                y: torch.Tensor,      # (B,T,2)
                u: torch.Tensor,      # (B,T)
                z0: torch.Tensor,     # (B,4) prior mean at time 0
                P0: torch.Tensor,     # (B,4,4) prior cov at time 0
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
                # no predict, use prior directly
                z_pred, P_pred = z, P
                sig = sigma_points(z_pred, P_pred, self.sqrt_c, jitter=self.jitter)
                X_sigma = sig
                z_prev_ref = z_pred  # makes delta_z_pred = 0 at t=0
            else:
                u_prev = u[:, t-1]
                z_pred, P_pred, X_sigma = ukf_predict(z, P, u_prev, self.Q,
                                                      self.wm, self.wc, self.sqrt_c,
                                                      dt=self.dt, c=self.c, jitter=self.jitter)
                z_prev_ref = z_prev

            y_pred, S, P_zy = ukf_measurement_stats(z_pred, P_pred, X_sigma, self.R,
                                                    self.wm, self.wc, jitter=self.jitter)

            nu = y[:, t, :] - y_pred  # innovation (B,2)
            K_ukf, cholS = compute_K_ukf(P_zy, S, jitter=self.jitter)

            feat = self._build_features(nu, cholS, z_pred, P_pred, P_zy, K_ukf, z_prev_ref)

            deltaK, rho_x, rho_theta, h = self.gain_net(feat, h, K_ukf)

            # row-wise gating: first two rows for x, last two rows for theta
            g_row = torch.cat([rho_x, rho_x, rho_theta, rho_theta], dim=-1)  # (B,4)
            K = K_ukf + deltaK * g_row.unsqueeze(-1)                         # (B,4,2)

            z_post = z_pred + (K @ nu.unsqueeze(-1)).squeeze(-1)             # (B,4)
            P_post = generalized_joseph(P_pred, K, S, P_zy, jitter=self.jitter)

            z_hist.append(z_post)

            if return_debug:
                debug.append({
                    "rho_x": rho_x.detach(),
                    "rho_theta": rho_theta.detach(),
                    "nu": nu.detach(),
                    "K_ukf": K_ukf.detach(),
                    "K": K.detach(),
                })

            z_prev = z_post
            z, P = z_post, P_post

        z_filt = torch.stack(z_hist, dim=1)  # (B,T,4)
        if return_debug:
            return z_filt, debug
        return z_filt


# -----------------------------
# Baseline: Augmented UKF (no NN)
# -----------------------------
class AugmentedUKF(nn.Module):
    def __init__(self,
                 dt: float = 0.05,
                 c: float = 0.25,
                 ut_params: UTParams = UTParams(alpha=0.8, beta=2.0, kappa=0.0),
                 Q: torch.Tensor = None,
                 R: torch.Tensor = None,
                 jitter: float = 1e-6):
        super().__init__()
        self.dt = dt
        self.c = c
        self.jitter = jitter
        self.n_z = 4
        self.n_y = 2

        if Q is None:
            Q = torch.diag(torch.tensor([1e-5, 1e-4, 1e-6, 1e-6], dtype=torch.float32))
        if R is None:
            R = torch.diag(torch.tensor([0.05**2, 0.05**2], dtype=torch.float32))

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
                                                      dt=self.dt, c=self.c, jitter=self.jitter)

            y_pred, S, P_zy = ukf_measurement_stats(z_pred, P_pred, X_sigma, self.R,
                                                    self.wm, self.wc, jitter=self.jitter)
            nu = y[:, t, :] - y_pred
            K_ukf, _ = compute_K_ukf(P_zy, S, jitter=self.jitter)

            z_post = z_pred + (K_ukf @ nu.unsqueeze(-1)).squeeze(-1)
            P_post = generalized_joseph(P_pred, K_ukf, S, P_zy, jitter=self.jitter)

            z_hist.append(z_post)
            z, P = z_post, P_post

        return torch.stack(z_hist, dim=1)


# -----------------------------
# Data simulation (Duffing toy with optional outliers)
# -----------------------------
def simulate_duffing_batch(B: int = 64,
                           T: int = 200,
                           dt: float = 0.05,
                           c: float = 0.25,
                           outlier: bool = True,
                           seed: int = 0):
    """
    Returns torch tensors:
      x: (B,T,2) true state
      y: (B,T,2) measurements
      theta: (B,2) true params [k, alpha]
      u: (B,T) input (piecewise constant)
    """
    rng = np.random.default_rng(seed)

    # True process noise (simulation)
    Qx = np.diag([1e-5, 1e-4])

    # Measurement noise
    sigma = 0.05
    R0 = np.diag([sigma**2, sigma**2])
    p_out = 0.05
    Rout = 25.0 * R0

    # input
    M = 5

    x = np.zeros((B, T, 2), dtype=np.float32)
    y = np.zeros((B, T, 2), dtype=np.float32)
    u = np.zeros((B, T), dtype=np.float32)
    theta = np.zeros((B, 2), dtype=np.float32)

    for b in range(B):
        k = rng.uniform(0.8, 1.2)
        alpha = rng.uniform(0.2, 0.6)
        theta[b] = [k, alpha]

        for t in range(T):
            if t % M == 0:
                u[b, t] = rng.uniform(-1.0, 1.0)
            else:
                u[b, t] = u[b, t-1]

        # initial state: make nonlinearity visible
        x[b, 0, 0] = rng.normal(1.0, 0.2)  # p0
        x[b, 0, 1] = rng.normal(0.0, 0.2)  # v0

        for t in range(T-1):
            p, v = x[b, t]
            w = rng.multivariate_normal(np.zeros(2), Qx)

            p_next = p + dt * v + w[0]
            v_next = v + dt * (-c*v - k*p - alpha*(p**3) + u[b, t]) + w[1]
            x[b, t+1] = [p_next, v_next]

        # measurement
        for t in range(T):
            if outlier and (rng.uniform() < p_out):
                v_meas = rng.multivariate_normal(np.zeros(2), Rout)
            else:
                v_meas = rng.multivariate_normal(np.zeros(2), R0)
            y[b, t] = x[b, t] + v_meas

    return (torch.tensor(x), torch.tensor(y), torch.tensor(theta), torch.tensor(u))


# -----------------------------
# Demo: forward + optional training loop
# -----------------------------
def make_initial_prior(y: torch.Tensor,
                       theta_prior=(1.0, 0.4),
                       P0_diag=(0.1**2, 0.1**2, 0.5**2, 0.5**2)):
    """
    Use y[:,0] as x0 prior, fixed theta prior.
    """
    B = y.shape[0]
    z0 = torch.zeros((B, 4), dtype=y.dtype, device=y.device)
    z0[:, 0:2] = y[:, 0, :]  # since we measure p,v

    z0[:, 2] = float(theta_prior[0])
    z0[:, 3] = float(theta_prior[1])

    P0 = torch.diag(torch.tensor(P0_diag, dtype=y.dtype, device=y.device)).unsqueeze(0).repeat(B, 1, 1)
    return z0, P0

def train_quick_demo(device="cpu"):
    torch.manual_seed(0)

    # Models
    ukn = UKNFilter().to(device)
    ukf = AugmentedUKF().to(device)

    opt = torch.optim.Adam(ukn.parameters(), lr=1e-3)

    lambda_theta = 1.0
    lambda_smooth = 0.1
    lambda_dK = 1e-3
    margin = 0.0         # (원하면 >0로 해서 "UKF보다 더 좋아져라" 마진 부여)
    lambda_dom = 1.0

    for step in range(200):
        x, y, theta, u = simulate_duffing_batch(B=64, T=120, outlier=True, seed=step)
        x = x.to(device); y = y.to(device); theta = theta.to(device); u = u.to(device)

        z0, P0 = make_initial_prior(y)
        z0 = z0.to(device); P0 = P0.to(device)

        z_hat = ukn(y, u, z0, P0)               # (B,T,4)
        x_hat = z_hat[..., 0:2]
        th_hat = z_hat[..., 2:4]

        # supervised losses
        loss_x = F.mse_loss(x_hat, x)
        th_true_seq = theta[:, None, :].expand_as(th_hat)
        loss_th = F.mse_loss(th_hat, th_true_seq)

        # theta smoothness (since theta is constant per episode)
        loss_smooth = F.mse_loss(th_hat[:, 1:, :] - th_hat[:, :-1, :], torch.zeros_like(th_hat[:, 1:, :]))

        # small regularization to keep ΔK small (optional)
        # (간단히 rho*deltaK를 직접 꺼내기 귀찮으면, 모델 파라미터 L2로도 대체 가능)
        reg = 0.0
        for p in ukn.parameters():
            reg = reg + (p**2).mean()

        # baseline dominance vs UKF (optional but fits your "must be better" goal)
        with torch.no_grad():
            z_ukf = ukf(y, u, z0, P0)
            x_ukf = z_ukf[..., 0:2]
            th_ukf = z_ukf[..., 2:4]
            loss_ukf = F.mse_loss(x_ukf, x) + lambda_theta * F.mse_loss(th_ukf, th_true_seq)

        loss_ukn = loss_x + lambda_theta * loss_th + lambda_smooth * loss_smooth + lambda_dK * reg
        dom = F.relu(loss_ukn - loss_ukf + margin)
        loss = loss_ukn + lambda_dom * dom

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ukn.parameters(), max_norm=5.0)
        opt.step()

        if step % 20 == 0:
            print(f"step {step:04d} | loss={loss.item():.5f} | ukf={loss_ukf.item():.5f} "
                  f"| x={loss_x.item():.5f} th={loss_th.item():.5f} dom={dom.item():.5f}")

    # quick eval
    x, y, theta, u = simulate_duffing_batch(B=256, T=200, outlier=True, seed=999)
    x = x.to(device); y = y.to(device); theta = theta.to(device); u = u.to(device)
    z0, P0 = make_initial_prior(y)
    z0 = z0.to(device); P0 = P0.to(device)

    with torch.no_grad():
        z_hat = ukn(y, u, z0, P0)
        z_ukf = ukf(y, u, z0, P0)

        rmse_ukn_x = torch.sqrt(((z_hat[...,0:2]-x)**2).mean()).item()
        rmse_ukf_x = torch.sqrt(((z_ukf[...,0:2]-x)**2).mean()).item()

        th_true_seq = theta[:, None, :].expand_as(z_hat[...,2:4])
        rmse_ukn_th = torch.sqrt(((z_hat[...,2:4]-th_true_seq)**2).mean()).item()
        rmse_ukf_th = torch.sqrt(((z_ukf[...,2:4]-th_true_seq)**2).mean()).item()

    print("\n[EVAL on dirty(outlier) test]")
    print(f"RMSE_x   UKN={rmse_ukn_x:.6f} | UKF={rmse_ukf_x:.6f}")
    print(f"RMSE_th  UKN={rmse_ukn_th:.6f} | UKF={rmse_ukf_th:.6f}")

if __name__ == "__main__":
    # For CPU demo:
    train_quick_demo(device="cpu")
