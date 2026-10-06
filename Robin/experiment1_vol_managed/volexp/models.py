"""GARCH(1,1), GJR-GARCH(1,1) and GARCH-LSTM, each trainable on
  Objective A (accuracy): daily Gaussian quasi-likelihood  sum[ log H_t + eps_t^2 / H_t ]
  Objective B (decision): monthly investor utility of the volatility-managed portfolio after costs
                          mean(R) - gamma/2 mean(R^2),  R_m = w_m f_m - kappa |w_m - w_{m-1}|,  w_m = c / V_m

Daily variance model
  GJR part : h_t = omega + (alpha + gam * 1[eps_{t-1}<0]) eps_{t-1}^2 + beta h_{t-1}      (gam = 0 -> GARCH)
  LSTM part: H_t = h_t * exp(g_t),  g_t = s * tanh(W . LSTM(features of days t-63..t-1))
             features: standardized residual eps/sqrt(h), its absolute value, log(h / sample var)
             The output layer starts at zero, so the GARCH-LSTM starts exactly at its GJR solution.
Monthly forecast made on the last trading day T of month m, for month m+1 (K = 21 trading days):
  V = K*hbar + (H_{T+1} - hbar) * (1 - P^K) / (1 - P),  P = alpha + beta + gam/2,  hbar = omega / (1-P)

The GJR recursion is computed exactly with an FFT convolution (h is linear in the shocks given the
parameters), which keeps it differentiable and fast over 25,000 days.
"""
import math
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
K_DAYS = 21
WIN = 63
ZHAO_LEN = 126   # input length used by Zhao et al.


def _logit(p):
    return math.log(p / (1 - p))


class VolModel(nn.Module):
    def __init__(self, kind, var0):
        super().__init__()
        assert kind in ("garch", "gjr", "lstm", "zhao")
        self.kind = kind
        self.asym = kind in ("gjr", "lstm", "zhao")
        self.var0 = float(var0)  # training-sample daily variance (scale + init + feature normalisation)
        # raw parameters (start from typical daily equity values)
        self.u_mu = nn.Parameter(torch.tensor(0.0))
        self.u_P = nn.Parameter(torch.tensor(_logit(0.98 / 0.9995)))
        self.u_b = nn.Parameter(torch.tensor(_logit(0.90 / 0.98)))
        self.u_a = nn.Parameter(torch.tensor(_logit(0.35)))
        self.u_w = nn.Parameter(torch.tensor(0.0))
        if kind == "lstm":
            self.lstm = nn.LSTM(3, 8, batch_first=True).float()
            self.head = nn.Linear(8, 1).float()
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.g_scale = 1.0
        if kind == "zhao":
            # Zhao et al. (2024) eq. (7): LSTM gates fed (eps_{t-1}, sigma2_{t-1}); output gate = GJR kernel
            self.H = 8
            self.L = ZHAO_LEN
            self.gates = nn.Linear(2, 3 * self.H)     # W_f,W_i,W_c (on eps) and U_f,U_i,U_c (on sigma2), biases
            self.u_out = nn.Parameter(torch.zeros(self.H))  # w in sigma2 = o (1 + w . tanh(c)); 0 -> pure GJR

    # ---------- parameters ----------
    def params(self):
        P = 0.9995 * torch.sigmoid(self.u_P)
        beta = P * torch.sigmoid(self.u_b)
        rest = P - beta
        if self.asym:
            # rest = alpha + gam/2 = (alpha + alpha_neg)/2 with alpha >= 0 and alpha_neg = alpha + gam >= 0,
            # so gam may be negative (as for Mom); u_a = 0 gives gam = 0
            s = torch.sigmoid(self.u_a)
            alpha = 2 * rest * s
            gam = 2 * rest * (1 - s) - alpha
        else:
            alpha, gam = rest, torch.zeros(())
        # omega = var0 * (1-P) * exp(u_w): u_w = 0 is variance targeting, u_w is free
        omega = self.var0 * (1 - P) * torch.exp(self.u_w)
        mu = self.u_mu * 1e-3
        return dict(mu=mu, P=P, alpha=alpha, beta=beta, gam=gam, omega=omega)

    def readable(self):
        with torch.no_grad():
            p = self.params()
        return {k: float(v) for k, v in p.items()}

    # ---------- daily variance path ----------
    def gjr_path(self, r):
        """r: (n,) daily returns. Returns eps, h (variance for each day, info to t-1) and h_next
        (one-step forecast for the day after each day)."""
        p = self.params()
        eps = r - p["mu"]
        a = p["alpha"] + p["gam"] * (eps < 0).to(r.dtype)
        x = p["omega"] + a * eps ** 2  # x_s drives h_{s+1}
        n = r.shape[0]
        # h_next[s] = sum_{j=0}^{s} beta^j x_{s-j} + beta^{s+1} h0
        j = torch.arange(n, dtype=r.dtype)
        kern = torch.exp(j * torch.log(p["beta"]))
        L = 1 << (2 * n - 1).bit_length()
        conv = torch.fft.irfft(torch.fft.rfft(x, L) * torch.fft.rfft(kern, L), L)[:n]
        h0 = torch.tensor(self.var0, dtype=r.dtype)
        h_next = conv + kern * p["beta"] * h0
        h = torch.cat([h0.reshape(1), h_next[:-1]])
        return eps, h, h_next, p

    def g_at(self, eps, h, idx):
        """LSTM log-adjustment for the forecast made at the close of day idx (uses days idx-62..idx)."""
        z = (eps / torch.sqrt(h)).clamp(-6, 6)
        feats = torch.stack([z, z.abs(), torch.log(h / self.var0)], 1).float()
        pad = torch.zeros(WIN - 1, 3, dtype=torch.float32)
        feats = torch.cat([pad, feats], 0)
        win = feats.unfold(0, WIN, 1).permute(0, 2, 1)  # (n, WIN, 3); row s ends at day s
        out, _ = self.lstm(win[idx])
        g = self.g_scale * torch.tanh(self.head(out[:, -1, :]).squeeze(-1))
        return g.double()

    def zhao_windows(self, r, ends, eps=None, h=None, p=None):
        """Zhao et al. GARCH-LSTM run over the ZHAO_LEN days ending at each index in `ends`.
        The window starts from the GJR-filtered variance of its first day and c = 0.
        Returns (B, L): column k is the variance forecast for day start+k+1 (k = L-1 -> day end+1)."""
        if eps is None:
            eps, h, _, p = self.gjr_path(r)
        # the recursion runs in float32 on variances in percent^2 (x1e4), as the paper's x100 scaling
        L = self.L
        starts = ends - L + 1
        f32 = torch.float32
        s2 = (1e4 * h[starts]).to(f32)
        E = (100 * eps[starts.unsqueeze(1) + torch.arange(L).unsqueeze(0)]).to(f32)   # (B, L)
        om, al, ga, be = [(p[k] * (1e4 if k == "omega" else 1)).to(f32) for k in ("omega", "alpha", "gam", "beta")]
        c = torch.zeros(len(ends), self.H, dtype=f32)
        u = self.u_out.to(f32)
        w = 0.99 * u / (1 + u.abs().sum())          # sum|w| < 1 keeps sigma2 > 0
        Wg, bg = self.gates.weight.to(f32), self.gates.bias.to(f32)
        H = self.H
        preds = []
        for k in range(L):
            e = E[:, k]
            o = om + (al + ga * (e < 0).to(f32)) * e * e + be * s2
            g = torch.addmm(bg, torch.stack([e, s2], 1), Wg.t())
            c = torch.sigmoid(g[:, :H]) * c + torch.sigmoid(g[:, H:2 * H]) * torch.tanh(g[:, 2 * H:])
            s2 = o * (1 + torch.tanh(c) @ w)
            preds.append(s2)
        return torch.stack(preds, 1).to(torch.float64) / 1e4

    def next_var(self, r, idx=None):
        """One-step variance forecast H_{s+1} for every day s (or for s in idx)."""
        eps, h, h_next, p = self.gjr_path(r)
        if idx is None:
            idx = torch.arange(r.shape[0])
        Hn = h_next[idx]
        if self.kind == "lstm":
            Hn = Hn * torch.exp(self.g_at(eps, h, idx))
        if self.kind == "zhao":
            ok = idx >= self.L - 1        # first half-year of the sample: plain GJR
            if ok.any():
                Hn = Hn.clone()
                Hn[ok] = self.zhao_windows(r, idx[ok], eps, h, p)[:, -1]
        return Hn, eps, p

    def monthly_forecast(self, r, me_idx):
        """Variance forecast for the next month, made at month-end day indices me_idx."""
        Hn, _, p = self.next_var(r, me_idx)
        P = p["P"]
        hbar = p["omega"] / (1 - P)
        return K_DAYS * hbar + (Hn - hbar) * (1 - P ** K_DAYS) / (1 - P)

    # ---------- objectives ----------
    def qmle_window_loss(self, r, ends):
        """Zhao model: Gaussian quasi-likelihood over every day of the windows ending at `ends`."""
        eps, h, _, p = self.gjr_path(r)
        pred = self.zhao_windows(r, ends, eps, h, p)
        tgt = (ends - self.L + 2).unsqueeze(1) + torch.arange(self.L).unsqueeze(0)
        e2 = eps[tgt] ** 2
        return (torch.log(pred) + e2 / pred).mean()

    def qmle_loss(self, r, days):
        """Mean daily Gaussian quasi-likelihood over the given day indices (t >= 1)."""
        Hn, eps, _ = self.next_var(r, days - 1)
        e2 = eps[days] ** 2
        return (torch.log(Hn) + e2 / Hn).mean()


def managed_returns(V, f, c=None, kappa=0.001):
    """w = c / V, net return w*f - kappa*|dw|. If c is None it is set so sd(w f) = sd(f) on this sample."""
    raw = f / V
    if c is None:
        c = f.std() / raw.std()
    w = c / V
    R = w * f
    dw = torch.cat([torch.zeros(1), (w[1:] - w[:-1]).abs()])
    return R - kappa * dw, w, c


def utility(V, f, gamma=5.0, kappa=0.001):
    R, _, _ = managed_returns(V, f, None, kappa)
    return R.mean() - 0.5 * gamma * (R ** 2).mean()
