"""Training routines for objectives A and B."""
import copy
import numpy as np
import torch
from models import VolModel, utility

GJR_KEYS = ["u_mu", "u_P", "u_b", "u_a", "u_w"]


def net_params(model):
    if model.kind == "lstm":
        return list(model.lstm.parameters()) + list(model.head.parameters())
    if model.kind == "zhao":
        return list(model.gates.parameters()) + [model.u_out]
    return []


def fit_A_garch(model, r, n_iter=200):
    """Full-batch QMLE with L-BFGS (GARCH and GJR)."""
    days = torch.arange(1, r.shape[0])
    ps = [getattr(model, k) for k in GJR_KEYS]
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=n_iter, tolerance_grad=1e-10, tolerance_change=1e-13,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = model.qmle_loss(r, days)
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        return float(model.qmle_loss(r, days))


def fit_A_lstm(model, r, steps=300, batch=4096, val_frac=0.2, seed=0, lr=3e-3):
    """QMLE for the GARCH-LSTM: minibatch Adam, early stopping on the last val_frac of days."""
    g = torch.Generator().manual_seed(seed)
    n = r.shape[0]
    n_tr = int(n * (1 - val_frac))
    tr_days = torch.arange(WARM := 64, n_tr)
    va_days = torch.arange(n_tr, n)
    opt = torch.optim.Adam([
        {"params": [getattr(model, k) for k in GJR_KEYS], "lr": lr / 10},
        {"params": list(model.lstm.parameters()) + list(model.head.parameters()), "lr": lr},
    ])
    best, best_state, since = np.inf, copy.deepcopy(model.state_dict()), 0
    for s in range(steps):
        idx = tr_days[torch.randint(len(tr_days), (min(batch, len(tr_days)),), generator=g)]
        opt.zero_grad()
        loss = model.qmle_loss(r, idx)
        loss.backward()
        opt.step()
        if s % 10 == 9:
            with torch.no_grad():
                vl = float(model.qmle_loss(r, va_days))
            if vl < best - 1e-6:
                best, best_state, since = vl, copy.deepcopy(model.state_dict()), 0
            else:
                since += 1
                if since >= 6:
                    break
    model.load_state_dict(best_state)
    return best


def fit_B(model, r, me_idx, f, steps=300, val_frac=0.2, lr=None, gamma=5.0, kappa=0.001):
    """Decision objective: maximise the managed portfolio's utility after costs over the training
    months (forecast made at me_idx[m], applied to the month return f[m]). Starts from the A solution,
    early-stops on the last val_frac of months."""
    M = len(f)
    m_tr = int(M * (1 - val_frac))
    if lr is None:
        lr = 1e-2 if model.kind != "lstm" else 1e-3
    # the mean is frozen at its A value: B only re-tunes the variance dynamics
    groups = [{"params": [getattr(model, k) for k in GJR_KEYS if k != "u_mu"], "lr": lr}]
    groups.append({"params": net_params(model), "lr": lr}) if model.kind in ("lstm", "zhao") else None
    opt = torch.optim.Adam(groups)

    def utils():
        V = model.monthly_forecast(r, me_idx)
        return utility(V[:m_tr], f[:m_tr], gamma, kappa), utility(V[m_tr:], f[m_tr:], gamma, kappa)

    # one forward per step: the validation utility of the current parameters comes from the same pass
    # that gives the training gradient (the same sequence of states is checked as with a second pass)
    best, best_state, since = -np.inf, None, 0
    for s in range(steps + 1):
        opt.zero_grad()
        tu, vu = utils()
        if float(vu) > best + 1e-7:
            best, best_state, since = float(vu), copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= 30:
                break
        if s == steps:
            break
        (-tu * 100).backward()
        opt.step()
    model.load_state_dict(best_state)
    return best


def fit_A_zhao(model, r, steps=400, batch=64, val_frac=0.2, seed=0, lr=3e-3):
    """QMLE for the Zhao et al. GARCH-LSTM on random 126-day windows (Adam), early stopping on
    non-overlapping windows covering the last val_frac of days."""
    g = torch.Generator().manual_seed(seed)
    n = r.shape[0]
    L = model.L
    n_tr = int(n * (1 - val_frac))
    lo, hi = L - 1, n_tr - 2                     # window ends whose targets stay in the training part
    va_ends = torch.arange(n_tr + L - 2, n - 1, L)
    if len(va_ends) == 0:
        va_ends = torch.tensor([n - 2])
    opt = torch.optim.Adam([
        {"params": [getattr(model, k) for k in GJR_KEYS], "lr": lr / 10},
        {"params": net_params(model), "lr": lr},
    ])
    best, best_state, since = np.inf, copy.deepcopy(model.state_dict()), 0
    for s in range(steps):
        ends = torch.randint(lo, hi + 1, (batch,), generator=g)
        opt.zero_grad()
        loss = model.qmle_window_loss(r, ends)
        loss.backward()
        opt.step()
        if s % 10 == 9:
            with torch.no_grad():
                vl = float(model.qmle_window_loss(r, va_ends))
            if vl < best - 1e-6:
                best, best_state, since = vl, copy.deepcopy(model.state_dict()), 0
            else:
                since += 1
                if since >= 6:
                    break
    model.load_state_dict(best_state)
    return best
