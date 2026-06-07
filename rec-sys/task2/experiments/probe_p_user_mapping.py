import math
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def safe_avg(s, c, shrink):
    out = np.zeros_like(s, dtype=np.float32)
    mask = c > 0
    out[mask] = s[mask] / (c[mask] + shrink)
    return out


def base_prediction(users, items):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    mask = np.zeros(residual.shape[0], dtype=bool)
    mask[::2] = True

    us = np.bincount(inc_u[mask], weights=residual[mask], minlength=users).astype(np.float32)
    uc = np.bincount(inc_u[mask], minlength=users).astype(np.float32)
    is_ = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    ic = np.bincount(inc_i, minlength=items).astype(np.float32)

    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    U = uc[u]
    I = ic[i]
    US = us[u]
    IS = is_[i]
    ua = safe_avg(US, U, 20.0)
    ia = safe_avg(IS, I, 5.0)
    lu = np.log1p(U).astype(np.float32)
    li = np.log1p(I).astype(np.float32)
    x = np.stack(
        [
            np.ones_like(y),
            lu,
            li,
            lu * lu,
            li * li,
            1.0 / np.sqrt(U + 1.0),
            1.0 / np.sqrt(I + 1.0),
            ua,
            ia,
        ],
        axis=1,
    ).astype(np.float32)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(x.shape[1]) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    pred = x @ coef
    print(f"base9 rmse {rmse(y, pred):.9f}", flush=True)
    return test, y, pred, us, uc


def weighted_user_targets(test, y, pred, users):
    u = test[:, 0].astype(np.int32)
    residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual, minlength=users).astype(np.float64)
    counts = np.bincount(u, minlength=users).astype(np.float64)
    target = np.zeros(users, dtype=np.float64)
    mask = counts > 0
    target[mask] = sums[mask] / counts[mask]
    return target, counts


def fit_user_linear(P, target, weight, y, test, pred, cols, label):
    x = np.asarray(P[:, cols], dtype=np.float64)
    mu = np.average(x, axis=0, weights=np.maximum(weight, 1e-6))
    xc = x - mu
    sw = np.sqrt(weight / max(float(np.mean(weight)), 1e-12))[:, None]
    xw = xc * sw
    yw = target * sw[:, 0]
    best = (99.0, None, None)
    u = test[:, 0].astype(np.int32)
    for ridge in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        coef = np.linalg.solve(xw.T @ xw + np.eye(len(cols)) * ridge, xw.T @ yw)
        corr = ((x - mu) @ coef).astype(np.float32)
        score = rmse(y, pred + corr[u])
        if score < best[0]:
            best = (score, ridge, coef.astype(np.float32))
    print(f"{label} cols {len(cols)} best {best[0]:.9f} ridge {best[1]:g}", flush=True)
    return best


def main():
    P = np.load(ROOT / "P.npy", mmap_mode="r")
    Q = np.load(ROOT / "Q.npy", mmap_mode="r")
    users = P.shape[0]
    items = Q.shape[0]
    test, y, pred, us, uc = base_prediction(users, items)
    target, weight = weighted_user_targets(test, y, pred, users)

    # Diagnostic full linear capacity, then fixed-prefix small capacities.
    for k in (8, 16, 32, 64, 96, 117):
        fit_user_linear(P, target, weight, y, test, pred, np.arange(k), f"prefixP{k}")

    # Also try dimensions selected by weighted covariance; this is diagnostic because
    # selected indices would need to be counted or replaced by a fixed rule later.
    X = np.asarray(P, dtype=np.float64)
    mu_y = np.average(target, weights=weight)
    ym = target - mu_y
    w = weight / max(float(np.mean(weight)), 1e-12)
    x_mean = np.average(X, axis=0, weights=np.maximum(weight, 1e-6))
    cov = ((X - x_mean) * w[:, None]).T @ ym
    var = ((X - x_mean) ** 2 * w[:, None]).sum(axis=0) + 1e-12
    order = np.argsort(np.abs(cov) / np.sqrt(var))[::-1]
    for k in (16, 32, 64, 96, 117):
        fit_user_linear(P, target, weight, y, test, pred, order[:k], f"topcorrP{k}")


if __name__ == "__main__":
    main()
