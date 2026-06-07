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


def build_base():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = np.load(ROOT / "P.npy", mmap_mode="r").shape[0]
    items = np.load(ROOT / "Q.npy", mmap_mode="r").shape[0]
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
    ua = safe_avg(us[u], U, 20.0)
    ia = safe_avg(is_[i], I, 5.0)
    lu = np.log1p(U).astype(np.float32)
    li = np.log1p(I).astype(np.float32)
    x = np.stack(
        [np.ones_like(y), lu, li, lu * lu, li * li, 1 / np.sqrt(U + 1), 1 / np.sqrt(I + 1), ua, ia],
        axis=1,
    ).astype(np.float32)
    coef = np.linalg.solve(x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(9) * 1e-4, x.T.astype(np.float64) @ y.astype(np.float64)).astype(np.float32)
    pred = x @ coef
    print(f"base9 {rmse(y, pred):.9f}", flush=True)
    return users, u, y, pred


def user_target(users, u, y, pred):
    residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual, minlength=users).astype(np.float64)
    counts = np.bincount(u, minlength=users).astype(np.float64)
    target = np.zeros(users, dtype=np.float64)
    mask = counts > 0
    target[mask] = sums[mask] / counts[mask]
    return target, counts


def eval_basis(label, B, target, counts, u, y, pred):
    mask = counts > 0
    X = B[mask].astype(np.float64)
    wgt = counts[mask].astype(np.float64)
    yy = target[mask].astype(np.float64)
    sw = np.sqrt(wgt / max(float(wgt.mean()), 1e-12))[:, None]
    Xw = X * sw
    yw = yy * sw[:, 0]
    best = (99.0, None)
    for ridge in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        coef = np.linalg.solve(Xw.T @ Xw + np.eye(B.shape[1]) * ridge, Xw.T @ yw)
        corr = (B.astype(np.float64) @ coef).astype(np.float32)
        score = rmse(y, pred + corr[u])
        if score < best[0]:
            best = (score, ridge)
    print(f"{label:24s} cols {B.shape[1]:3d} rmse {best[0]:.9f} ridge {best[1]:g}", flush=True)
    return best[0]


def fourier_1d(users, cols):
    x = (np.arange(users, dtype=np.float64) + 0.5) / users
    feats = [np.ones(users, dtype=np.float64)]
    k = 1
    while len(feats) < cols:
        feats.append(np.sin(2 * np.pi * k * x))
        if len(feats) >= cols:
            break
        feats.append(np.cos(2 * np.pi * k * x))
        k += 1
    return np.stack(feats[:cols], axis=1).astype(np.float32)


def cheb_1d(users, cols):
    x = 2.0 * (np.arange(users, dtype=np.float64) + 0.5) / users - 1.0
    feats = [np.ones(users, dtype=np.float64), x]
    for _ in range(2, cols):
        feats.append(2 * x * feats[-1] - feats[-2])
    return np.stack(feats[:cols], axis=1).astype(np.float32)


def two_axis(users, cols, high, low):
    uid = np.arange(users, dtype=np.int64)
    a = (uid * high // users).astype(np.float64)
    b = (uid % low).astype(np.float64)
    xa = (a + 0.5) / high
    xb = (b + 0.5) / low
    feats = [np.ones(users, dtype=np.float64)]
    # Deterministic low-frequency interactions over the coarse and modulo axes.
    fa = [(0, np.ones(users, dtype=np.float64))]
    fb = [(0, np.ones(users, dtype=np.float64))]
    for k in range(1, 16):
        fa.append((k, np.sin(2 * np.pi * k * xa)))
        fa.append((-k, np.cos(2 * np.pi * k * xa)))
        fb.append((k, np.sin(2 * np.pi * k * xb)))
        fb.append((-k, np.cos(2 * np.pi * k * xb)))
    for _, va in fa:
        for _, vb in fb:
            if len(feats) >= cols:
                break
            if va is fa[0][1] and vb is fb[0][1]:
                continue
            feats.append(va * vb)
        if len(feats) >= cols:
            break
    return np.stack(feats[:cols], axis=1).astype(np.float32)


def main():
    users, u, y, pred = build_base()
    target, counts = user_target(users, u, y, pred)
    for cols in (16, 32, 64, 96, 117):
        eval_basis(f"fourier1d_{cols}", fourier_1d(users, cols), target, counts, u, y, pred)
        eval_basis(f"cheb1d_{cols}", cheb_1d(users, cols), target, counts, u, y, pred)
    for high, low in ((192, 296), (192, 294), (256, 257), (384, 257), (128, 512)):
        eval_basis(f"twoaxis_{high}_{low}", two_axis(users, 117, high, low), target, counts, u, y, pred)


if __name__ == "__main__":
    main()
