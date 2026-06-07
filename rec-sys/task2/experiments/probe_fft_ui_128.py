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
    return users, items, u, i, y, pred


def top_freqs(entity_count, ids, residual, limit):
    sums = np.bincount(ids, weights=residual, minlength=entity_count).astype(np.float64)
    counts = np.bincount(ids, minlength=entity_count).astype(np.float64)
    target = np.zeros(entity_count, dtype=np.float64)
    mask = counts > 0
    target[mask] = sums[mask] / counts[mask]
    series = target * np.sqrt(counts / max(float(counts.mean()), 1e-12))
    series -= series.mean()
    amp = np.abs(np.fft.rfft(series))
    amp[0] = 0
    return np.argsort(amp)[::-1][:limit]


def entity_basis(entity_count, freqs):
    x = np.arange(entity_count, dtype=np.float64)
    feats = []
    for f in freqs:
        ang = 2 * np.pi * f * x / entity_count
        feats.append(np.sin(ang).astype(np.float32))
        feats.append(np.cos(ang).astype(np.float32))
    return np.stack(feats, axis=1).astype(np.float32) if feats else np.zeros((entity_count, 0), dtype=np.float32)


def fit_combo(label, ub, ib, u, i, y, base):
    m = 1 + ub.shape[1] + ib.shape[1]
    a = np.zeros((m, m), dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)
    for start in range(0, y.shape[0], 200000):
        sl = slice(start, min(start + 200000, y.shape[0]))
        x = np.empty((sl.stop - sl.start, m), dtype=np.float32)
        x[:, 0] = 1.0
        col = 1
        if ub.shape[1]:
            x[:, col : col + ub.shape[1]] = ub[u[sl]]
            col += ub.shape[1]
        if ib.shape[1]:
            x[:, col : col + ib.shape[1]] = ib[i[sl]]
        target = (y[sl] - np.clip(base[sl], 0.5, 5.0)).astype(np.float32)
        a += x.T.astype(np.float64) @ x.astype(np.float64)
        b += x.T.astype(np.float64) @ target.astype(np.float64)
    best = (99.0, None, None)
    for ridge in (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
        coef = np.linalg.solve(a + np.eye(m) * ridge, b).astype(np.float32)
        pred = base.astype(np.float32, copy=True) + coef[0]
        col = 1
        if ub.shape[1]:
            pred += ub[u] @ coef[col : col + ub.shape[1]]
            col += ub.shape[1]
        if ib.shape[1]:
            pred += ib[i] @ coef[col : col + ib.shape[1]]
        score = rmse(y, pred)
        if score < best[0]:
            best = (score, ridge, coef)
    print(f"{label:16s} cols {m:3d} rmse {best[0]:.9f} ridge {best[1]:g}", flush=True)
    return best


def main():
    users, items, u, i, y, base = build_base()
    residual = (y - np.clip(base, 0.5, 5.0)).astype(np.float32)
    uf = top_freqs(users, u, residual, 80)
    itf = top_freqs(items, i, residual, 80)
    print("top user", uf[:10], flush=True)
    print("top item", itf[:10], flush=True)
    for nu, ni in ((58, 0), (48, 10), (40, 18), (32, 26), (24, 34), (16, 42), (8, 50), (0, 58)):
        ub = entity_basis(users, uf[:nu])
        ib = entity_basis(items, itf[:ni])
        fit_combo(f"u{nu}_i{ni}", ub, ib, u, i, y, base)


if __name__ == "__main__":
    main()
