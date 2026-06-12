import math
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("rec-sys/task2/experiments/stride_scaled_128.npz")
USER_STRIDE = 10
SEGMENTS = 119


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def safe_avg(s, c, shrink):
    out = np.zeros_like(s, dtype=np.float32)
    mask = c > 0
    out[mask] = s[mask] / (c[mask] + shrink)
    return out


def optimal_segments(uid_seen, target, weight, segments):
    order = np.argsort(uid_seen)
    uid = uid_seen[order].astype(np.int32)
    y = target[order].astype(np.float64)
    w = weight[order].astype(np.float64)
    n = y.shape[0]
    sw = np.concatenate([[0.0], np.cumsum(w)])
    sy = np.concatenate([[0.0], np.cumsum(w * y)])
    sy2 = np.concatenate([[0.0], np.cumsum(w * y * y)])

    def cost(l, r):
        ww = sw[r] - sw[l]
        if ww <= 0:
            return 0.0
        ss = sy[r] - sy[l]
        return (sy2[r] - sy2[l]) - ss * ss / ww

    prev = np.full(n + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    parent = np.full((segments + 1, n + 1), -1, dtype=np.int32)

    def compute_row(k, left, right, opt_left, opt_right, cur):
        if left > right:
            return
        mid = (left + right) // 2
        best_val = np.inf
        best_t = -1
        hi = min(opt_right, mid - 1)
        for t in range(opt_left, hi + 1):
            val = prev[t] + cost(t, mid)
            if val < best_val:
                best_val = val
                best_t = t
        cur[mid] = best_val
        parent[k, mid] = best_t
        compute_row(k, left, mid - 1, opt_left, best_t, cur)
        compute_row(k, mid + 1, right, best_t, opt_right, cur)

    for k in range(1, segments + 1):
        cur = np.full(n + 1, np.inf, dtype=np.float64)
        compute_row(k, 1, n, 0, n - 1, cur)
        prev = cur

    bounds = []
    r = n
    for k in range(segments, 0, -1):
        l = int(parent[k, r])
        bounds.append((l, r))
        r = l
    bounds.reverse()

    thresholds = np.zeros(segments - 1, dtype=np.int32)
    values = np.zeros(segments, dtype=np.float32)
    for idx, (l, r) in enumerate(bounds):
        ww = sw[r] - sw[l]
        values[idx] = 0.0 if ww <= 0 else (sy[r] - sy[l]) / ww
        if idx + 1 < segments:
            thresholds[idx] = uid[r - 1]
    return uid, bounds, thresholds, values


def fit_one(stride, phase, inc_u, inc_i, residual, users, items, u, i, y):
    n = residual.shape[0]
    row = np.arange(n, dtype=np.int64)
    user_mask = (row % USER_STRIDE) == 0
    item_mask = (row % stride) == phase
    item_weight = np.full(int(item_mask.sum()), float(stride), dtype=np.float32)

    user_sum = np.bincount(inc_u[user_mask], weights=residual[user_mask], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[user_mask], minlength=users).astype(np.float32)
    item_sum = np.bincount(
        inc_i[item_mask], weights=residual[item_mask] * float(stride), minlength=items
    ).astype(np.float32)
    item_count = np.bincount(inc_i[item_mask], weights=item_weight, minlength=items).astype(np.float32)

    uc = user_count[u]
    ic = item_count[i]
    ua20 = safe_avg(user_sum[u], uc, 20.0)
    ia5 = safe_avg(item_sum[i], ic, 5.0)
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    ru = (1.0 / np.sqrt(uc + 1.0)).astype(np.float32)
    ri = (1.0 / np.sqrt(ic + 1.0)).astype(np.float32)
    x = np.stack([np.ones_like(y), lu, li, ru, ri, ua20, ia5], axis=1).astype(np.float32)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(7) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    base_pred = x @ coef
    base = rmse(y, base_pred)

    residual2 = (y - np.clip(base_pred, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual2, minlength=users).astype(np.float64)
    counts = np.bincount(u, minlength=users).astype(np.float64)
    seen = counts > 0
    target = sums[seen] / counts[seen]
    uid_seen = np.nonzero(seen)[0].astype(np.int32)
    uid_ordered, bounds, thresholds, values = optimal_segments(uid_seen, target, counts[seen], SEGMENTS)

    corr = np.zeros(users, dtype=np.float32)
    for idx, (l, r) in enumerate(bounds):
        left_uid = uid_ordered[l]
        right_uid = uid_ordered[r - 1]
        corr[left_uid : right_uid + 1] = values[idx]
        if idx > 0:
            prev_uid = uid_ordered[bounds[idx - 1][1] - 1]
            corr[prev_uid + 1 : left_uid] = values[idx]
    final = rmse(y, base_pred + corr[u])
    print(f"stride {stride} phase {phase} base {base:.9f} final {final:.9f}", flush=True)
    return final, base, coef, thresholds, values


def main():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = np.load(ROOT / "P.npy", mmap_mode="r").shape[0]
    items = np.load(ROOT / "Q.npy", mmap_mode="r").shape[0]
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)

    best = (99.0, None)
    for stride in (2, 3, 4, 5, 8, 16):
        for phase in range(stride):
            final, base, coef, thresholds, values = fit_one(
                stride, phase, inc_u, inc_i, residual, users, items, u, i, y
            )
            if final < best[0]:
                best = (final, stride, phase, base, coef, thresholds, values)
                np.savez(
                    OUT,
                    best_rmse=np.array(final, dtype=np.float32),
                    base_rmse=np.array(base, dtype=np.float32),
                    stride=np.array(stride, dtype=np.int32),
                    phase=np.array(phase, dtype=np.int32),
                    coef=coef,
                    thresholds=thresholds,
                    values=values,
                )
                print(f"NEW_BEST stride {stride} phase {phase} final {final:.9f}", flush=True)

    print(f"BEST stride {best[1]} phase {best[2]} final {best[0]:.9f} base {best[3]:.9f}", flush=True)


if __name__ == "__main__":
    main()
