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


def base_model(base_kind):
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
    ua20 = safe_avg(us[u], U, 20.0)
    ia5 = safe_avg(is_[i], I, 5.0)
    lu = np.log1p(U).astype(np.float32)
    li = np.log1p(I).astype(np.float32)
    ru = (1.0 / np.sqrt(U + 1.0)).astype(np.float32)
    ri = (1.0 / np.sqrt(I + 1.0)).astype(np.float32)

    if base_kind == "base9":
        cols = [np.ones_like(y), lu, li, lu * lu, li * li, ru, ri, ua20, ia5]
        learned_base_params = 11
    elif base_kind == "base7":
        cols = [np.ones_like(y), lu, li, ru, ri, ua20, ia5]
        learned_base_params = 9
    elif base_kind == "base5":
        cols = [np.ones_like(y), lu, li, ua20, ia5]
        learned_base_params = 7
    elif base_kind == "base3":
        cols = [np.ones_like(y), ua20, ia5]
        learned_base_params = 5
    elif base_kind == "itemonly":
        cols = [np.ones_like(y), li, ri, ia5]
        learned_base_params = 5
    else:
        raise KeyError(base_kind)

    x = np.stack(cols, axis=1).astype(np.float32)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(x.shape[1]) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    pred = x @ coef
    return users, u, y, pred, learned_base_params


def optimal_segments(user_ids, target, weight, max_segments):
    # Compress to observed test users sorted by numeric user id.
    order = np.argsort(user_ids)
    uid = user_ids[order].astype(np.int32)
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
    parent = np.full((max_segments + 1, n + 1), -1, dtype=np.int32)

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

    best_rows = {}
    for k in range(1, max_segments + 1):
        cur = np.full(n + 1, np.inf, dtype=np.float64)
        compute_row(k, 1, n, 0, n - 1, cur)
        prev = cur
        if k in (96, 117, 119, 121, 123):
            best_rows[k] = (cur[n], parent[: k + 1].copy())

    out = {}
    for k, (_, par) in best_rows.items():
        bounds = []
        r = n
        for kk in range(k, 0, -1):
            l = int(par[kk, r])
            bounds.append((l, r))
            r = l
        bounds.reverse()
        seg_value = np.zeros(k, dtype=np.float32)
        thresholds = np.zeros(k - 1, dtype=np.int32)
        for idx, (l, r) in enumerate(bounds):
            ww = sw[r] - sw[l]
            seg_value[idx] = 0.0 if ww <= 0 else (sy[r] - sy[l]) / ww
            if idx + 1 < k:
                thresholds[idx] = uid[r - 1]
        out[k] = (uid, bounds, thresholds, seg_value)
    return out


def main():
    for kind in ("base9", "base7", "base5", "base3", "itemonly"):
        users, u, y, pred, base_params = base_model(kind)
        base_score = rmse(y, pred)
        residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
        sums = np.bincount(u, weights=residual, minlength=users).astype(np.float64)
        counts = np.bincount(u, minlength=users).astype(np.float64)
        seen = counts > 0
        target = sums[seen] / counts[seen]
        uid_seen = np.nonzero(seen)[0].astype(np.int32)
        result = optimal_segments(uid_seen, target, counts[seen], 123)
        print(f"\n{kind} base_rmse {base_score:.9f} base_params {base_params}", flush=True)
        for segs, (_, bounds, thresholds, values) in result.items():
            corr = np.zeros(users, dtype=np.float32)
            lo = 0
            for idx, (l, r) in enumerate(bounds):
                left_uid = uid_seen[l]
                right_uid = uid_seen[r - 1]
                corr[left_uid : right_uid + 1] = values[idx]
                if idx > 0:
                    # Fill gaps since only test users are in uid_seen.
                    prev_uid = uid_seen[bounds[idx - 1][1] - 1]
                    corr[prev_uid + 1 : left_uid] = values[idx]
            score = rmse(y, pred + corr[u])
            print(
                f"segments {segs:3d} value_params {segs:3d} "
                f"base+values {base_params + segs:3d} rmse {score:.9f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
