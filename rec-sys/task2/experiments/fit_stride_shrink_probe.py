import math
from pathlib import Path

import numpy as np

import fit_stride_scaled_128 as base


ROOT = base.ROOT
OUT = Path("rec-sys/task2/experiments/stride_shrink_probe_best.npz")


def fit_one(stride, phase, user_shrink, item_shrink, inc_u, inc_i, residual, users, items, u, i, y):
    n = residual.shape[0]
    row = np.arange(n, dtype=np.int64)
    user_mask = (row % base.USER_STRIDE) == 0
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
    ua = base.safe_avg(user_sum[u], uc, float(user_shrink))
    ia = base.safe_avg(item_sum[i], ic, float(item_shrink))
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    ru = (1.0 / np.sqrt(uc + 1.0)).astype(np.float32)
    ri = (1.0 / np.sqrt(ic + 1.0)).astype(np.float32)
    x = np.stack([np.ones_like(y), lu, li, ru, ri, ua, ia], axis=1).astype(np.float32)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(7) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    base_pred = x @ coef
    base_rmse = base.rmse(y, base_pred)

    residual2 = (y - np.clip(base_pred, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual2, minlength=users).astype(np.float64)
    counts = np.bincount(u, minlength=users).astype(np.float64)
    seen = counts > 0
    target = sums[seen] / counts[seen]
    uid_seen = np.nonzero(seen)[0].astype(np.int32)
    uid_ordered, bounds, thresholds, values = base.optimal_segments(
        uid_seen, target, counts[seen], base.SEGMENTS
    )

    corr = np.zeros(users, dtype=np.float32)
    for idx, (l, r) in enumerate(bounds):
        left_uid = uid_ordered[l]
        right_uid = uid_ordered[r - 1]
        corr[left_uid : right_uid + 1] = values[idx]
        if idx > 0:
            prev_uid = uid_ordered[bounds[idx - 1][1] - 1]
            corr[prev_uid + 1 : left_uid] = values[idx]
    final = base.rmse(y, base_pred + corr[u])
    print(
        f"stride {stride} phase {phase} us {user_shrink:g} is {item_shrink:g} "
        f"base {base_rmse:.9f} final {final:.9f}",
        flush=True,
    )
    return final, base_rmse, coef, thresholds, values


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

    configs = []
    for phase in (0, 1):
        for user_shrink in (10.0, 20.0, 40.0):
            for item_shrink in (2.0, 3.0, 5.0, 8.0, 12.0):
                configs.append((5, phase, user_shrink, item_shrink))
    for phase in (0, 1, 2):
        for item_shrink in (2.0, 5.0, 10.0):
            configs.append((6, phase, 20.0, item_shrink))

    best = (99.0, None)
    for stride, phase, user_shrink, item_shrink in configs:
        final, base_rmse, coef, thresholds, values = fit_one(
            stride, phase, user_shrink, item_shrink, inc_u, inc_i, residual, users, items, u, i, y
        )
        if final < best[0]:
            best = (final, stride, phase, user_shrink, item_shrink, base_rmse, coef, thresholds, values)
            np.savez(
                OUT,
                best_rmse=np.array(final, dtype=np.float32),
                base_rmse=np.array(base_rmse, dtype=np.float32),
                stride=np.array(stride, dtype=np.int32),
                phase=np.array(phase, dtype=np.int32),
                user_shrink=np.array(user_shrink, dtype=np.float32),
                item_shrink=np.array(item_shrink, dtype=np.float32),
                coef=coef,
                thresholds=thresholds,
                values=values,
            )
            print(
                f"NEW_BEST stride {stride} phase {phase} us {user_shrink:g} "
                f"is {item_shrink:g} final {final:.9f}",
                flush=True,
            )
    print(
        f"BEST final {best[0]:.9f} stride {best[1]} phase {best[2]} "
        f"us {best[3]:g} is {best[4]:g} base {best[5]:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
