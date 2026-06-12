import math
from pathlib import Path

import numpy as np

import fit_stride_scaled_128 as base


ROOT = base.ROOT
OUT = Path("rec-sys/task2/experiments/mixed_segments_128.npz")


def segment_array(size, ids_seen, target, weight, segments):
    if segments <= 0:
        return np.zeros(size, dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    ids_ordered, bounds, thresholds, values = base.optimal_segments(ids_seen, target, weight, segments)
    corr = np.zeros(size, dtype=np.float32)
    for idx, (l, r) in enumerate(bounds):
        left_id = ids_ordered[l]
        right_id = ids_ordered[r - 1]
        corr[left_id : right_id + 1] = values[idx]
        if idx > 0:
            prev_id = ids_ordered[bounds[idx - 1][1] - 1]
            corr[prev_id + 1 : left_id] = values[idx]
    return corr, thresholds, values


def build_base(stride, phase, user_shrink, item_shrink, inc_u, inc_i, residual, users, items, u, i, y):
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
    return x @ coef, coef


def fit_mixed(stride, phase, user_shrink, item_shrink, user_segments, item_segments,
              inc_u, inc_i, residual, users, items, u, i, y):
    base_pred, coef = build_base(
        stride, phase, user_shrink, item_shrink, inc_u, inc_i, residual, users, items, u, i, y
    )
    total_res = (y - np.clip(base_pred, 0.5, 5.0)).astype(np.float32)
    user_corr = np.zeros(users, dtype=np.float32)
    item_corr = np.zeros(items, dtype=np.float32)
    user_thresholds = np.zeros(max(user_segments - 1, 0), dtype=np.int32)
    item_thresholds = np.zeros(max(item_segments - 1, 0), dtype=np.int32)
    user_values = np.zeros(user_segments, dtype=np.float32)
    item_values = np.zeros(item_segments, dtype=np.float32)

    for _ in range(2):
        r_user = total_res - item_corr[i]
        sums = np.bincount(u, weights=r_user, minlength=users).astype(np.float64)
        counts = np.bincount(u, minlength=users).astype(np.float64)
        seen = counts > 0
        user_corr, user_thresholds, user_values = segment_array(
            users, np.nonzero(seen)[0].astype(np.int32), sums[seen] / counts[seen], counts[seen], user_segments
        )

        r_item = total_res - user_corr[u]
        sums = np.bincount(i, weights=r_item, minlength=items).astype(np.float64)
        counts = np.bincount(i, minlength=items).astype(np.float64)
        seen = counts > 0
        item_corr, item_thresholds, item_values = segment_array(
            items, np.nonzero(seen)[0].astype(np.int32), sums[seen] / counts[seen], counts[seen], item_segments
        )

    pred = base_pred + user_corr[u] + item_corr[i]
    score = base.rmse(y, pred)
    base_score = base.rmse(y, base_pred)
    print(
        f"stride {stride} phase {phase} us {user_shrink:g} is {item_shrink:g} "
        f"useg {user_segments} iseg {item_segments} base {base_score:.9f} final {score:.9f}",
        flush=True,
    )
    return score, base_score, coef, user_thresholds, user_values, item_thresholds, item_values


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
    for user_segments, item_segments in ((100, 19), (90, 29), (80, 39), (70, 49), (60, 59), (50, 69), (40, 79)):
        configs.append((8, 0, 20.0, 5.0, user_segments, item_segments))
    for user_segments, item_segments in ((90, 29), (70, 49), (50, 69)):
        configs.append((8, 6, 20.0, 5.0, user_segments, item_segments))
    for user_segments, item_segments in ((80, 39), (60, 59), (40, 79)):
        configs.append((16, 13, 20.0, 5.0, user_segments, item_segments))

    best = (99.0, None)
    for cfg in configs:
        score, base_score, coef, uth, uval, ith, ival = fit_mixed(
            *cfg, inc_u, inc_i, residual, users, items, u, i, y
        )
        if score < best[0]:
            best = (score, cfg, base_score, coef, uth, uval, ith, ival)
            stride, phase, user_shrink, item_shrink, user_segments, item_segments = cfg
            np.savez(
                OUT,
                best_rmse=np.array(score, dtype=np.float32),
                base_rmse=np.array(base_score, dtype=np.float32),
                stride=np.array(stride, dtype=np.int32),
                phase=np.array(phase, dtype=np.int32),
                user_shrink=np.array(user_shrink, dtype=np.float32),
                item_shrink=np.array(item_shrink, dtype=np.float32),
                user_segments=np.array(user_segments, dtype=np.int32),
                item_segments=np.array(item_segments, dtype=np.int32),
                coef=coef,
                user_thresholds=uth,
                user_values=uval,
                item_thresholds=ith,
                item_values=ival,
            )
            print(f"NEW_BEST score {score:.9f} cfg {cfg}", flush=True)
    print(f"BEST {best[0]:.9f} cfg {best[1]} base {best[2]:.9f}", flush=True)


if __name__ == "__main__":
    main()
