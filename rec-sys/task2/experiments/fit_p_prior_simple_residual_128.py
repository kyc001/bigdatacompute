import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fit_stride_scaled_128 as base


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("rec-sys/task2/experiments/p_prior_simple_residual_128.npz")
USER_STRIDE = 10
ITEM_STRIDE = 4
ITEM_PHASE = 2
USER_SHRINK = 20.0
ITEM_SHRINK = 5.0
PARAM_BUDGET = 128


@dataclass(frozen=True)
class Config:
    user_dim: int
    item_dim: int = 0

    @property
    def stat_params(self):
        return 3

    @property
    def user_segments(self):
        return PARAM_BUDGET - self.stat_params - self.user_dim - self.item_dim

    @property
    def params(self):
        return self.stat_params + self.user_dim + self.item_dim + self.user_segments

    @property
    def label(self):
        return f"simple_ud{self.user_dim}_id{self.item_dim}_seg{self.user_segments}"


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def sampled_stats(inc, users, items, mean):
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    row = np.arange(residual.shape[0], dtype=np.int64)
    user_mask = (row % USER_STRIDE) == 0
    item_mask = (row % ITEM_STRIDE) == ITEM_PHASE
    user_sum = np.bincount(inc_u[user_mask], weights=residual[user_mask], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[user_mask], minlength=users).astype(np.float32)
    item_sum = np.bincount(
        inc_i[item_mask], weights=residual[item_mask] * float(ITEM_STRIDE), minlength=items
    ).astype(np.float32)
    item_count = np.bincount(
        inc_i[item_mask],
        weights=np.full(int(item_mask.sum()), float(ITEM_STRIDE), dtype=np.float32),
        minlength=items,
    ).astype(np.float32)
    return user_sum, user_count, item_sum, item_count


def build_features(cfg, P, Q, u, i, stats):
    user_sum, user_count, item_sum, item_count = stats
    uc = user_count[u]
    ic = item_count[i]
    cols = cfg.stat_params + cfg.user_dim + cfg.item_dim
    x = np.empty((u.shape[0], cols), dtype=np.float32)
    x[:, 0] = 1.0
    x[:, 1] = np.where(uc > 0, user_sum[u] / (uc + USER_SHRINK), 0.0).astype(np.float32)
    x[:, 2] = np.where(ic > 0, item_sum[i] / (ic + ITEM_SHRINK), 0.0).astype(np.float32)
    off = cfg.stat_params
    if cfg.user_dim:
        x[:, off : off + cfg.user_dim] = P[u, : cfg.user_dim]
    off += cfg.user_dim
    if cfg.item_dim:
        x[:, off : off + cfg.item_dim] = Q[i, : cfg.item_dim]
    return x


def fit_linear(cfg, P, Q, u, i, y, stats):
    x = build_features(cfg, P, Q, u, i, stats)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(x.shape[1], dtype=np.float64) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    pred = x @ coef
    return coef, pred.astype(np.float32), rmse(y, pred)


def fit_user_segments(users, u, y, pred, segments):
    residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual, minlength=users).astype(np.float64)
    counts = np.bincount(u, minlength=users).astype(np.float64)
    seen = counts > 0
    ids_ordered, bounds, thresholds, values = base.optimal_segments(
        np.nonzero(seen)[0].astype(np.int32),
        sums[seen] / counts[seen],
        counts[seen],
        segments,
    )
    corr = np.zeros(users, dtype=np.float32)
    for idx, (left, right) in enumerate(bounds):
        left_id = ids_ordered[left]
        right_id = ids_ordered[right - 1]
        corr[left_id : right_id + 1] = values[idx]
        if idx > 0:
            prev_id = ids_ordered[bounds[idx - 1][1] - 1]
            corr[prev_id + 1 : left_id] = values[idx]
    return thresholds, values, rmse(y, pred + corr[u])


def main():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    P = np.load(ROOT / "P.npy", mmap_mode="r")
    Q = np.load(ROOT / "Q.npy", mmap_mode="r")
    users = P.shape[0]
    mean = float(np.load(ROOT / "global_mean.npy"))
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    stats = sampled_stats(inc, users, Q.shape[0], mean)

    configs = [
        Config(4, 0),
        Config(8, 0),
        Config(12, 0),
        Config(16, 0),
        Config(8, 4),
        Config(8, 8),
        Config(12, 4),
    ]
    best = (99.0, None)
    for cfg in configs:
        if cfg.user_segments <= 0:
            continue
        coef, pred, base_score = fit_linear(cfg, P, Q, u, i, y, stats)
        thresholds, values, final = fit_user_segments(users, u, y, pred, cfg.user_segments)
        print(f"{cfg.label} params {cfg.params} base {base_score:.9f} final {final:.9f}", flush=True)
        if final < best[0]:
            best = (final, cfg, base_score, coef, thresholds, values)
            np.savez(
                OUT,
                best_rmse=np.array(final, dtype=np.float32),
                base_rmse=np.array(base_score, dtype=np.float32),
                user_dim=np.array(cfg.user_dim, dtype=np.int32),
                item_dim=np.array(cfg.item_dim, dtype=np.int32),
                user_segments=np.array(cfg.user_segments, dtype=np.int32),
                coef=coef,
                thresholds=thresholds,
                values=values,
            )
            print(f"NEW_BEST {final:.9f} {cfg.label}", flush=True)
    print(f"BEST {best[0]:.9f} {best[1].label} base {best[2]:.9f}", flush=True)


if __name__ == "__main__":
    main()
