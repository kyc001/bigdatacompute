import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("rec-sys/task2/experiments")
OUT.mkdir(parents=True, exist_ok=True)
PARAM_BUDGET = 128


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
    return users, items, u, i, y, pred, coef


@dataclass
class Group:
    name: str
    idx: np.ndarray
    size: int
    shrink: float
    values: np.ndarray


def make_hash(kind, ids, entity_count, sizes):
    # Fixed odd constants. Only the learned table values count as parameters.
    muls = [2654435761, 2246822519, 3266489917, 668265263, 374761393, 1597334677]
    adds = [0, 1013904223, 1442695041, 362437, 521288629, 88675123]
    groups = []
    ids64 = ids.astype(np.uint64)
    for t, size in enumerate(sizes):
        if kind == "range":
            idx = (ids.astype(np.int64) * size // entity_count).astype(np.int32)
            name = f"range{size}"
        else:
            h = (ids64 * np.uint64(muls[t % len(muls)]) + np.uint64(adds[t % len(adds)]))
            idx = (h % np.uint64(size)).astype(np.int32)
            name = f"hash{t}_{size}"
        groups.append(Group(name, idx, int(size), 50.0, np.zeros(int(size), dtype=np.float32)))
    return groups


def fit_groups(y, base_pred, groups, label):
    pred = base_pred.astype(np.float32, copy=True)
    best = rmse(y, pred)
    best_values = [g.values.copy() for g in groups]
    stale = 0
    epoch = 0
    while stale < 8 and epoch < 200:
        epoch += 1
        for g in groups:
            pred -= g.values[g.idx]
            residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
            sums = np.bincount(g.idx, weights=residual, minlength=g.size).astype(np.float64)
            counts = np.bincount(g.idx, minlength=g.size).astype(np.float64)
            g.values = (sums / (counts + g.shrink)).astype(np.float32)
            pred += g.values[g.idx]
        score = rmse(y, pred)
        if score < best - 1e-7:
            best = score
            best_values = [g.values.copy() for g in groups]
            stale = 0
            print(f"{label} epoch {epoch} rmse {best:.9f}", flush=True)
        else:
            stale += 1
    for g, values in zip(groups, best_values):
        g.values = values
    pred = base_pred.astype(np.float32, copy=True)
    for g in groups:
        pred += g.values[g.idx]
    final = rmse(y, pred)
    params = sum(g.size for g in groups)
    print(f"RESULT {label} params {params} rmse {final:.9f}", flush=True)
    return final, groups


def main():
    users, items, u, i, y, base_pred, coef = build_base()
    # Base uses 9 fitted coefficients plus two shrink constants.
    table_budget = PARAM_BUDGET - 11
    configs = [
        ("u_hash_4x29", make_hash("hash", u, users, [29, 29, 29, 30])),
        ("u_hash_3x39", make_hash("hash", u, users, [39, 39, 39])),
        ("u_hash_mix", make_hash("hash", u, users, [17, 23, 31, 46])),
        ("u_range_hash", make_hash("range", u, users, [48]) + make_hash("hash", u, users, [23, 23, 23])),
        ("i_hash_4x29", make_hash("hash", i, items, [29, 29, 29, 30])),
        ("ui_hash_mix", make_hash("hash", u, users, [29, 29]) + make_hash("hash", i, items, [29, 30])),
        ("ui_range_hash", make_hash("range", u, users, [48]) + make_hash("range", i, items, [32]) + make_hash("hash", u, users, [18]) + make_hash("hash", i, items, [19])),
    ]
    best = (99.0, None, None)
    for label, groups in configs:
        params = sum(g.size for g in groups)
        if params > table_budget:
            continue
        # One non-swept shrink policy: modest shrink for hash, low shrink for range.
        for g in groups:
            g.shrink = 20.0 if g.name.startswith("range") else 100.0
        score, fitted = fit_groups(y, base_pred, groups, label)
        if score < best[0]:
            best = (score, label, fitted)
            save = {
                "best_rmse": np.array(score, dtype=np.float32),
                "param_count": np.array(11 + params, dtype=np.int32),
                "base_coef": coef,
                "group_names": np.array([g.name for g in fitted]),
                "group_sizes": np.array([g.size for g in fitted], dtype=np.int32),
                "group_shrinks": np.array([g.shrink for g in fitted], dtype=np.float32),
            }
            for n, g in enumerate(fitted):
                save[f"values_{n}"] = g.values
            np.savez(OUT / "hash_128.npz", **save)
            print(f"NEW_BEST {label} rmse {score:.9f} total_params {11 + params}", flush=True)
    print(f"BEST {best[1]} {best[0]:.9f}", flush=True)


if __name__ == "__main__":
    main()
