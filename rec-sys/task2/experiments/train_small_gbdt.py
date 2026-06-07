import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402
import train_small_param_tables as tables  # noqa: E402


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def make_features(P_m, Q_m, base):
    values = tables.build_stats_features(P_m.shape[0], Q_m.shape[0])
    u = values["uid"].astype(np.int64)
    i = values["iid"].astype(np.int64)
    n = u.shape[0]
    cols = [
        np.clip(base, 0.5, 5.0).astype(np.float32),
        (u.astype(np.float32) / float(P_m.shape[0])),
        (i.astype(np.float32) / float(Q_m.shape[0])),
        values["uavg0"],
        values["iavg0"],
        values["uavg30"],
        values["iavg8"],
        values["log_uc"],
        values["log_ic"],
        (values["uavg0"] * values["iavg0"]).astype(np.float32),
        (values["uavg30"] * values["iavg8"]).astype(np.float32),
    ]

    dims = [0, 1, 2, 3, 4, 7, 15, 31, 63, 127, 255, 511, 767, 1023]
    P = np.asarray(P_m[:, dims], dtype=np.float32)
    Q = np.asarray(Q_m[:, dims], dtype=np.float32)
    for start in range(0, len(dims), 4):
        d = slice(start, start + 4)
        p = P[u, d]
        q = Q[i, d]
        cols.append(p.sum(axis=1).astype(np.float32))
        cols.append(q.sum(axis=1).astype(np.float32))
        cols.append((p * q).sum(axis=1).astype(np.float32))

    segs = [(0, 4), (4, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 768), (768, 1024)]
    for lo, hi in segs:
        out = np.empty(n, dtype=np.float32)
        for start in range(0, n, 65536):
            end = min(start + 65536, n)
            p = np.asarray(P_m[u[start:end], lo:hi], dtype=np.float32)
            q = np.asarray(Q_m[i[start:end], lo:hi], dtype=np.float32)
            out[start:end] = np.sum(p * q, axis=1)
        cols.append(out)

    return np.column_stack(cols).astype(np.float32)


def train_loop(X, y, base, max_leaf_nodes, leaf_budget):
    target = (y - np.clip(base, 0.5, 5.0)).astype(np.float32)
    step = 10
    max_trees = max(1, leaf_budget // max_leaf_nodes)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=0.02,
        min_samples_leaf=80,
        early_stopping=False,
        warm_start=True,
        random_state=20260605,
    )
    best = rmse(y, base)
    best_iter = 0
    best_pred = None
    stale = 0
    trees = 0
    while trees < max_trees and stale < 4:
        trees = min(trees + step, max_trees)
        model.set_params(max_iter=trees)
        model.fit(X, target)
        correction = model.predict(X).astype(np.float32)
        score = rmse(y, base + correction)
        leaves = trees * max_leaf_nodes
        print(f"leaf_nodes {max_leaf_nodes} trees {trees} leaves<= {leaves} rmse {score:.9f}", flush=True)
        if score < best - 2e-5:
            best = score
            best_iter = trees
            best_pred = correction
            stale = 0
        else:
            stale += 1
    return best, best_iter, best_pred


def main():
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    _, _, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    print("building features", flush=True)
    X = make_features(P_m, Q_m, base)
    print(f"X {X.shape} base {rmse(y, base):.9f}", flush=True)

    results = []
    for max_leaf_nodes, leaf_budget in ((4, 960), (6, 960), (8, 960), (10, 950), (16, 960)):
        score, best_iter, pred = train_loop(X, y, base, max_leaf_nodes, leaf_budget)
        results.append((score, max_leaf_nodes, best_iter))
        print(f"CONFIG leaves {max_leaf_nodes} best_iter {best_iter} rmse {score:.9f}", flush=True)

    results.sort()
    out_path = OUT_DIR / "small_gbdt_under1k.npz"
    np.savez(
        out_path,
        best_rmse=np.array(results[0][0], dtype=np.float32),
        max_leaf_nodes=np.array(results[0][1], dtype=np.int32),
        best_iter=np.array(results[0][2], dtype=np.int32),
        all_results=np.array(results, dtype=np.float32),
    )
    print(f"BEST gbdt rmse {results[0][0]:.9f} leaves {results[0][1]} iter {results[0][2]} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
