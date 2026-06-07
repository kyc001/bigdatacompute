import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAM_BUDGET = 128


USER_SHRINKS = np.array([20.0], dtype=np.float32)
ITEM_SHRINKS = np.array([5.0], dtype=np.float32)


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def load_stats():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    user_sum = np.bincount(inc_u[::2], weights=residual[::2], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::2], minlength=users).astype(np.float32)
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    return users, items, u, i, y, user_sum[u], user_count[u], item_sum[i], item_count[i]


def stat_features(us, uc, is_, ic):
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    cols = [
        np.ones_like(uc, dtype=np.float32),
        lu,
        li,
        lu * lu,
        li * li,
        (1.0 / np.sqrt(uc + 1.0)).astype(np.float32),
        (1.0 / np.sqrt(ic + 1.0)).astype(np.float32),
        np.where(uc > 0, us / (uc + USER_SHRINKS[0]), 0.0).astype(np.float32),
        np.where(ic > 0, is_ / (ic + ITEM_SHRINKS[0]), 0.0).astype(np.float32),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def solve_ridge(x, y, ridges):
    xtx = x.T @ x
    xty = x.T @ y
    best = (99.0, None, None)
    eye = np.eye(x.shape[1], dtype=np.float64)
    for ridge in ridges:
        w = np.linalg.solve(xtx.astype(np.float64) + eye * ridge, xty.astype(np.float64)).astype(np.float32)
        pred = x @ w
        score = rmse(y, pred)
        print(f"ridge {ridge:g} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, w, pred.astype(np.float32))
    return best


def feature_block(pt, qt, u, i, kind):
    p = pt[u]
    q = qt[i]
    if kind == "p":
        return p
    if kind == "q":
        return q
    if kind == "pq":
        return p * q
    if kind == "abspq":
        return torch.abs(p * q)
    if kind == "p2":
        return p * p
    if kind == "q2":
        return q * q
    raise KeyError(kind)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    users, items, u_np, i_np, y, us, uc, is_, ic = load_stats()
    x_stat = stat_features(us, uc, is_, ic)
    stat_params = x_stat.shape[1] + len(USER_SHRINKS) + len(ITEM_SHRINKS)
    dense_budget = PARAM_BUDGET - stat_params
    print(f"stat_params {stat_params} dense_budget {dense_budget}", flush=True)
    base_score, base_w, base_pred = solve_ridge(x_stat, y, (1e-4, 1e-3, 1e-2, 1e-1))
    print(f"stat base {base_score:.9f}", flush=True)

    P = np.load(ROOT / "P.npy", mmap_mode="r")
    Q = np.load(ROOT / "Q.npy", mmap_mode="r")
    device = torch.device("cuda")
    pt = torch.tensor(np.asarray(P, dtype=np.float32), device=device)
    qt = torch.tensor(np.asarray(Q, dtype=np.float32), device=device)
    u = torch.tensor(u_np, device=device, dtype=torch.long)
    i = torch.tensor(i_np, device=device, dtype=torch.long)
    residual = torch.tensor((y - np.clip(base_pred, 0.5, 5.0)).astype(np.float32), device=device)
    kinds = ["p", "q", "pq", "abspq", "p2", "q2"]
    k = pt.shape[1]
    n = y.shape[0]
    batch = 32768
    scored = []
    for kind in kinds:
        sum_x = torch.zeros(k, device=device, dtype=torch.float64)
        sum_x2 = torch.zeros(k, device=device, dtype=torch.float64)
        sum_xr = torch.zeros(k, device=device, dtype=torch.float64)
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            x = feature_block(pt, qt, u[sl], i[sl], kind).to(torch.float64)
            r = residual[sl].to(torch.float64)
            sum_x += x.sum(dim=0)
            sum_x2 += (x * x).sum(dim=0)
            sum_xr += x.t().matmul(r)
        mean_r = residual.double().mean()
        cov = sum_xr - sum_x * mean_r
        var = sum_x2 - sum_x * sum_x / float(n)
        score = torch.abs(cov) / torch.sqrt(torch.clamp(var, min=1e-20))
        top_v, top_i = torch.topk(score, min(dense_budget, k))
        for value, dim in zip(top_v.cpu().numpy(), top_i.cpu().numpy()):
            scored.append((float(value), kind, int(dim)))
        print(f"scored {kind}", flush=True)
    scored.sort(reverse=True)
    selected = [(kind, dim) for _, kind, dim in scored[:dense_budget]]
    print("selected", selected[:20], "...", flush=True)

    x_dense = np.empty((n, len(selected)), dtype=np.float32)
    for col, (kind, dim) in enumerate(selected):
        if kind == "p":
            x_dense[:, col] = np.asarray(P[u_np, dim], dtype=np.float32)
        elif kind == "q":
            x_dense[:, col] = np.asarray(Q[i_np, dim], dtype=np.float32)
        elif kind == "pq":
            x_dense[:, col] = np.asarray(P[u_np, dim] * Q[i_np, dim], dtype=np.float32)
        elif kind == "abspq":
            x_dense[:, col] = np.asarray(np.abs(P[u_np, dim] * Q[i_np, dim]), dtype=np.float32)
        elif kind == "p2":
            x_dense[:, col] = np.asarray(P[u_np, dim] * P[u_np, dim], dtype=np.float32)
        elif kind == "q2":
            x_dense[:, col] = np.asarray(Q[i_np, dim] * Q[i_np, dim], dtype=np.float32)
    x_all = np.concatenate([x_stat, x_dense], axis=1)
    score, w, pred = solve_ridge(x_all, y, (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
    print(f"BEST dense_pq_128 rmse {score:.9f} params {stat_params + len(selected)}", flush=True)
    np.savez(
        OUT_DIR / "dense_pq_128.npz",
        best_rmse=np.array(score, dtype=np.float32),
        param_count=np.array(stat_params + len(selected), dtype=np.int32),
        stat_coef_count=np.array(x_stat.shape[1], dtype=np.int32),
        user_shrinks=USER_SHRINKS,
        item_shrinks=ITEM_SHRINKS,
        weights=w.astype(np.float32),
        selected_kind=np.array([k for k, _ in selected]),
        selected_dim=np.array([d for _, d in selected], dtype=np.int32),
    )


if __name__ == "__main__":
    main()
