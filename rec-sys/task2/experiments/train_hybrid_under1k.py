import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402
import train_small_param_additive as add  # noqa: E402
import train_small_param_tables as tables  # noqa: E402


PARAM_BUDGET = 999
CAND_TYPES = (
    "p",
    "q",
    "p2",
    "q2",
    "pq",
    "abspq",
    "p_uavg",
    "q_iavg",
    "p_loguc",
    "q_logic",
)


@dataclass
class SparseResult:
    label: str
    pred: np.ndarray
    groups: list[add.Group]
    rmse: float
    params: int


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def fit_sparse(y, base, groups, label):
    groups = [
        add.Group(g.name, g.idx, g.size, g.shrink, np.zeros(g.size, dtype=np.float32))
        for g in groups
    ]
    pred = np.clip(base, 0.5, 5.0).astype(np.float32)
    best = rmse(y, pred)
    best_values = [g.values.copy() for g in groups]
    stale = 0
    epoch = 0
    while stale < add.PATIENCE:
        epoch += 1
        for g in groups:
            pred -= g.values[g.idx]
            residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
            sums = np.bincount(g.idx, weights=residual, minlength=g.size).astype(np.float64)
            counts = np.bincount(g.idx, minlength=g.size).astype(np.float64)
            g.values = (sums / (counts + g.shrink)).astype(np.float32)
            pred += g.values[g.idx]
        score = rmse(y, pred)
        if score < best - add.MIN_DELTA:
            best = score
            best_values = [g.values.copy() for g in groups]
            stale = 0
        else:
            stale += 1
    pred = np.clip(base, 0.5, 5.0).astype(np.float32)
    for g, values in zip(groups, best_values):
        g.values = values
        pred += g.values[g.idx]
    params = sum(g.size for g in groups)
    score = rmse(y, pred)
    print(f"sparse {label} params {params} rmse {score:.9f}", flush=True)
    return SparseResult(label, pred, groups, score, params)


def feature_block(pt, qt, uids, iids, kind, values_t):
    p = pt[uids]
    q = qt[iids]
    if kind == "p":
        return p
    if kind == "q":
        return q
    if kind == "p2":
        return p * p
    if kind == "q2":
        return q * q
    if kind == "pq":
        return p * q
    if kind == "abspq":
        return torch.abs(p * q)
    if kind == "p_uavg":
        return p * values_t["uavg0"][uids].unsqueeze(1)
    if kind == "q_iavg":
        return q * values_t["iavg0_item"][iids].unsqueeze(1)
    if kind == "p_loguc":
        return p * values_t["log_uc_user"][uids].unsqueeze(1)
    if kind == "q_logic":
        return q * values_t["log_ic_item"][iids].unsqueeze(1)
    raise KeyError(kind)


def selected_matrix(pt, qt, uids, iids, selected, values_t):
    cols = []
    for kind in CAND_TYPES:
        dims = [dim for k, dim in selected if k == kind]
        if not dims:
            continue
        dim_t = torch.tensor(dims, device=uids.device, dtype=torch.long)
        block = feature_block(pt, qt, uids, iids, kind, values_t).index_select(1, dim_t)
        cols.append(block)
    return torch.cat(cols, dim=1)


def dense_feature_search(P_m, Q_m, u_all, i_all, y, sparse, values):
    remaining = PARAM_BUDGET - sparse.params
    if remaining <= 0:
        return sparse.rmse, [], np.zeros(0, dtype=np.float32)
    n_select = remaining
    device = torch.device("cuda")
    pt = torch.tensor(np.asarray(P_m, dtype=np.float32), device=device)
    qt = torch.tensor(np.asarray(Q_m, dtype=np.float32), device=device)
    u_t = torch.tensor(u_all, device=device, dtype=torch.long)
    i_t = torch.tensor(i_all, device=device, dtype=torch.long)
    y_t = torch.tensor(y, device=device)
    sparse_t = torch.tensor(sparse.pred, device=device)

    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    users, items = P_m.shape[0], Q_m.shape[0]
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    res = inc[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(inc_i, weights=res, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    user_sum = np.bincount(inc_u[::2], weights=res[::2], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::2], minlength=users).astype(np.float32)
    uavg = np.where(user_count > 0, user_sum / np.maximum(user_count, 1), 0.0).astype(np.float32)
    iavg = np.where(item_count > 0, item_sum / np.maximum(item_count, 1), 0.0).astype(np.float32)
    values_t = {
        "uavg0": torch.tensor(uavg, device=device),
        "iavg0_item": torch.tensor(iavg, device=device),
        "log_uc_user": torch.tensor(np.log1p(user_count).astype(np.float32), device=device),
        "log_ic_item": torch.tensor(np.log1p(item_count).astype(np.float32), device=device),
    }

    residual_t = y_t - torch.clamp(sparse_t, 0.5, 5.0)
    k = P_m.shape[1]
    sum_x = {kind: torch.zeros(k, device=device, dtype=torch.float64) for kind in CAND_TYPES}
    sum_x2 = {kind: torch.zeros(k, device=device, dtype=torch.float64) for kind in CAND_TYPES}
    sum_xr = {kind: torch.zeros(k, device=device, dtype=torch.float64) for kind in CAND_TYPES}

    batch = 32768
    for start in range(0, y.shape[0], batch):
        sl = slice(start, min(start + batch, y.shape[0]))
        ids_u = u_t[sl]
        ids_i = i_t[sl]
        r = residual_t[sl].to(torch.float64)
        for kind in CAND_TYPES:
            x = feature_block(pt, qt, ids_u, ids_i, kind, values_t).to(torch.float64)
            sum_x[kind] += x.sum(dim=0)
            sum_x2[kind] += (x * x).sum(dim=0)
            sum_xr[kind] += x.t().matmul(r)

    n = float(y.shape[0])
    mean_r = residual_t.double().mean()
    scored = []
    for kind in CAND_TYPES:
        cov = sum_xr[kind] - sum_x[kind] * mean_r
        var = sum_x2[kind] - sum_x[kind] * sum_x[kind] / n
        score = torch.abs(cov) / torch.sqrt(torch.clamp(var, min=1e-20))
        top_v, top_i = torch.topk(score, min(n_select, k))
        for value, dim in zip(top_v.detach().cpu().numpy(), top_i.detach().cpu().numpy()):
            scored.append((float(value), kind, int(dim)))
    scored.sort(reverse=True)
    selected = [(kind, dim) for _, kind, dim in scored[:n_select]]
    print(f"dense selected {len(selected)} remaining {remaining}", flush=True)

    m = len(selected)
    a = torch.zeros((m, m), device=device, dtype=torch.float64)
    b = torch.zeros(m, device=device, dtype=torch.float64)
    r_all = (y_t - sparse_t).to(torch.float64)
    for start in range(0, y.shape[0], batch):
        sl = slice(start, min(start + batch, y.shape[0]))
        x = selected_matrix(pt, qt, u_t[sl], i_t[sl], selected, values_t).to(torch.float64)
        a += x.t().matmul(x)
        b += x.t().matmul(r_all[sl])

    best = sparse.rmse
    best_w = None
    for ridge in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        reg = a + torch.eye(m, device=device, dtype=torch.float64) * ridge
        w = torch.linalg.solve(reg, b).to(torch.float32)
        pred_dense = sparse_t.clone()
        for start in range(0, y.shape[0], batch):
            sl = slice(start, min(start + batch, y.shape[0]))
            x = selected_matrix(pt, qt, u_t[sl], i_t[sl], selected, values_t)
            pred_dense[sl] += x.matmul(w)
        score = math.sqrt(float(torch.mean((torch.clamp(pred_dense, 0.5, 5.0) - y_t) ** 2).item()))
        print(f"ridge {ridge:g} rmse {score:.9f}", flush=True)
        if score < best:
            best = score
            best_w = w.detach().cpu().numpy().astype(np.float32)
    if best_w is None:
        best_w = np.zeros(m, dtype=np.float32)
    return best, selected, best_w


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    values = tables.build_stats_features(P_m.shape[0], Q_m.shape[0])
    values["base"] = np.clip(base, 0.5, 5.0).astype(np.float32)
    specs = tables.build_bins(values)
    u = values["uid"].astype(np.int32)
    i = values["iid"].astype(np.int32)

    def g1(name, shrink=20.0):
        idx, size = specs[name]
        return add.make_group(name, idx, size, shrink)

    def gp(names, shrink=50.0):
        idx, size = add.combined(specs, names)
        return add.make_group("x".join(names), idx, size, shrink)

    configs = [
        (
            "hash_sparse_640",
            [
                add.make_group("uh512_s0", add.hash_bins(u, 512, 0), 512, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
                g1("uavg30q32", 80.0),
            ],
        ),
        (
            "stat_sparse_640",
            [
                gp(("uid64", "uavg0q8"), 10.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
                g1("uavg30q32", 80.0),
            ],
        ),
        (
            "hash_sparse_768",
            [
                add.make_group("uh512_s0", add.hash_bins(u, 512, 0), 512, 20.0),
                g1("iid128", 10.0),
                g1("base64", 50.0),
                g1("uavg30q32", 80.0),
                g1("iavg8q32", 50.0),
            ],
        ),
        (
            "stat_sparse_768",
            [
                gp(("uid64", "uavg0q8"), 10.0),
                g1("iid128", 10.0),
                g1("base64", 50.0),
                g1("uavg30q32", 80.0),
                g1("iavg8q32", 50.0),
            ],
        ),
    ]

    best = (99.0, "", [], np.zeros(0, dtype=np.float32), None)
    for label, groups in configs:
        sparse = fit_sparse(y, base, groups, label)
        score, selected, weights = dense_feature_search(P_m, Q_m, u_all, i_all, y, sparse, values)
        print(f"HYBRID {label} rmse {score:.9f} params {sparse.params + len(weights)}", flush=True)
        if score < best[0]:
            best = (score, label, selected, weights, sparse)

    score, label, selected, weights, sparse = best
    out_path = OUT_DIR / "hybrid_under1k.npz"
    save = {
        "best_rmse": np.array(score, dtype=np.float32),
        "label": np.array(label),
        "param_count": np.array(sparse.params + len(weights), dtype=np.int32),
        "selected_kind": np.array([k for k, _ in selected]),
        "selected_dim": np.array([d for _, d in selected], dtype=np.int32),
        "dense_weights": weights,
        "group_names": np.array([g.name for g in sparse.groups]),
        "group_sizes": np.array([g.size for g in sparse.groups], dtype=np.int32),
        "group_shrinks": np.array([g.shrink for g in sparse.groups], dtype=np.float32),
    }
    for idx, group in enumerate(sparse.groups):
        save[f"group_values_{idx}"] = group.values
    np.savez(out_path, **save)
    print(f"BEST hybrid {label} rmse {score:.9f} params {sparse.params + len(weights)} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
