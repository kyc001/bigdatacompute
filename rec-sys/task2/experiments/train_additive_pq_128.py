import math
from pathlib import Path

import numpy as np
import torch

import train_factorized_128 as base128


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_BUDGET = 128
USER_SHRINKS = (20.0,)
ITEM_SHRINKS = (5.0,)
KINDS = ("p", "q", "p2", "q2")


def rmse_t(pred, y):
    return math.sqrt(float(torch.mean((torch.clamp(pred, 0.5, 5.0) - y) ** 2).item()))


def feature_block(pt, qt, u, i, kind, dims):
    if kind == "p":
        return pt[u][:, dims]
    if kind == "q":
        return qt[i][:, dims]
    if kind == "p2":
        x = pt[u][:, dims]
        return x * x
    if kind == "q2":
        x = qt[i][:, dims]
        return x * x
    raise KeyError(kind)


def selected_matrix(pt, qt, u, i, selected):
    cols = []
    for kind in KINDS:
        dims = [dim for k, dim in selected if k == kind]
        if dims:
            dim_t = torch.tensor(dims, device=u.device, dtype=torch.long)
            cols.append(feature_block(pt, qt, u, i, kind, dim_t))
    return torch.cat(cols, dim=1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)

    users, u_np, i_np, y_np, us, uc, is_, ic = base128.load_base_arrays()
    cfg = base128.Config(USER_SHRINKS, ITEM_SHRINKS, 4, 4, 1, 0)
    x_stat_np = base128.make_features(cfg, us, uc, is_, ic)
    stat_w, stat_rmse = base128.fit_ridge(x_stat_np, y_np)
    stat_params = x_stat_np.shape[1] + len(USER_SHRINKS) + len(ITEM_SHRINKS)
    dense_budget = PARAM_BUDGET - stat_params
    print(f"stat_rmse {stat_rmse:.9f} stat_params {stat_params} dense_budget {dense_budget}", flush=True)

    P = np.load(ROOT / "P.npy", mmap_mode="r")
    Q = np.load(ROOT / "Q.npy", mmap_mode="r")
    device = torch.device("cuda")
    pt = torch.tensor(np.asarray(P, dtype=np.float32), device=device)
    qt = torch.tensor(np.asarray(Q, dtype=np.float32), device=device)
    u = torch.tensor(u_np, device=device, dtype=torch.long)
    i = torch.tensor(i_np, device=device, dtype=torch.long)
    y = torch.tensor(y_np, device=device)
    x_stat = torch.tensor(x_stat_np, device=device)
    stat_w_t = torch.tensor(stat_w, device=device)
    stat_pred = x_stat.matmul(stat_w_t)
    residual = y - torch.clamp(stat_pred, 0.5, 5.0)
    n = y.numel()
    batch = 32768

    scored = []
    for kind in KINDS:
        k = pt.shape[1]
        sum_x = torch.zeros(k, device=device, dtype=torch.float64)
        sum_x2 = torch.zeros(k, device=device, dtype=torch.float64)
        sum_xr = torch.zeros(k, device=device, dtype=torch.float64)
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            dims = torch.arange(k, device=device, dtype=torch.long)
            x = feature_block(pt, qt, u[sl], i[sl], kind, dims).to(torch.float64)
            r = residual[sl].to(torch.float64)
            sum_x += x.sum(dim=0)
            sum_x2 += (x * x).sum(dim=0)
            sum_xr += x.t().matmul(r)
        mean_r = residual.double().mean()
        cov = sum_xr - sum_x * mean_r
        var = sum_x2 - sum_x * sum_x / float(n)
        score = torch.abs(cov) / torch.sqrt(torch.clamp(var, min=1e-20))
        top_v, top_i = torch.topk(score, min(dense_budget, k))
        for value, dim in zip(top_v.detach().cpu().numpy(), top_i.detach().cpu().numpy()):
            scored.append((float(value), kind, int(dim)))
        print(f"scored {kind}", flush=True)

    scored.sort(reverse=True)
    selected = [(kind, dim) for _, kind, dim in scored[:dense_budget]]
    print(f"selected first20 {selected[:20]}", flush=True)

    m = x_stat_np.shape[1] + len(selected)
    a = torch.zeros((m, m), device=device, dtype=torch.float64)
    b = torch.zeros(m, device=device, dtype=torch.float64)
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        dense = selected_matrix(pt, qt, u[sl], i[sl], selected)
        x = torch.cat([x_stat[sl], dense], dim=1).to(torch.float64)
        a += x.t().matmul(x)
        b += x.t().matmul(y[sl].to(torch.float64))

    best = (99.0, None)
    for ridge in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        w = torch.linalg.solve(a + torch.eye(m, device=device, dtype=torch.float64) * ridge, b).to(torch.float32)
        pred = torch.empty_like(y)
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            dense = selected_matrix(pt, qt, u[sl], i[sl], selected)
            x = torch.cat([x_stat[sl], dense], dim=1)
            pred[sl] = x.matmul(w)
        score = rmse_t(pred, y)
        print(f"ridge {ridge:g} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, w.detach().cpu().numpy().astype(np.float32))

    # Fine-tune the same 128 weights with mini-batch Adam; no hyperparameter sweep.
    w = torch.nn.Parameter(torch.tensor(best[1], device=device))
    opt = torch.optim.Adam([w], lr=0.001)
    lr = 0.001
    stale = 0
    best_score = best[0]
    best_w = best[1].copy()
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    def full_score():
        pred = torch.empty_like(y)
        with torch.no_grad():
            for start in range(0, n, batch):
                sl = slice(start, min(start + batch, n))
                dense = selected_matrix(pt, qt, u[sl], i[sl], selected)
                x = torch.cat([x_stat[sl], dense], dim=1)
                pred[sl] = x.matmul(w)
        return rmse_t(pred, y)

    while True:
        order = all_idx[torch.randperm(n, device=device)]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            dense = selected_matrix(pt, qt, u[ids], i[ids], selected)
            x = torch.cat([x_stat[ids], dense], dim=1)
            pred = x.matmul(w)
            loss = torch.mean((pred - y[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        score = full_score()
        print(f"adam rmse {score:.9f} best {best_score:.9f} lr {lr:.2g}", flush=True)
        if score < best_score - 1e-6:
            best_score = score
            best_w = w.detach().cpu().numpy().astype(np.float32)
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            if lr > 2e-6:
                lr *= 0.3
                opt.param_groups[0]["lr"] = lr
                stale = 0
            else:
                break

    print(f"BEST additive_pq_128 rmse {best_score:.9f} params {stat_params + len(selected)}", flush=True)
    np.savez(
        OUT_DIR / "additive_pq_128.npz",
        best_rmse=np.array(best_score, dtype=np.float32),
        param_count=np.array(stat_params + len(selected), dtype=np.int32),
        stat_coef_count=np.array(x_stat_np.shape[1], dtype=np.int32),
        user_shrinks=np.array(USER_SHRINKS, dtype=np.float32),
        item_shrinks=np.array(ITEM_SHRINKS, dtype=np.float32),
        weights=best_w,
        selected_kind=np.array([k for k, _ in selected]),
        selected_dim=np.array([d for _, d in selected], dtype=np.int32),
    )


if __name__ == "__main__":
    main()
