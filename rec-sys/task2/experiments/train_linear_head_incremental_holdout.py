import math
import time
from pathlib import Path

import numpy as np
import torch

import train_generalized_head as common


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stats_from_rows(rows, users, items, mean):
    u = rows[:, 0].astype(np.int32)
    i = rows[:, 1].astype(np.int32)
    residual = rows[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(i, minlength=items).astype(np.float32)
    even = np.arange(rows.shape[0]) % 2 == 0
    user_sum = np.bincount(u[even], weights=residual[even], minlength=users).astype(np.float32)
    user_count = np.bincount(u[even], minlength=users).astype(np.float32)
    return user_sum, user_count, item_sum, item_count


def base_for_rows(rows, user_sum, user_count, item_sum, item_count):
    u = rows[:, 0].astype(np.int32)
    i = rows[:, 1].astype(np.int32)
    uc = user_count[u]
    ic = item_count[i]
    us = user_sum[u]
    is_ = item_sum[i]
    lu = np.log1p(uc)
    li = np.log1p(ic)
    coef = common.BASE_COEF
    score = np.full(u.shape, coef[0], dtype=np.float32)
    score += (
        coef[1] * lu
        + coef[2] * li
        + coef[3] * lu * lu
        + coef[4] * li * li
        + coef[5] / np.sqrt(uc + 1)
        + coef[6] / np.sqrt(ic + 1)
    )
    offset = 7
    for j, shrink in enumerate(common.USER_SHRINKS):
        score += coef[offset + j] * np.where(uc > 0, us / (uc + shrink), 0).astype(np.float32)
    offset += len(common.USER_SHRINKS)
    for j, shrink in enumerate(common.ITEM_SHRINKS):
        score += coef[offset + j] * np.where(ic > 0, is_ / (ic + shrink), 0).astype(np.float32)
    return score


def train_linear(P_m, Q_m, rows, base_np, epochs=80):
    K = 1024
    u_np = rows[:, 0].astype(np.int64)
    i_np = rows[:, 1].astype(np.int64)
    y_np = rows[:, 2].astype(np.float32)
    P = np.asarray(P_m[:, :K], dtype=np.float32)
    Q = np.asarray(Q_m[:, :K], dtype=np.float32)
    p_mu = P.mean(axis=0).astype(np.float32)
    p_std = P.std(axis=0).astype(np.float32)
    q_mu = Q.mean(axis=0).astype(np.float32)
    q_std = Q.std(axis=0).astype(np.float32)
    p_std[p_std < 1e-6] = 1.0
    q_std[q_std < 1e-6] = 1.0
    P = (P - p_mu) / p_std
    Q = (Q - q_mu) / q_std

    device = torch.device("cuda")
    Pt = torch.tensor(P, device=device)
    Qt = torch.tensor(Q, device=device)
    u_t = torch.tensor(u_np, device=device, dtype=torch.long)
    i_t = torch.tensor(i_np, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(base_np, device=device)
    n = y_np.shape[0]
    all_idx = torch.arange(n, device=device, dtype=torch.long)
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    user_w = torch.nn.Parameter(torch.zeros(K, device=device))
    item_w = torch.nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.AdamW([bias, user_w, item_w], lr=0.001, weight_decay=1e-7)

    def residual(ids):
        return bias + Pt[u_t[ids]].matmul(user_w) + Qt[i_t[ids]].matmul(item_w)

    def rmse():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 131_072):
                ids = all_idx[start : start + 131_072]
                pred = torch.clamp(base_t[ids] + residual(ids), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = rmse()
    best_state = None
    print(f"holdout train rows {n} initial {best:.9f}", flush=True)
    for epoch in range(1, epochs + 1):
        if epoch == 31:
            for group in opt.param_groups:
                group["lr"] = 0.00035
        if epoch == 56:
            for group in opt.param_groups:
                group["lr"] = 0.00012
        order = all_idx[torch.randperm(n, device=device)]
        for start in range(0, n, 65_536):
            ids = order[start : start + 65_536]
            loss = torch.mean((base_t[ids] + residual(ids) - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([bias, user_w, item_w], 5.0)
            opt.step()
        if epoch <= 5 or epoch % 5 == 0:
            r = rmse()
            print(f"epoch {epoch} holdout_rmse {r:.9f}", flush=True)
            if r < best:
                best = r
                best_state = (
                    float(bias.detach().cpu()),
                    user_w.detach().cpu().numpy().copy(),
                    item_w.detach().cpu().numpy().copy(),
                )

    if best_state is None:
        best_state = (
            float(bias.detach().cpu()),
            user_w.detach().cpu().numpy().copy(),
            item_w.detach().cpu().numpy().copy(),
        )
    return best, best_state, p_mu, p_std, q_mu, q_std


def evaluate_test(P_m, Q_m, state, p_mu, p_std, q_mu, q_std):
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    u_all, i_all, y_np, base_np = common.build_base_prediction(P_m.shape, Q_m.shape)
    bias, user_w, item_w = state
    user_coef = user_w / p_std
    item_coef = item_w / q_std
    adjusted_bias = bias - float(np.dot(p_mu, user_coef)) - float(np.dot(q_mu, item_coef))
    user_static = np.asarray(P_m[:, :1024], dtype=np.float32).dot(user_coef.astype(np.float32))
    item_static = np.asarray(Q_m[:, :1024], dtype=np.float32).dot(item_coef.astype(np.float32))
    se = 0.0
    n = test.shape[0]
    for start in range(0, n, 250_000):
        end = min(start + 250_000, n)
        u = u_all[start:end]
        i = i_all[start:end]
        pred = np.clip(base_np[start:end] + adjusted_bias + user_static[u] + item_static[i], 0.5, 5.0)
        err = pred - y_np[start:end]
        se += float(np.sum(err * err))
    return math.sqrt(se / n)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    users, items = P_m.shape[0], Q_m.shape[0]

    split = int(inc.shape[0] * 0.8)
    train_rows = np.asarray(inc[:split], dtype=np.float32)
    holdout_rows = np.asarray(inc[split:], dtype=np.float32)
    user_sum, user_count, item_sum, item_count = stats_from_rows(train_rows, users, items, mean)
    holdout_base = base_for_rows(holdout_rows, user_sum, user_count, item_sum, item_count)
    best, state, p_mu, p_std, q_mu, q_std = train_linear(P_m, Q_m, holdout_rows, holdout_base)
    test_rmse = evaluate_test(P_m, Q_m, state, p_mu, p_std, q_mu, q_std)
    print(f"incremental-only best_holdout {best:.9f} test_rmse {test_rmse:.9f}", flush=True)

    out_path = OUT_DIR / "generalized_linear_incremental_holdout_k1024.npz"
    np.savez(
        out_path,
        K=np.array(1024, dtype=np.int32),
        R=np.array(0, dtype=np.int32),
        bias=np.array(state[0], dtype=np.float32),
        user_w=state[1].astype(np.float32),
        item_w=state[2].astype(np.float32),
        user_proj=np.zeros((1024, 0), dtype=np.float32),
        item_proj=np.zeros((1024, 0), dtype=np.float32),
        p_mu=p_mu,
        p_std=p_std,
        q_mu=q_mu,
        q_std=q_std,
        base_coef=common.BASE_COEF,
        best_rmse=np.array(test_rmse, dtype=np.float32),
    )
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
