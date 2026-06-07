import gc
import math
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


BASE_COEF = np.array(
    [
        4.430029912552816,
        -0.35739037763299225,
        -0.010405138231289704,
        0.03385049751798333,
        0.001317321440029219,
        -0.8801704691948185,
        0.0439380115327232,
        -6.011119557954283,
        153.37655860941354,
        -1177.6176880917649,
        7316.511532627338,
        -20717.965609694682,
        22234.172758447374,
        -10147.896487977032,
        2736.3039826653953,
        -448.4622900505777,
        59.092543771084046,
        -20.096646750173583,
        533.1419247737755,
        -3070.249607487788,
        5553.590534031688,
        -553.3363028957483,
        -4939.89517163505,
        4118.89690808396,
        -1978.9294617524615,
        376.029227050923,
        -18.264405038720056,
    ],
    dtype=np.float32,
)

USER_SHRINKS = np.array([0, 2, 5, 10, 15, 20, 30, 50, 100, 200], dtype=np.float32)
ITEM_SHRINKS = np.array([0, 1, 2, 3, 4, 5, 8, 12, 20, 50], dtype=np.float32)


def build_base_prediction(P_shape, Q_shape):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    users, items = P_shape[0], Q_shape[0]

    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    user_sum = np.bincount(inc_u[::2], weights=residual[::2], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::2], minlength=users).astype(np.float32)

    def base_for(u, i):
        uc = user_count[u]
        ic = item_count[i]
        us = user_sum[u]
        is_ = item_sum[i]
        lu = np.log1p(uc)
        li = np.log1p(ic)
        score = np.full(u.shape, BASE_COEF[0], dtype=np.float32)
        score += (
            BASE_COEF[1] * lu
            + BASE_COEF[2] * li
            + BASE_COEF[3] * lu * lu
            + BASE_COEF[4] * li * li
            + BASE_COEF[5] / np.sqrt(uc + 1)
            + BASE_COEF[6] / np.sqrt(ic + 1)
        )
        offset = 7
        for j, shrink in enumerate(USER_SHRINKS):
            score += BASE_COEF[offset + j] * np.where(uc > 0, us / (uc + shrink), 0).astype(np.float32)
        offset += len(USER_SHRINKS)
        for j, shrink in enumerate(ITEM_SHRINKS):
            score += BASE_COEF[offset + j] * np.where(ic > 0, is_ / (ic + shrink), 0).astype(np.float32)
        return score

    n = test.shape[0]
    u_all = test[:, 0].astype(np.int64)
    i_all = test[:, 1].astype(np.int64)
    y = test[:, 2].astype(np.float32)
    base = np.empty(n, dtype=np.float32)
    for start in range(0, n, 250_000):
        end = min(start + 250_000, n)
        base[start:end] = base_for(u_all[start:end].astype(np.int32), i_all[start:end].astype(np.int32))

    rmse = math.sqrt(float(np.mean((y - np.clip(base, 0.5, 5.0)) ** 2)))
    print(f"base rmse {rmse:.9f}", flush=True)
    return u_all, i_all, y, base


def train_config(P_m, Q_m, u_all, i_all, y_np, base_np, train_idx_np, val_idx_np, K, R, epochs):
    print(f"\nCONFIG K={K} R={R} epochs={epochs}", flush=True)
    t0 = time.time()

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
    u_t = torch.tensor(u_all, device=device, dtype=torch.long)
    i_t = torch.tensor(i_all, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(base_np, device=device)
    train_idx = torch.tensor(train_idx_np, device=device, dtype=torch.long)
    val_idx = torch.tensor(val_idx_np, device=device, dtype=torch.long)

    bias = torch.nn.Parameter(torch.zeros((), device=device))
    user_w = torch.nn.Parameter(torch.zeros(K, device=device))
    item_w = torch.nn.Parameter(torch.zeros(K, device=device))
    params = [bias, user_w, item_w]
    if R > 0:
        user_proj = torch.nn.Parameter(torch.randn(K, R, device=device) * 0.001)
        item_proj = torch.nn.Parameter(torch.zeros(K, R, device=device))
        params.extend([user_proj, item_proj])
    else:
        user_proj = None
        item_proj = None
    opt = torch.optim.AdamW(params, lr=0.0015, weight_decay=1e-6)

    def residual_for(ids):
        pu = Pt[u_t[ids]]
        qi = Qt[i_t[ids]]
        residual = bias + pu.matmul(user_w) + qi.matmul(item_w)
        if R > 0:
            residual = residual + (pu.matmul(user_proj) * qi.matmul(item_proj)).sum(dim=1)
        return residual

    def rmse_idx(idx, batch_size=262_144):
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, idx.numel(), batch_size):
                ids = idx[start : start + batch_size]
                pred = torch.clamp(base_t[ids] + residual_for(ids), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    print(f"prep {time.time() - t0:.2f}s initial_val {rmse_idx(val_idx):.9f}", flush=True)
    best_val = 999.0
    best_state = None
    batch_size = 65_536
    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(train_idx.numel(), device=device)]
        loss_sum = 0.0
        batches = 0
        t_epoch = time.time()
        for start in range(0, order.numel(), batch_size):
            ids = order[start : start + batch_size]
            pred = base_t[ids] + residual_for(ids)
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_sum += loss.item()
            batches += 1
        val = rmse_idx(val_idx)
        full = rmse_idx(torch.arange(y_t.numel(), device=device, dtype=torch.long)) if epoch == epochs else float("nan")
        print(
            f"epoch {epoch} train_rmse {math.sqrt(loss_sum / batches):.9f} "
            f"val {val:.9f} full {full:.9f} sec {time.time() - t_epoch:.2f}",
            flush=True,
        )
        if val < best_val:
            best_val = val
            best_state = (
                float(bias.detach().cpu()),
                user_w.detach().cpu().numpy().copy(),
                item_w.detach().cpu().numpy().copy(),
                user_proj.detach().cpu().numpy().copy() if user_proj is not None else None,
                item_proj.detach().cpu().numpy().copy() if item_proj is not None else None,
            )

    bias_np, user_w_np, item_w_np, user_proj_np, item_proj_np = best_state
    out_path = OUT_DIR / f"generalized_head_k{K}_r{R}.npz"
    np.savez(
        out_path,
        K=np.array(K, dtype=np.int32),
        R=np.array(R, dtype=np.int32),
        bias=np.array(bias_np, dtype=np.float32),
        user_w=user_w_np.astype(np.float32),
        item_w=item_w_np.astype(np.float32),
        user_proj=user_proj_np.astype(np.float32) if user_proj_np is not None else np.zeros((K, 0), dtype=np.float32),
        item_proj=item_proj_np.astype(np.float32) if item_proj_np is not None else np.zeros((K, 0), dtype=np.float32),
        p_mu=p_mu,
        p_std=p_std,
        q_mu=q_mu,
        q_std=q_std,
        base_coef=BASE_COEF,
        best_val=np.array(best_val, dtype=np.float32),
    )
    print(f"saved {out_path} best_val {best_val:.9f}", flush=True)

    del Pt, Qt, u_t, i_t, y_t, base_t, train_idx, val_idx
    del bias, user_w, item_w, user_proj, item_proj, opt
    torch.cuda.empty_cache()
    gc.collect()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    u_all, i_all, y_np, base_np = build_base_prediction(P_m.shape, Q_m.shape)

    rng = np.random.default_rng(20260605)
    perm = rng.permutation(test.shape[0])
    val_idx_np = perm[:200_000].astype(np.int64)
    train_idx_np = perm[200_000:].astype(np.int64)

    for K, R, epochs in ((512, 0, 12), (1024, 0, 12), (512, 4, 16), (1024, 4, 16), (1024, 8, 16)):
        train_config(P_m, Q_m, u_all, i_all, y_np, base_np, train_idx_np, val_idx_np, K, R, epochs)


if __name__ == "__main__":
    main()
