import gc
import math
import time
from pathlib import Path

import numpy as np
import torch

import train_generalized_head as common


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    K = 1024
    root = ROOT
    P_m = np.load(root / "P.npy", mmap_mode="r")
    Q_m = np.load(root / "Q.npy", mmap_mode="r")
    test = np.load(root / "test.npy", mmap_mode="r")
    u_all, i_all, y_np, base_np = common.build_base_prediction(P_m.shape, Q_m.shape)
    n = y_np.shape[0]

    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"linear all-data K={K} n={n}", flush=True)
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
    print(f"standardized {time.time() - t0:.2f}s", flush=True)

    device = torch.device("cuda")
    Pt = torch.tensor(P, device=device)
    Qt = torch.tensor(Q, device=device)
    u_t = torch.tensor(u_all, device=device, dtype=torch.long)
    i_t = torch.tensor(i_all, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(base_np, device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    bias = torch.nn.Parameter(torch.zeros((), device=device))
    user_w = torch.nn.Parameter(torch.zeros(K, device=device))
    item_w = torch.nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.AdamW([bias, user_w, item_w], lr=0.001, weight_decay=1e-7)
    batch_size = 65_536

    def residual_for(ids):
        return bias + Pt[u_t[ids]].matmul(user_w) + Qt[i_t[ids]].matmul(item_w)

    def rmse_all():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 262_144):
                ids = all_idx[start : start + 262_144]
                pred = torch.clamp(base_t[ids] + residual_for(ids), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best_rmse = rmse_all()
    best_state = None
    print(f"initial full {best_rmse:.9f}", flush=True)

    for epoch in range(1, 101):
        if epoch == 41:
            for group in opt.param_groups:
                group["lr"] = 0.00035
        if epoch == 71:
            for group in opt.param_groups:
                group["lr"] = 0.00012
        order = all_idx[torch.randperm(n, device=device)]
        loss_sum = 0.0
        batches = 0
        t_epoch = time.time()
        for start in range(0, n, batch_size):
            ids = order[start : start + batch_size]
            pred = base_t[ids] + residual_for(ids)
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([bias, user_w, item_w], 5.0)
            opt.step()
            loss_sum += loss.item()
            batches += 1
        if epoch <= 5 or epoch % 5 == 0:
            full = rmse_all()
            print(
                f"epoch {epoch} train_rmse {math.sqrt(loss_sum / batches):.9f} "
                f"full {full:.9f} sec {time.time() - t_epoch:.2f}",
                flush=True,
            )
            if full < best_rmse:
                best_rmse = full
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

    out_path = OUT_DIR / "generalized_linear_all_k1024.npz"
    np.savez(
        out_path,
        K=np.array(K, dtype=np.int32),
        R=np.array(0, dtype=np.int32),
        bias=np.array(best_state[0], dtype=np.float32),
        user_w=best_state[1].astype(np.float32),
        item_w=best_state[2].astype(np.float32),
        user_proj=np.zeros((K, 0), dtype=np.float32),
        item_proj=np.zeros((K, 0), dtype=np.float32),
        p_mu=p_mu,
        p_std=p_std,
        q_mu=q_mu,
        q_std=q_std,
        base_coef=common.BASE_COEF,
        best_rmse=np.array(best_rmse, dtype=np.float32),
    )
    print(f"saved {out_path} best_rmse {best_rmse:.9f}", flush=True)

    del Pt, Qt, u_t, i_t, y_t, base_t, all_idx, bias, user_w, item_w, opt
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
