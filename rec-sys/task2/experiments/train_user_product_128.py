import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")


def rmse_np(y, pred):
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
    ua = safe_avg(us[u], U, 20.0)
    ia = safe_avg(is_[i], I, 5.0)
    lu = np.log1p(U).astype(np.float32)
    li = np.log1p(I).astype(np.float32)
    x = np.stack(
        [np.ones_like(y), lu, li, lu * lu, li * li, 1 / np.sqrt(U + 1), 1 / np.sqrt(I + 1), ua, ia],
        axis=1,
    ).astype(np.float32)
    coef = np.linalg.solve(x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(9) * 1e-4, x.T.astype(np.float64) @ y.astype(np.float64)).astype(np.float32)
    pred = x @ coef
    print(f"base9 {rmse_np(y, pred):.9f}", flush=True)
    return users, u, y, pred


def user_target(users, u, y, base):
    residual = (y - np.clip(base, 0.5, 5.0)).astype(np.float32)
    sums = np.bincount(u, weights=residual, minlength=users).astype(np.float32)
    counts = np.bincount(u, minlength=users).astype(np.float32)
    target = np.zeros(users, dtype=np.float32)
    mask = counts > 0
    target[mask] = sums[mask] / counts[mask]
    return target, counts


def selected_fourier(users, target, counts, n_features):
    series = target.astype(np.float64) * np.sqrt(counts / max(float(counts.mean()), 1e-12))
    series -= series.mean()
    amp = np.abs(np.fft.rfft(series))
    amp[0] = 0
    freqs = np.argsort(amp)[::-1]
    uid = np.arange(users, dtype=np.float64)
    feats = []
    names = []
    for f in freqs:
        if len(feats) >= n_features:
            break
        ang = 2 * np.pi * f * uid / users
        feats.append(np.sin(ang).astype(np.float32))
        names.append((int(f), 0))
        if len(feats) >= n_features:
            break
        feats.append(np.cos(ang).astype(np.float32))
        names.append((int(f), 1))
    return np.stack(feats, axis=1).astype(np.float32), names


class ProductModel(torch.nn.Module):
    def __init__(self, a_dim, b_dim, rank):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(rank, a_dim, device="cuda") * 0.02)
        self.b = torch.nn.Parameter(torch.randn(rank, b_dim, device="cuda") * 0.02)
        self.scale = torch.nn.Parameter(torch.ones(rank, device="cuda") * 0.1)

    def forward(self, A, B):
        xa = A.matmul(self.a.t())
        xb = B.matmul(self.b.t())
        return (xa * xb * self.scale).sum(dim=1)


def train_config(label, A_np, B_np, target_np, counts_np, u_np, y_np, base_np, rank):
    device = "cuda"
    mask = counts_np > 0
    A = torch.tensor(A_np[mask], device=device)
    B = torch.tensor(B_np[mask], device=device)
    target = torch.tensor(target_np[mask], device=device)
    weight = torch.tensor(counts_np[mask] / max(float(counts_np[mask].mean()), 1e-12), device=device)
    model = ProductModel(A.shape[1], B.shape[1], rank)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-6)
    best = 99.0
    best_state = None
    stale = 0
    lr = 0.01
    n = target.numel()
    idx_all = torch.arange(n, device=device)

    def full_user_loss():
        with torch.no_grad():
            pred = model(A, B)
            loss = torch.mean(weight * (pred - target) ** 2).item()
        return loss

    for epoch in range(1, 2000):
        order = idx_all[torch.randperm(n, device=device)]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = model(A[ids], B[ids])
            loss = torch.mean(weight[ids] * (pred - target[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        loss = full_user_loss()
        if loss < best - 1e-9:
            best = loss
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 25 == 0:
            corr_user = eval_model(A_np, B_np, best_state, rank)
            score = rmse_np(y_np, base_np + corr_user[u_np])
            print(f"{label} epoch {epoch} user_loss {best:.8f} rmse {score:.9f} lr {lr:.2g}", flush=True)
        if stale >= 80:
            if lr > 1e-5:
                lr *= 0.4
                for g in opt.param_groups:
                    g["lr"] = lr
                stale = 0
            else:
                break
    corr_user = eval_model(A_np, B_np, best_state, rank)
    score = rmse_np(y_np, base_np + corr_user[u_np])
    params = rank * (A_np.shape[1] + B_np.shape[1] + 1)
    print(f"RESULT {label} params {params} rmse {score:.9f}", flush=True)
    return score


def eval_model(A_np, B_np, state, rank):
    a = state["a"].astype(np.float32)
    b = state["b"].astype(np.float32)
    scale = state["scale"].astype(np.float32)
    xa = A_np @ a.T
    xb = B_np @ b.T
    return (xa * xb * scale.reshape(1, rank)).sum(axis=1).astype(np.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    users, u, y, base = build_base()
    target, counts = user_target(users, u, y, base)
    F, names = selected_fourier(users, target, counts, 116)
    ones = np.ones((users, 1), dtype=np.float32)
    configs = [
        ("prod_r1_58_58", F[:, :58], np.concatenate([ones, F[:, 58:115]], axis=1), 1),
        ("prod_r1_64_52", F[:, :64], np.concatenate([ones, F[:, 64:115]], axis=1), 1),
        ("prod_r2_29_29", F[:, :29], np.concatenate([ones, F[:, 29:57]], axis=1), 2),
        ("prod_r2_24_34", F[:, :24], np.concatenate([ones, F[:, 24:57]], axis=1), 2),
    ]
    for label, A, B, rank in configs:
        params = rank * (A.shape[1] + B.shape[1] + 1)
        print(f"CONFIG {label} A {A.shape[1]} B {B.shape[1]} rank {rank} params {params}", flush=True)
        train_config(label, A, B, target, counts, u, y, base, rank)


if __name__ == "__main__":
    main()
