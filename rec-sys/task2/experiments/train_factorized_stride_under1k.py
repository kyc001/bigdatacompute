import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402


def build_base(stride):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    user_sum = np.bincount(inc_u[::stride], weights=residual[::stride], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::stride], minlength=users).astype(np.float32)
    u = test[:, 0].astype(np.int64)
    i = test[:, 1].astype(np.int64)
    y = test[:, 2].astype(np.float32)
    uc = user_count[u]
    ic = item_count[i]
    us = user_sum[u]
    is_ = item_sum[i]
    lu = np.log1p(uc)
    li = np.log1p(ic)
    score = np.full(y.shape, common.BASE_COEF[0], dtype=np.float32)
    score += common.BASE_COEF[1] * lu + common.BASE_COEF[2] * li
    score += common.BASE_COEF[3] * lu * lu + common.BASE_COEF[4] * li * li
    score += common.BASE_COEF[5] / np.sqrt(uc + 1.0)
    score += common.BASE_COEF[6] / np.sqrt(ic + 1.0)
    for idx, shrink in enumerate(common.USER_SHRINKS):
        score += common.BASE_COEF[7 + idx] * np.where(uc > 0, us / (uc + shrink), 0.0).astype(np.float32)
    for idx, shrink in enumerate(common.ITEM_SHRINKS):
        score += common.BASE_COEF[17 + idx] * np.where(ic > 0, is_ / (ic + shrink), 0.0).astype(np.float32)
    return users, u.astype(np.int32), y, score.astype(np.float32)


class Model(torch.nn.Module):
    def __init__(self, high, low, rank):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(high, rank, device="cuda") * 0.01)
        self.b = torch.nn.Parameter(torch.randn(low, rank, device="cuda") * 0.002)

    def forward(self, hi, lo):
        return (self.a[hi] * self.b[lo]).sum(dim=1)


def train(stride, high, low, rank, seed):
    torch.manual_seed(20260605 + seed)
    users, u_np, y_np, base_np = build_base(stride)
    print(f"stride {stride} base {math.sqrt(float(np.mean((np.clip(base_np,0.5,5.0)-y_np)**2))):.9f}", flush=True)
    hi_np = (u_np.astype(np.int64) * high // users).astype(np.int64)
    lo_np = (u_np.astype(np.int64) % low).astype(np.int64)
    y = torch.tensor(y_np, device="cuda")
    base = torch.tensor(np.clip(base_np, 0.5, 5.0), device="cuda")
    hi = torch.tensor(hi_np, device="cuda", dtype=torch.long)
    lo = torch.tensor(lo_np, device="cuda", dtype=torch.long)
    idx_all = torch.arange(y.numel(), device="cuda", dtype=torch.long)
    model = Model(high, low, rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    best = 99.0
    best_state = None
    stale = 0
    min_lr = 8e-7
    while True:
        order = idx_all[torch.randperm(idx_all.numel(), device="cuda")]
        for start in range(0, order.numel(), 65536):
            ids = order[start : start + 65536]
            pred = base[ids] + model(hi[ids], lo[ids])
            loss = torch.mean((pred - y[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        se = 0.0
        with torch.no_grad():
            for start in range(0, idx_all.numel(), 262144):
                ids = idx_all[start : start + 262144]
                err = torch.clamp(base[ids] + model(hi[ids], lo[ids]), 0.5, 5.0) - y[ids]
                se += torch.sum(err * err).item()
        score = math.sqrt(se / y_np.shape[0])
        if score < best - 1e-6:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            print(f"stride {stride} rmse {best:.9f} lr {lr:.2g}", flush=True)
        else:
            stale += 1
        if stale >= 24:
            if lr > min_lr:
                lr *= 0.4
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
            else:
                break
    out = OUT_DIR / f"factorized_stride{stride}_under1k.npz"
    np.savez(
        out,
        best_rmse=np.array(best, dtype=np.float32),
        stride=np.array(stride, dtype=np.int32),
        high=np.array(high, dtype=np.int32),
        low=np.array(low, dtype=np.int32),
        rank=np.array(rank, dtype=np.int32),
        param_count=np.array(rank * (high + low) + 27, dtype=np.int32),
        **best_state,
    )
    print(f"saved {out} rmse {best:.9f}", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train(3, 192, 294, 2, 0)


if __name__ == "__main__":
    main()
