import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import train_generalized_head as common
import train_small_param_tables as tables


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    high: int
    low: int
    rank: int
    base_bins: int
    seed: int = 0

    @property
    def params(self):
        return self.rank * (self.high + self.low) + self.base_bins

    @property
    def label(self):
        return f"r{self.rank}_h{self.high}_l{self.low}_b{self.base_bins}_s{self.seed}"


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device="cuda") * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device="cuda") * 0.002)
        self.base = torch.nn.Parameter(torch.zeros(cfg.base_bins, device="cuda"))

    def forward(self, hi, lo, bb):
        return (self.a[hi] * self.b[lo]).sum(dim=1) + self.base[bb]


def build_user_base(users, items):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[::2, 0].astype(np.int32)
    residual = inc[::2, 2].astype(np.float32) - mean
    user_sum = np.bincount(inc_u, weights=residual, minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u, minlength=users).astype(np.float32)
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    uc = user_count[u]
    us = user_sum[u]
    lu = np.log1p(uc).astype(np.float32)
    score = np.full(test.shape[0], common.BASE_COEF[0] + common.BASE_COEF[6], dtype=np.float32)
    score += common.BASE_COEF[1] * lu + common.BASE_COEF[3] * lu * lu
    score += common.BASE_COEF[5] / np.sqrt(uc + 1.0).astype(np.float32)
    for idx, shrink in enumerate(common.USER_SHRINKS):
        avg = np.where(uc > 0, us / (uc + shrink), 0.0).astype(np.float32)
        score += common.BASE_COEF[7 + idx] * avg
    return u, i, test[:, 2].astype(np.float32), score


def train_one(cfg, items, i_np, y_np, base_np):
    if cfg.params > 999:
        return None
    torch.manual_seed(20260605 + cfg.seed)
    n = y_np.shape[0]
    hi = (i_np.astype(np.int64) * cfg.high // items).astype(np.int64)
    lo = (i_np.astype(np.int64) % cfg.low).astype(np.int64)
    bb = tables.uniform_bins(np.clip(base_np, 0.5, 5.0), cfg.base_bins, 0.5, 5.0).astype(np.int64)
    hi_t = torch.tensor(hi, device="cuda", dtype=torch.long)
    lo_t = torch.tensor(lo, device="cuda", dtype=torch.long)
    bb_t = torch.tensor(bb, device="cuda", dtype=torch.long)
    y_t = torch.tensor(y_np, device="cuda")
    base_t = torch.tensor(np.clip(base_np, 0.5, 5.0).astype(np.float32), device="cuda")
    all_idx = torch.arange(n, device="cuda", dtype=torch.long)
    model = Model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    stale = 0
    best = 99.0
    best_state = None

    def full_rmse():
        se = 0.0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = torch.clamp(base_t[ids] + model(hi_t[ids], lo_t[ids], bb_t[ids]), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
        return math.sqrt(se / n)

    print(f"\nCONFIG {cfg.label} params {cfg.params} initial {full_rmse():.9f}", flush=True)
    t0 = time.time()
    while True:
        order = all_idx[torch.randperm(n, device="cuda")]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base_t[ids] + model(hi_t[ids], lo_t[ids], bb_t[ids])
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = full_rmse()
        if score < best - 1e-6:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            print(f"rmse {best:.9f} lr {lr:.2g}", flush=True)
        else:
            stale += 1
        if stale >= 24:
            if lr > 8e-7:
                lr *= 0.4
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
            else:
                break
    print(f"RESULT {cfg.label} rmse {best:.9f} sec {time.time() - t0:.1f}", flush=True)
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    _, i, y, base = build_user_base(users, items)
    print(f"user_base rmse {math.sqrt(float(np.mean((np.clip(base,0.5,5.0)-y)**2))):.9f}", flush=True)
    configs = [
        Config(192, 296, 2, 16, 0),
        Config(200, 288, 2, 16, 0),
        Config(184, 304, 2, 16, 0),
        Config(128, 224, 2, 292, 0),
        Config(160, 240, 2, 190, 0),
    ]
    best = (99.0, None, None)
    for cfg in configs:
        result = train_one(cfg, items, i, y, base)
        if result is None:
            continue
        score, state = result
        if score < best[0]:
            best = (score, cfg, state)
            np.savez(
                OUT_DIR / "factorized_useronly_under1k.npz",
                best_rmse=np.array(score, dtype=np.float32),
                high=np.array(cfg.high, dtype=np.int32),
                low=np.array(cfg.low, dtype=np.int32),
                rank=np.array(cfg.rank, dtype=np.int32),
                base_bins=np.array(cfg.base_bins, dtype=np.int32),
                param_count=np.array(cfg.params, dtype=np.int32),
                **state,
            )
            print(f"NEW_BEST {cfg.label} {score:.9f}", flush=True)
    print(f"BEST {best[1].label} {best[0]:.9f}", flush=True)


if __name__ == "__main__":
    main()
