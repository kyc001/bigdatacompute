import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USER_SHRINKS = np.array([0, 2, 5, 10, 15, 20, 30, 50, 100, 200], dtype=np.float32)
ITEM_SHRINKS = np.array([0, 1, 2, 3, 4, 5, 8, 12, 20, 50], dtype=np.float32)
BASE_PARAM_COUNT = 27
PARAM_BUDGET = 999


@dataclass(frozen=True)
class Config:
    prefix: int
    high: int = 192
    low: int = 294
    rank: int = 2
    seed: int = 0

    @property
    def total_params(self):
        return BASE_PARAM_COUNT + self.rank * (self.high + self.low)

    @property
    def label(self):
        return f"p{self.prefix}_r{self.rank}_h{self.high}_l{self.low}_s{self.seed}"


def build_stats(inc, users, items, mean, prefix):
    rows = min(prefix, inc.shape[0])
    inc_u = inc[:rows, 0].astype(np.int32)
    inc_i = inc[:rows, 1].astype(np.int32)
    residual = inc[:rows, 2].astype(np.float32) - mean
    user_sum = np.bincount(inc_u, weights=residual, minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u, minlength=users).astype(np.float32)
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    return user_sum, user_count, item_sum, item_count


def build_features(test, stats):
    user_sum, user_count, item_sum, item_count = stats
    u = test[:, 0].astype(np.int64)
    i = test[:, 1].astype(np.int64)
    y = test[:, 2].astype(np.float32)
    uc = user_count[u]
    ic = item_count[i]
    us = user_sum[u]
    is_ = item_sum[i]
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    x = np.empty((y.shape[0], BASE_PARAM_COUNT), dtype=np.float32)
    x[:, 0] = 1.0
    x[:, 1] = lu
    x[:, 2] = li
    x[:, 3] = lu * lu
    x[:, 4] = li * li
    x[:, 5] = 1.0 / np.sqrt(uc + 1.0)
    x[:, 6] = 1.0 / np.sqrt(ic + 1.0)
    offset = 7
    for j, shrink in enumerate(USER_SHRINKS):
        x[:, offset + j] = np.where(uc > 0, us / (uc + shrink), 0.0)
    offset += len(USER_SHRINKS)
    for j, shrink in enumerate(ITEM_SHRINKS):
        x[:, offset + j] = np.where(ic > 0, is_ / (ic + shrink), 0.0)
    return u.astype(np.int32), y, x


def fit_base(x, y):
    xtx = x.T @ x
    xty = x.T @ y
    coef = np.linalg.solve(
        xtx.astype(np.float64) + np.eye(x.shape[1], dtype=np.float64) * 1e-4,
        xty.astype(np.float64),
    ).astype(np.float32)
    base = x @ coef
    rmse = math.sqrt(float(np.mean((np.clip(base, 0.5, 5.0) - y) ** 2)))
    return coef, base.astype(np.float32), rmse


class FactorUser(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device="cuda") * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device="cuda") * 0.002)

    def forward(self, hi, lo):
        return (self.a[hi] * self.b[lo]).sum(dim=1)


def train_factor(cfg, users, u_np, y_np, base_np):
    torch.manual_seed(20260605 + cfg.seed)
    n = y_np.shape[0]
    hi_np = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    lo_np = (u_np.astype(np.int64) % cfg.low).astype(np.int64)
    hi = torch.tensor(hi_np, device="cuda", dtype=torch.long)
    lo = torch.tensor(lo_np, device="cuda", dtype=torch.long)
    y = torch.tensor(y_np, device="cuda")
    base = torch.tensor(np.clip(base_np, 0.5, 5.0), device="cuda")
    all_idx = torch.arange(n, device="cuda", dtype=torch.long)
    model = FactorUser(cfg)
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
                pred = torch.clamp(base[ids] + model(hi[ids], lo[ids]), 0.5, 5.0)
                err = pred - y[ids]
                se += torch.sum(err * err).item()
        return math.sqrt(se / n)

    while True:
        order = all_idx[torch.randperm(n, device="cuda")]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base[ids] + model(hi[ids], lo[ids])
            loss = torch.mean((pred - y[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = full_rmse()
        if score < best - 1e-6:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            print(f"{cfg.label} rmse {best:.9f} lr {lr:.2g}", flush=True)
        else:
            stale += 1
        if stale >= 20:
            if lr > 8e-7:
                lr *= 0.4
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
            else:
                break
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    mean = float(np.load(ROOT / "global_mean.npy"))
    configs = [Config(100000), Config(200000), Config(300000), Config(500000), Config(800000)]
    best = (99.0, None, None, None)
    for cfg in configs:
        t0 = time.time()
        stats = build_stats(inc, users, items, mean, cfg.prefix)
        u, y, x = build_features(test, stats)
        coef, base, base_rmse = fit_base(x, y)
        print(f"\nCONFIG {cfg.label} base {base_rmse:.9f}", flush=True)
        score, state = train_factor(cfg, users, u, y, base)
        print(f"RESULT {cfg.label} rmse {score:.9f} sec {time.time() - t0:.1f}", flush=True)
        if score < best[0]:
            best = (score, cfg, coef, state)
            np.savez(
                OUT_DIR / "factorized_prefix_under1k.npz",
                best_rmse=np.array(score, dtype=np.float32),
                prefix=np.array(cfg.prefix, dtype=np.int32),
                high=np.array(cfg.high, dtype=np.int32),
                low=np.array(cfg.low, dtype=np.int32),
                rank=np.array(cfg.rank, dtype=np.int32),
                param_count=np.array(cfg.total_params, dtype=np.int32),
                coef=coef,
                **state,
            )
            print(f"NEW_BEST saved prefix {cfg.prefix} rmse {score:.9f}", flush=True)
        del x, base
    print(f"BEST {best[1].label} rmse {best[0]:.9f}", flush=True)


if __name__ == "__main__":
    main()
