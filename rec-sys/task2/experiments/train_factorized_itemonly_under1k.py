import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402
import train_small_param_tables as tables  # noqa: E402


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
    def __init__(self, cfg, device):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device=device) * 0.002)
        self.base = torch.nn.Parameter(torch.zeros(cfg.base_bins, device=device))

    def forward(self, hi, lo, bb):
        return (self.a[hi] * self.b[lo]).sum(dim=1) + self.base[bb]


def build_item_base(users, items):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    i = test[:, 1].astype(np.int32)
    ic = item_count[i]
    is_ = item_sum[i]
    li = np.log1p(ic).astype(np.float32)
    score = np.full(test.shape[0], common.BASE_COEF[0] + common.BASE_COEF[5], dtype=np.float32)
    score += common.BASE_COEF[2] * li + common.BASE_COEF[4] * li * li
    score += common.BASE_COEF[6] / np.sqrt(ic + 1.0).astype(np.float32)
    for idx, shrink in enumerate(common.ITEM_SHRINKS):
        avg = np.where(ic > 0, is_ / (ic + shrink), 0.0).astype(np.float32)
        score += common.BASE_COEF[17 + idx] * avg
    return test[:, 0].astype(np.int32), i, test[:, 2].astype(np.float32), score


def train_one(cfg, users, u_np, y_np, base_np):
    if cfg.params > 999:
        print(f"skip {cfg.label} params {cfg.params}", flush=True)
        return None
    torch.manual_seed(20260605 + cfg.seed)
    device = torch.device("cuda")
    n = y_np.shape[0]
    hi = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    lo = (u_np.astype(np.int64) % cfg.low).astype(np.int64)
    bb = tables.uniform_bins(np.clip(base_np, 0.5, 5.0), cfg.base_bins, 0.5, 5.0).astype(np.int64)
    hi_t = torch.tensor(hi, device=device, dtype=torch.long)
    lo_t = torch.tensor(lo, device=device, dtype=torch.long)
    bb_t = torch.tensor(bb, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(np.clip(base_np, 0.5, 5.0).astype(np.float32), device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)
    model = Model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    min_lr = 8e-7
    patience = 24
    min_delta = 1e-6
    stale = 0

    def full_rmse():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = base_t[ids] + model(hi_t[ids], lo_t[ids], bb_t[ids])
                err = torch.clamp(pred, 0.5, 5.0) - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = full_rmse()
    best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    print(f"\nCONFIG {cfg.label} params {cfg.params} initial {best:.9f}", flush=True)
    start_time = time.time()
    epoch = 0
    while True:
        epoch += 1
        order = all_idx[torch.randperm(n, device=device)]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base_t[ids] + model(hi_t[ids], lo_t[ids], bb_t[ids])
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = full_rmse()
        if score < best - min_delta:
            best = score
            best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch <= 5 or epoch % 20 == 0 or stale == 0:
            print(f"epoch {epoch} rmse {score:.9f} best {best:.9f} stale {stale} lr {lr:.2g}", flush=True)
        if stale >= patience:
            if lr > min_lr:
                lr *= 0.4
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
                print(f"reduce_lr {lr:.4g}", flush=True)
            else:
                break
    print(f"RESULT {cfg.label} rmse {best:.9f} sec {time.time()-start_time:.1f}", flush=True)
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    u, _, y, base = build_item_base(users, items)
    print(f"item_base rmse {math.sqrt(float(np.mean((np.clip(base,0.5,5.0)-y)**2))):.9f}", flush=True)
    configs = [
        Config(192, 296, 2, 16, 0),
        Config(192, 296, 2, 16, 1),
        Config(200, 288, 2, 16, 0),
        Config(184, 304, 2, 16, 0),
        Config(192, 304, 2, 1, 0),
        Config(200, 296, 2, 1, 0),
    ]
    best = (99.0, None, None)
    for cfg in configs:
        result = train_one(cfg, users, u, y, base)
        if result is None:
            continue
        score, state = result
        if score < best[0]:
            best = (score, cfg, state)
            print(f"NEW_BEST {cfg.label} {score:.9f}", flush=True)
    score, cfg, state = best
    out_path = OUT_DIR / "factorized_itemonly_under1k.npz"
    np.savez(
        out_path,
        best_rmse=np.array(score, dtype=np.float32),
        label=np.array(cfg.label),
        high=np.array(cfg.high, dtype=np.int32),
        low=np.array(cfg.low, dtype=np.int32),
        rank=np.array(cfg.rank, dtype=np.int32),
        base_bins=np.array(cfg.base_bins, dtype=np.int32),
        param_count=np.array(cfg.params, dtype=np.int32),
        **state,
    )
    print(f"BEST itemonly {cfg.label} rmse {score:.9f} params {cfg.params} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
