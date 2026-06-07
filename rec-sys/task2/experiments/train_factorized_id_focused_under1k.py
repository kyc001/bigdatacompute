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


PARAM_BUDGET = 999


@dataclass(frozen=True)
class Config:
    high: int
    low: int
    rank: int
    item_bins: int
    base_bins: int
    seed: int

    @property
    def params(self):
        return self.rank * (self.high + self.low) + self.item_bins + self.base_bins

    @property
    def label(self):
        return f"r{self.rank}_h{self.high}_l{self.low}_i{self.item_bins}_b{self.base_bins}_s{self.seed}"


class Model(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device=device) * 0.002)
        self.item = torch.nn.Parameter(torch.zeros(cfg.item_bins, device=device))
        self.base = torch.nn.Parameter(torch.zeros(cfg.base_bins, device=device))

    def forward(self, hi, lo, ib, bb):
        return (self.a[hi] * self.b[lo]).sum(dim=1) + self.item[ib] + self.base[bb]


def train_one(cfg, users, items, u_np, i_np, y_np, base_np):
    if cfg.params > PARAM_BUDGET:
        return None
    torch.manual_seed(20260605 + cfg.seed)
    np.random.seed(20260605 + cfg.seed)
    device = torch.device("cuda")
    n = y_np.shape[0]
    hi = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    lo = (u_np.astype(np.int64) % cfg.low).astype(np.int64)
    ib = (i_np.astype(np.int64) * cfg.item_bins // items).astype(np.int64)
    bb = tables.uniform_bins(np.clip(base_np, 0.5, 5.0), cfg.base_bins, 0.5, 5.0).astype(np.int64)

    hi_t = torch.tensor(hi, device=device, dtype=torch.long)
    lo_t = torch.tensor(lo, device=device, dtype=torch.long)
    ib_t = torch.tensor(ib, device=device, dtype=torch.long)
    bb_t = torch.tensor(bb, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(np.clip(base_np, 0.5, 5.0).astype(np.float32), device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    model = Model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    min_lr = 8e-7
    patience = 28
    stale = 0
    min_delta = 1e-6

    def full_rmse():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = base_t[ids] + model(hi_t[ids], lo_t[ids], ib_t[ids], bb_t[ids])
                err = torch.clamp(pred, 0.5, 5.0) - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = full_rmse()
    best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    epoch = 0
    t_start = time.time()
    while True:
        epoch += 1
        order = all_idx[torch.randperm(n, device=device)]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base_t[ids] + model(hi_t[ids], lo_t[ids], ib_t[ids], bb_t[ids])
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = full_rmse()
        if score < best - min_delta:
            best = score
            stale = 0
            best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch <= 5 or epoch % 25 == 0 or stale == 0:
            print(f"{cfg.label} epoch {epoch} rmse {score:.9f} best {best:.9f} stale {stale} lr {lr:.2g}", flush=True)
        if stale >= patience:
            if lr > min_lr:
                lr *= 0.4
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
                print(f"{cfg.label} reduce_lr {lr:.4g}", flush=True)
            else:
                break
    print(f"RESULT {cfg.label} params {cfg.params} rmse {best:.9f} sec {time.time()-t_start:.1f}", flush=True)
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    users, items = P_m.shape[0], Q_m.shape[0]
    u_np = u_all.astype(np.int32)
    i_np = i_all.astype(np.int32)

    configs = []
    for high, low, base_bins in (
        (176, 312, 16),
        (184, 304, 16),
        (192, 296, 16),
        (200, 288, 16),
        (208, 280, 16),
        (216, 272, 16),
        (224, 264, 16),
        (192, 306, 1),
        (200, 298, 1),
        (208, 290, 1),
        (216, 282, 1),
        (184, 306, 8),
        (192, 298, 8),
        (200, 290, 8),
        (208, 282, 8),
    ):
        configs.append(Config(high, low, 2, 1, base_bins, 0))
    for high, low, base_bins in (
        (192, 296, 16),
        (200, 298, 1),
        (192, 298, 8),
    ):
        configs.append(Config(high, low, 2, 1, base_bins, 1))
        configs.append(Config(high, low, 2, 1, base_bins, 2))
    for high, low in ((96, 232), (104, 224), (112, 216), (120, 208), (128, 200)):
        configs.append(Config(high, low, 3, 1, 8, 0))

    best = (99.0, None, None)
    for cfg in configs:
        result = train_one(cfg, users, items, u_np, i_np, y, base)
        if result is None:
            continue
        score, state = result
        if score < best[0]:
            best = (score, cfg, state)
            print(f"NEW_BEST {cfg.label} {score:.9f}", flush=True)

    score, cfg, state = best
    out_path = OUT_DIR / "factorized_id_focused_under1k.npz"
    np.savez(
        out_path,
        best_rmse=np.array(score, dtype=np.float32),
        label=np.array(cfg.label),
        high=np.array(cfg.high, dtype=np.int32),
        low=np.array(cfg.low, dtype=np.int32),
        rank=np.array(cfg.rank, dtype=np.int32),
        item_bins=np.array(cfg.item_bins, dtype=np.int32),
        base_bins=np.array(cfg.base_bins, dtype=np.int32),
        seed=np.array(cfg.seed, dtype=np.int32),
        param_count=np.array(cfg.params, dtype=np.int32),
        **state,
    )
    print(f"BEST focused {cfg.label} rmse {score:.9f} params {cfg.params} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
