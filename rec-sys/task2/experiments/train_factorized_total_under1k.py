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


BASE_PARAM_COUNT = 27
PARAM_BUDGET = 999


@dataclass(frozen=True)
class Config:
    high: int
    low: int
    rank: int = 2
    seed: int = 0

    @property
    def head_params(self):
        return self.rank * (self.high + self.low)

    @property
    def total_params(self):
        return BASE_PARAM_COUNT + self.head_params

    @property
    def label(self):
        return f"r{self.rank}_h{self.high}_l{self.low}_s{self.seed}"


class FactorUser(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device=device) * 0.002)

    def forward(self, hi, lo):
        return (self.a[hi] * self.b[lo]).sum(dim=1)


def train_one(cfg, users, u_np, y_np, base_np):
    if cfg.total_params > PARAM_BUDGET:
        print(f"skip {cfg.label} total_params {cfg.total_params}", flush=True)
        return None
    torch.manual_seed(20260605 + cfg.seed)
    device = torch.device("cuda")
    n = y_np.shape[0]
    hi = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    lo = (u_np.astype(np.int64) % cfg.low).astype(np.int64)

    hi_t = torch.tensor(hi, device=device, dtype=torch.long)
    lo_t = torch.tensor(lo, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(np.clip(base_np, 0.5, 5.0).astype(np.float32), device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)
    model = FactorUser(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    min_lr = 8e-7
    patience = 28
    min_delta = 1e-6
    stale = 0

    def full_rmse():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = base_t[ids] + model(hi_t[ids], lo_t[ids])
                err = torch.clamp(pred, 0.5, 5.0) - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = full_rmse()
    best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    epoch = 0
    t0 = time.time()
    print(f"\nCONFIG {cfg.label} total_params {cfg.total_params} initial {best:.9f}", flush=True)
    while True:
        epoch += 1
        order = all_idx[torch.randperm(n, device=device)]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base_t[ids] + model(hi_t[ids], lo_t[ids])
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
        if epoch <= 5 or epoch % 10 == 0 or stale == 0:
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
    print(f"RESULT {cfg.label} rmse {best:.9f} sec {time.time()-t0:.1f}", flush=True)
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, _, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    users = P_m.shape[0]
    u_np = u_all.astype(np.int32)

    configs = [
        Config(192, 294, seed=0),
        Config(190, 296, seed=0),
        Config(184, 302, seed=0),
        Config(200, 286, seed=0),
        Config(192, 294, seed=1),
        Config(190, 296, seed=1),
    ]
    best = (99.0, None, None)
    for cfg in configs:
        result = train_one(cfg, users, u_np, y, base)
        if result is None:
            continue
        score, state = result
        if score < best[0]:
            best = (score, cfg, state)
            print(f"NEW_BEST {cfg.label} {score:.9f}", flush=True)

    score, cfg, state = best
    out_path = OUT_DIR / "factorized_total_under1k.npz"
    np.savez(
        out_path,
        best_rmse=np.array(score, dtype=np.float32),
        label=np.array(cfg.label),
        high=np.array(cfg.high, dtype=np.int32),
        low=np.array(cfg.low, dtype=np.int32),
        rank=np.array(cfg.rank, dtype=np.int32),
        base_param_count=np.array(BASE_PARAM_COUNT, dtype=np.int32),
        head_param_count=np.array(cfg.head_params, dtype=np.int32),
        param_count=np.array(cfg.total_params, dtype=np.int32),
        **state,
    )
    print(f"BEST total_under1k {cfg.label} rmse {score:.9f} total_params {cfg.total_params} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
