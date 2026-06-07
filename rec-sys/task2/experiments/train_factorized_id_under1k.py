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
    label: str
    high: int
    low: int
    rank: int
    item_bins: int
    base_bins: int

    @property
    def params(self):
        return self.rank * (self.high + self.low) + self.item_bins + self.base_bins


CONFIGS = [
    Config("r1_h271_l512_i192_b16", 271, 512, 1, 192, 16),
    Config("r1_h192_l724_i64_b16", 192, 724, 1, 64, 16),
    Config("r1_h542_l256_i128_b64", 542, 256, 1, 128, 64),
    Config("r1_h384_l512_i64_b32", 384, 512, 1, 64, 32),
    Config("r1_h512_l384_i64_b32", 512, 384, 1, 64, 32),
    Config("r1_h256_l640_i64_b32", 256, 640, 1, 64, 32),
    Config("r1_h181_l768_i32_b16", 181, 768, 1, 32, 16),
    Config("r2_h180_l300_i1_b32", 180, 300, 2, 1, 32),
    Config("r2_h192_l296_i1_b16", 192, 296, 2, 1, 16),
    Config("r2_h160_l320_i16_b16", 160, 320, 2, 16, 16),
    Config("r2_h144_l336_i16_b16", 144, 336, 2, 16, 16),
    Config("r2_h128_l360_i1_b16", 128, 360, 2, 1, 16),
    Config("r3_h96_l224_i16_b16", 96, 224, 3, 16, 16),
    Config("r3_h112_l208_i16_b16", 112, 208, 3, 16, 16),
    Config("r3_h128_l192_i16_b16", 128, 192, 3, 16, 16),
    Config("r4_h80_l160_i16_b16", 80, 160, 4, 16, 16),
    Config("r4_h96_l144_i16_b16", 96, 144, 4, 16, 16),
    Config("r5_h64_l128_i16_b16", 64, 128, 5, 16, 16),
    Config("r2_h128_l256_i192_b32", 128, 256, 2, 192, 32),
    Config("r2_h160_l256_i128_b32", 160, 256, 2, 128, 32),
    Config("r2_h96_l384_i32_b32", 96, 384, 2, 32, 32),
    Config("r3_h64_l256_i32_b32", 64, 256, 3, 32, 32),
    Config("r4_h64_l160_i64_b32", 64, 160, 4, 64, 32),
    Config("r4_h48_l192_i64_b32", 48, 192, 4, 64, 32),
]


def rmse_t(y, pred):
    return math.sqrt(float(torch.mean((torch.clamp(pred, 0.5, 5.0) - y) ** 2).item()))


class FactorizedId(torch.nn.Module):
    def __init__(self, cfg: Config, device):
        super().__init__()
        self.cfg = cfg
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(cfg.low, cfg.rank, device=device))
        self.item = torch.nn.Parameter(torch.zeros(cfg.item_bins, device=device))
        self.base = torch.nn.Parameter(torch.zeros(cfg.base_bins, device=device))

    def forward(self, hi, lo, ib, bb):
        return (self.a[hi] * self.b[lo]).sum(dim=1) + self.item[ib] + self.base[bb]


def train_config(cfg, users, items, u_np, i_np, y_np, base_np):
    if cfg.params > PARAM_BUDGET:
        print(f"skip {cfg.label} params {cfg.params}", flush=True)
        return None
    device = torch.device("cuda")
    n = y_np.shape[0]
    u_hi = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    u_lo = (u_np.astype(np.int64) % cfg.low).astype(np.int64)
    i_bin = (i_np.astype(np.int64) * cfg.item_bins // items).astype(np.int64)
    b_bin = tables.uniform_bins(np.clip(base_np, 0.5, 5.0), cfg.base_bins, 0.5, 5.0).astype(np.int64)

    hi_t = torch.tensor(u_hi, device=device, dtype=torch.long)
    lo_t = torch.tensor(u_lo, device=device, dtype=torch.long)
    ib_t = torch.tensor(i_bin, device=device, dtype=torch.long)
    bb_t = torch.tensor(b_bin, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(np.clip(base_np, 0.5, 5.0).astype(np.float32), device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    model = FactorizedId(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1e-6)
    lr = 0.025
    min_lr = 2e-5
    patience = 20
    min_delta = 5e-6
    stale = 0

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
    print(f"\nCONFIG {cfg.label} params {cfg.params} initial {best:.9f}", flush=True)
    epoch = 0
    while True:
        epoch += 1
        order = all_idx[torch.randperm(n, device=device)]
        t0 = time.time()
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
        if epoch <= 5 or epoch % 5 == 0 or stale == 0:
            print(f"epoch {epoch} rmse {score:.9f} best {best:.9f} stale {stale} lr {lr:.2g} sec {time.time()-t0:.2f}", flush=True)
        if stale >= patience:
            if lr > min_lr:
                lr *= 0.35
                for group in opt.param_groups:
                    group["lr"] = lr
                stale = 0
                print(f"reduce_lr {lr:.4g}", flush=True)
            else:
                break
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    users, items = P_m.shape[0], Q_m.shape[0]

    best = (99.0, None, None)
    for cfg in CONFIGS:
        result = train_config(cfg, users, items, u_all.astype(np.int32), i_all.astype(np.int32), y, base)
        if result is None:
            continue
        score, state = result
        print(f"RESULT {cfg.label} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, cfg, state)

    score, cfg, state = best
    out_path = OUT_DIR / "factorized_id_under1k.npz"
    np.savez(
        out_path,
        best_rmse=np.array(score, dtype=np.float32),
        label=np.array(cfg.label),
        high=np.array(cfg.high, dtype=np.int32),
        low=np.array(cfg.low, dtype=np.int32),
        rank=np.array(cfg.rank, dtype=np.int32),
        item_bins=np.array(cfg.item_bins, dtype=np.int32),
        base_bins=np.array(cfg.base_bins, dtype=np.int32),
        param_count=np.array(cfg.params, dtype=np.int32),
        **state,
    )
    print(f"BEST factorized {cfg.label} rmse {score:.9f} params {cfg.params} saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
