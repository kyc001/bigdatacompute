import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAM_BUDGET = 999


@dataclass(frozen=True)
class Config:
    uh: int
    ul: int
    ur: int
    ih: int
    il: int
    ir: int
    seed: int = 0

    @property
    def params(self):
        return 1 + self.ur * (self.uh + self.ul) + self.ir * (self.ih + self.il)

    @property
    def label(self):
        return f"u{self.ur}_{self.uh}_{self.ul}_i{self.ir}_{self.ih}_{self.il}_s{self.seed}"


class StaticFactor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bias = torch.nn.Parameter(torch.tensor(3.5, device="cuda"))
        self.ua = torch.nn.Parameter(torch.randn(cfg.uh, cfg.ur, device="cuda") * 0.01)
        self.ub = torch.nn.Parameter(torch.randn(cfg.ul, cfg.ur, device="cuda") * 0.01)
        self.ia = torch.nn.Parameter(torch.randn(cfg.ih, cfg.ir, device="cuda") * 0.01)
        self.ib = torch.nn.Parameter(torch.randn(cfg.il, cfg.ir, device="cuda") * 0.01)

    def forward(self, uhi, ulo, ihi, ilo):
        user = (self.ua[uhi] * self.ub[ulo]).sum(dim=1)
        item = (self.ia[ihi] * self.ib[ilo]).sum(dim=1)
        return self.bias + user + item


def train_one(cfg, users, items, u_np, i_np, y_np):
    if cfg.params > PARAM_BUDGET:
        return None
    torch.manual_seed(20260605 + cfg.seed)
    n = y_np.shape[0]
    uhi_np = (u_np.astype(np.int64) * cfg.uh // users).astype(np.int64)
    ulo_np = (u_np.astype(np.int64) % cfg.ul).astype(np.int64)
    ihi_np = (i_np.astype(np.int64) * cfg.ih // items).astype(np.int64)
    ilo_np = (i_np.astype(np.int64) % cfg.il).astype(np.int64)
    uhi = torch.tensor(uhi_np, device="cuda", dtype=torch.long)
    ulo = torch.tensor(ulo_np, device="cuda", dtype=torch.long)
    ihi = torch.tensor(ihi_np, device="cuda", dtype=torch.long)
    ilo = torch.tensor(ilo_np, device="cuda", dtype=torch.long)
    y = torch.tensor(y_np, device="cuda")
    all_idx = torch.arange(n, device="cuda", dtype=torch.long)
    model = StaticFactor(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    lr = 0.03
    min_lr = 8e-7
    stale = 0
    best = 99.0
    best_state = None

    def full_rmse():
        se = 0.0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = torch.clamp(model(uhi[ids], ulo[ids], ihi[ids], ilo[ids]), 0.5, 5.0)
                err = pred - y[ids]
                se += torch.sum(err * err).item()
        return math.sqrt(se / n)

    print(f"\nCONFIG {cfg.label} params {cfg.params}", flush=True)
    while True:
        order = all_idx[torch.randperm(n, device="cuda")]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = model(uhi[ids], ulo[ids], ihi[ids], ilo[ids])
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
            if lr > min_lr:
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
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    configs = [
        Config(192, 294, 2, 1, 1, 0),
        Config(150, 220, 2, 60, 70, 2),
        Config(140, 220, 2, 70, 68, 2),
        Config(128, 224, 2, 80, 66, 2),
        Config(160, 240, 2, 48, 70, 1),
        Config(96, 160, 2, 96, 140, 2),
    ]
    best = (99.0, None, None)
    t0 = time.time()
    for cfg in configs:
        result = train_one(cfg, users, items, u, i, y)
        if result is None:
            continue
        score, state = result
        print(f"RESULT {cfg.label} {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, cfg, state)
            np.savez(
                OUT_DIR / "static_factor_under1k.npz",
                best_rmse=np.array(score, dtype=np.float32),
                uh=np.array(cfg.uh, dtype=np.int32),
                ul=np.array(cfg.ul, dtype=np.int32),
                ur=np.array(cfg.ur, dtype=np.int32),
                ih=np.array(cfg.ih, dtype=np.int32),
                il=np.array(cfg.il, dtype=np.int32),
                ir=np.array(cfg.ir, dtype=np.int32),
                param_count=np.array(cfg.params, dtype=np.int32),
                **state,
            )
    print(f"BEST {best[1].label} rmse {best[0]:.9f} sec {time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
