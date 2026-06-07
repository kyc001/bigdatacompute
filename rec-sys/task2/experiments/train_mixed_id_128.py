import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import train_factorized_128 as base128


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAM_BUDGET = 128
USER_SHRINKS = (20.0,)
ITEM_SHRINKS = (5.0,)


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
    def coef_count(self):
        return 7 + len(USER_SHRINKS) + len(ITEM_SHRINKS)

    @property
    def shrink_count(self):
        return len(USER_SHRINKS) + len(ITEM_SHRINKS)

    @property
    def factor_count(self):
        return self.ur * (self.uh + self.ul) + self.ir * (self.ih + self.il)

    @property
    def param_count(self):
        return self.coef_count + self.shrink_count + self.factor_count

    @property
    def label(self):
        return f"u{self.ur}_{self.uh}_{self.ul}_i{self.ir}_{self.ih}_{self.il}_s{self.seed}"


class Model(torch.nn.Module):
    def __init__(self, cfg, init_coef):
        super().__init__()
        self.coef = torch.nn.Parameter(torch.tensor(init_coef, device="cuda"))
        self.ua = torch.nn.Parameter(torch.randn(cfg.uh, cfg.ur, device="cuda") * 0.01)
        self.ub = torch.nn.Parameter(torch.randn(cfg.ul, cfg.ur, device="cuda") * 0.002)
        self.ia = torch.nn.Parameter(torch.randn(cfg.ih, cfg.ir, device="cuda") * 0.01)
        self.ib = torch.nn.Parameter(torch.randn(cfg.il, cfg.ir, device="cuda") * 0.002)

    def forward(self, x, uhi, ulo, ihi, ilo):
        stat = x.matmul(self.coef)
        user = (self.ua[uhi] * self.ub[ulo]).sum(dim=1)
        item = (self.ia[ihi] * self.ib[ilo]).sum(dim=1)
        return stat + user + item


def train_one(cfg, users, items, u_np, i_np, y_np, x_np, init_coef):
    if cfg.param_count > PARAM_BUDGET:
        return None
    torch.manual_seed(20260605 + cfg.seed)
    n = y_np.shape[0]
    uhi_np = (u_np.astype(np.int64) * cfg.uh // users).astype(np.int64)
    ulo_np = (u_np.astype(np.int64) % cfg.ul).astype(np.int64)
    ihi_np = (i_np.astype(np.int64) * cfg.ih // items).astype(np.int64)
    ilo_np = (i_np.astype(np.int64) % cfg.il).astype(np.int64)
    x = torch.tensor(x_np, device="cuda")
    uhi = torch.tensor(uhi_np, device="cuda", dtype=torch.long)
    ulo = torch.tensor(ulo_np, device="cuda", dtype=torch.long)
    ihi = torch.tensor(ihi_np, device="cuda", dtype=torch.long)
    ilo = torch.tensor(ilo_np, device="cuda", dtype=torch.long)
    y = torch.tensor(y_np, device="cuda")
    all_idx = torch.arange(n, device="cuda", dtype=torch.long)
    model = Model(cfg, init_coef)
    opt = torch.optim.Adam(
        [
            {"params": [model.coef], "lr": 0.003},
            {"params": [model.ua, model.ub, model.ia, model.ib], "lr": 0.03},
        ]
    )
    factor_lr = 0.03
    coef_lr = 0.003
    stale = 0
    best = 99.0
    best_state = None

    def full_rmse():
        se = 0.0
        with torch.no_grad():
            for start in range(0, n, 262144):
                ids = all_idx[start : start + 262144]
                pred = torch.clamp(model(x[ids], uhi[ids], ulo[ids], ihi[ids], ilo[ids]), 0.5, 5.0)
                err = pred - y[ids]
                se += torch.sum(err * err).item()
        return math.sqrt(se / n)

    while True:
        order = all_idx[torch.randperm(n, device="cuda")]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = model(x[ids], uhi[ids], ulo[ids], ihi[ids], ilo[ids])
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
            print(f"{cfg.label} rmse {best:.9f} lr {factor_lr:.2g}", flush=True)
        else:
            stale += 1
        if stale >= 24:
            if factor_lr > 8e-7:
                factor_lr *= 0.4
                coef_lr *= 0.4
                opt.param_groups[0]["lr"] = coef_lr
                opt.param_groups[1]["lr"] = factor_lr
                stale = 0
            else:
                break
    return best, best_state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    users, u, i, y, us, uc, is_, ic = base128.load_base_arrays()
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    template = base128.Config(USER_SHRINKS, ITEM_SHRINKS, 4, 4, 1, 0)
    x = base128.make_features(template, us, uc, is_, ic)
    init_coef, base_rmse = base128.fit_ridge(x, y)
    print(f"base rmse {base_rmse:.9f} coef {template.coef_count} params used by stat {template.coef_count + template.shrink_count}", flush=True)

    configs = []
    allocations = [
        (48, 33, 18, 18),
        (48, 25, 22, 22),
        (40, 33, 22, 22),
        (40, 25, 26, 26),
        (32, 33, 30, 22),
        (32, 25, 32, 28),
        (24, 33, 36, 24),
        (56, 21, 20, 20),
        (64, 17, 18, 18),
    ]
    for seed in (0, 1):
        for uh, ul, ih, il in allocations:
            configs.append(Config(uh, ul, 1, ih, il, 1, seed))
        for uh, ul, ih, il in [(16, 16, 12, 12), (18, 14, 14, 12), (20, 12, 12, 14)]:
            configs.append(Config(uh, ul, 2, ih, il, 2, seed))

    best = (99.0, None, None)
    existing = OUT_DIR / "mixed_id_128.npz"
    for cfg in configs:
        if cfg.param_count > PARAM_BUDGET:
            continue
        print(f"\nCONFIG {cfg.label} params {cfg.param_count}", flush=True)
        result = train_one(cfg, users, items, u, i, y, x, init_coef)
        if result is None:
            continue
        score, state = result
        print(f"RESULT {cfg.label} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, cfg, state)
            np.savez(
                existing,
                best_rmse=np.array(score, dtype=np.float32),
                param_count=np.array(cfg.param_count, dtype=np.int32),
                user_shrinks=np.array(USER_SHRINKS, dtype=np.float32),
                item_shrinks=np.array(ITEM_SHRINKS, dtype=np.float32),
                uh=np.array(cfg.uh, dtype=np.int32),
                ul=np.array(cfg.ul, dtype=np.int32),
                ur=np.array(cfg.ur, dtype=np.int32),
                ih=np.array(cfg.ih, dtype=np.int32),
                il=np.array(cfg.il, dtype=np.int32),
                ir=np.array(cfg.ir, dtype=np.int32),
                **state,
            )
            print(f"NEW_BEST saved {existing} rmse {score:.9f}", flush=True)
    print(f"BEST {best[1]} {best[0]:.9f}", flush=True)


if __name__ == "__main__":
    main()
