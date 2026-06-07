import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAM_BUDGET = 128


@dataclass(frozen=True)
class Config:
    user_shrinks: tuple[float, ...]
    item_shrinks: tuple[float, ...]
    high: int
    low: int
    rank: int
    seed: int = 0

    @property
    def coef_count(self):
        return 7 + len(self.user_shrinks) + len(self.item_shrinks)

    @property
    def shrink_count(self):
        return len(self.user_shrinks) + len(self.item_shrinks)

    @property
    def factor_count(self):
        return self.rank * (self.high + self.low)

    @property
    def param_count(self):
        return self.coef_count + self.shrink_count + self.factor_count

    @property
    def label(self):
        return (
            f"c{self.coef_count}_su{len(self.user_shrinks)}_si{len(self.item_shrinks)}_"
            f"r{self.rank}_h{self.high}_l{self.low}_s{self.seed}"
        )


def load_base_arrays():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = int(np.load(ROOT / "P.npy", mmap_mode="r").shape[0])
    items = int(np.load(ROOT / "Q.npy", mmap_mode="r").shape[0])
    mean = float(np.load(ROOT / "global_mean.npy"))

    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    user_sum = np.bincount(inc_u[::2], weights=residual[::2], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::2], minlength=users).astype(np.float32)
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)

    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    return users, u, i, y, user_sum[u], user_count[u], item_sum[i], item_count[i]


def make_features(cfg, us, uc, is_, ic):
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    x = np.empty((uc.shape[0], cfg.coef_count), dtype=np.float32)
    x[:, 0] = 1.0
    x[:, 1] = lu
    x[:, 2] = li
    x[:, 3] = lu * lu
    x[:, 4] = li * li
    x[:, 5] = 1.0 / np.sqrt(uc + 1.0)
    x[:, 6] = 1.0 / np.sqrt(ic + 1.0)
    col = 7
    for shrink in cfg.user_shrinks:
        x[:, col] = np.where(uc > 0, us / (uc + shrink), 0.0)
        col += 1
    for shrink in cfg.item_shrinks:
        x[:, col] = np.where(ic > 0, is_ / (ic + shrink), 0.0)
        col += 1
    return x


def fit_ridge(x, y):
    xtx = x.T @ x
    xty = x.T @ y
    coef = np.linalg.solve(
        xtx.astype(np.float64) + np.eye(x.shape[1], dtype=np.float64) * 1e-4,
        xty.astype(np.float64),
    ).astype(np.float32)
    pred = x @ coef
    score = math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))
    return coef, score


class Model(torch.nn.Module):
    def __init__(self, cfg, init_coef):
        super().__init__()
        self.coef = torch.nn.Parameter(torch.tensor(init_coef, device="cuda"))
        self.a = torch.nn.Parameter(torch.randn(cfg.high, cfg.rank, device="cuda") * 0.01)
        self.b = torch.nn.Parameter(torch.randn(cfg.low, cfg.rank, device="cuda") * 0.002)

    def forward(self, x, hi, lo):
        base = x.matmul(self.coef)
        factor = (self.a[hi] * self.b[lo]).sum(dim=1)
        return base + factor


def train_one(cfg, users, u_np, y_np, x_np, init_coef):
    if cfg.param_count > PARAM_BUDGET:
        return None
    torch.manual_seed(20260605 + cfg.seed)
    n = y_np.shape[0]
    hi_np = (u_np.astype(np.int64) * cfg.high // users).astype(np.int64)
    lo_np = (u_np.astype(np.int64) % cfg.low).astype(np.int64)
    x = torch.tensor(x_np, device="cuda")
    hi = torch.tensor(hi_np, device="cuda", dtype=torch.long)
    lo = torch.tensor(lo_np, device="cuda", dtype=torch.long)
    y = torch.tensor(y_np, device="cuda")
    all_idx = torch.arange(n, device="cuda", dtype=torch.long)
    model = Model(cfg, init_coef)
    opt = torch.optim.Adam(
        [
            {"params": [model.coef], "lr": 0.003},
            {"params": [model.a, model.b], "lr": 0.03},
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
                pred = torch.clamp(model(x[ids], hi[ids], lo[ids]), 0.5, 5.0)
                err = pred - y[ids]
                se += torch.sum(err * err).item()
        return math.sqrt(se / n)

    while True:
        order = all_idx[torch.randperm(n, device="cuda")]
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = model(x[ids], hi[ids], lo[ids])
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
            print(f"{cfg.label} rmse {best:.9f} factor_lr {factor_lr:.2g}", flush=True)
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
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    users, u, _, y, us, uc, is_, ic = load_base_arrays()

    shrink_sets = [
        ((0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0),
         (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 50.0)),
        ((0.0, 5.0, 20.0, 100.0), (0.0, 2.0, 5.0, 20.0)),
        ((0.0, 5.0, 30.0), (0.0, 3.0, 8.0)),
        ((5.0, 30.0), (3.0, 8.0)),
        ((20.0,), (5.0,)),
    ]
    configs: list[Config] = []
    for user_shrinks, item_shrinks in shrink_sets:
        base_count = (7 + len(user_shrinks) + len(item_shrinks)) + len(user_shrinks) + len(item_shrinks)
        remaining = PARAM_BUDGET - base_count
        for rank, pairs in [
            (1, [(48, remaining - 48), (64, remaining - 64), (80, remaining - 80)]),
            (2, [(20, remaining // 2 - 20), (24, remaining // 2 - 24), (32, remaining // 2 - 32)]),
        ]:
            for high, low in pairs:
                if high >= 4 and low >= 4:
                    cfg = Config(user_shrinks, item_shrinks, high, low, rank, 0)
                    if cfg.param_count <= PARAM_BUDGET:
                        configs.append(cfg)

    best = (99.0, None, None)
    feature_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], tuple[np.ndarray, np.ndarray, float]] = {}
    t0 = time.time()
    for cfg in configs:
        key = (cfg.user_shrinks, cfg.item_shrinks)
        if key not in feature_cache:
            x = make_features(cfg, us, uc, is_, ic)
            init_coef, base_rmse = fit_ridge(x, y)
            feature_cache[key] = (x, init_coef, base_rmse)
            print(f"\nFEATURES coef {cfg.coef_count} shrinks {cfg.shrink_count} base {base_rmse:.9f}", flush=True)
        x, init_coef, base_rmse = feature_cache[key]
        print(f"CONFIG {cfg.label} params {cfg.param_count} base {base_rmse:.9f}", flush=True)
        result = train_one(cfg, users, u, y, x, init_coef)
        if result is None:
            continue
        score, state = result
        print(f"RESULT {cfg.label} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, cfg, state)
            np.savez(
                OUT_DIR / "factorized_128.npz",
                best_rmse=np.array(score, dtype=np.float32),
                param_count=np.array(cfg.param_count, dtype=np.int32),
                coef_count=np.array(cfg.coef_count, dtype=np.int32),
                user_shrinks=np.array(cfg.user_shrinks, dtype=np.float32),
                item_shrinks=np.array(cfg.item_shrinks, dtype=np.float32),
                high=np.array(cfg.high, dtype=np.int32),
                low=np.array(cfg.low, dtype=np.int32),
                rank=np.array(cfg.rank, dtype=np.int32),
                **state,
            )
            print(f"NEW_BEST saved factorized_128.npz {score:.9f} params {cfg.param_count}", flush=True)
    print(f"BEST {best[1].label} rmse {best[0]:.9f} sec {time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
