from pathlib import Path

import numpy as np
import torch

import train_factorized_128 as base


OUT_DIR = Path("rec-sys/task2/experiments")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    users, u, _, y, us, uc, is_, ic = base.load_base_arrays()
    configs = []
    for seed in (0, 1):
        for high, low in [(32, 89), (40, 81), (48, 73), (56, 65), (64, 57), (80, 41)]:
            configs.append(base.Config((), (), high, low, 1, seed))
        for high, low in [(40, 77), (48, 69), (56, 61), (64, 53), (72, 45)]:
            configs.append(base.Config((20.0,), (5.0,), high, low, 1, seed))
        for high, low in [(44, 65), (52, 57), (60, 49)]:
            configs.append(base.Config((0.0, 5.0, 30.0), (0.0, 3.0, 8.0), high, low, 1, seed))

    best_score = 99.0
    best_cfg = None
    best_state = None
    feature_cache = {}
    existing = OUT_DIR / "factorized_128.npz"
    if existing.exists():
        old = np.load(existing)
        best_score = float(old["best_rmse"])
        best_cfg = "existing"
        print(f"starting from existing {best_score:.9f}", flush=True)

    for cfg in configs:
        key = (cfg.user_shrinks, cfg.item_shrinks)
        if key not in feature_cache:
            x = base.make_features(cfg, us, uc, is_, ic)
            coef, base_rmse = base.fit_ridge(x, y)
            feature_cache[key] = (x, coef, base_rmse)
            print(f"\nFEATURES {key} coef {cfg.coef_count} shrinks {cfg.shrink_count} base {base_rmse:.9f}", flush=True)
        x, coef, base_rmse = feature_cache[key]
        print(f"CONFIG {cfg.label} params {cfg.param_count} base {base_rmse:.9f}", flush=True)
        result = base.train_one(cfg, users, u, y, x, coef)
        if result is None:
            continue
        score, state = result
        print(f"RESULT {cfg.label} rmse {score:.9f}", flush=True)
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_state = state
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
            print(f"NEW_BEST {score:.9f} saved factorized_128.npz", flush=True)
    print(f"BEST {best_cfg} {best_score:.9f}", flush=True)


if __name__ == "__main__":
    main()
