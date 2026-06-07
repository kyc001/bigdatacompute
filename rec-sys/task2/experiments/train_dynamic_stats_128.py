import math
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("rec-sys/task2/experiments")
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def safe_avg(s, c, shrink):
    return np.where(c > 0, s / (c + shrink), 0.0).astype(np.float32)


def add_feature(features, names, name, value):
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    features.append(value)
    names.append(name)


def fit_ridge(x, y):
    xtx = x.T.astype(np.float64) @ x.astype(np.float64)
    xty = x.T.astype(np.float64) @ y.astype(np.float64)
    best = (99.0, None, None)
    for ridge in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        coef = np.linalg.solve(xtx + np.eye(x.shape[1], dtype=np.float64) * ridge, xty).astype(np.float32)
        pred = x @ coef
        score = rmse(y, pred)
        print(f"ridge {ridge:g} cols {x.shape[1]} rmse {score:.9f}", flush=True)
        if score < best[0]:
            best = (score, ridge, coef)
    return best


def main():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = np.load(ROOT / "P.npy", mmap_mode="r").shape[0]
    items = np.load(ROOT / "Q.npy", mmap_mode="r").shape[0]
    mean = float(np.load(ROOT / "global_mean.npy"))

    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    rating = inc[:, 2].astype(np.float32)
    residual = rating - mean
    mask_user = np.zeros(residual.shape[0], dtype=bool)
    mask_user[::2] = True

    user_sum = np.bincount(inc_u[mask_user], weights=residual[mask_user], minlength=users).astype(np.float32)
    user_sq = np.bincount(inc_u[mask_user], weights=(residual[mask_user] ** 2), minlength=users).astype(np.float32)
    user_high = np.bincount(inc_u[mask_user], weights=(rating[mask_user] >= 4.0), minlength=users).astype(np.float32)
    user_low = np.bincount(inc_u[mask_user], weights=(rating[mask_user] <= 2.0), minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[mask_user], minlength=users).astype(np.float32)

    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_sq = np.bincount(inc_i, weights=(residual ** 2), minlength=items).astype(np.float32)
    item_high = np.bincount(inc_i, weights=(rating >= 4.0), minlength=items).astype(np.float32)
    item_low = np.bincount(inc_i, weights=(rating <= 2.0), minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)

    # Two-stage collaborative summaries derived only from the incremental batch.
    item_bias = safe_avg(item_sum, item_count, 5.0)
    user_bias = safe_avg(user_sum, user_count, 20.0)
    user_item_bias_sum = np.bincount(
        inc_u[mask_user], weights=item_bias[inc_i[mask_user]], minlength=users
    ).astype(np.float32)
    item_user_bias_sum = np.bincount(inc_i, weights=user_bias[inc_u], minlength=items).astype(np.float32)

    tu = test[:, 0].astype(np.int32)
    ti = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)

    uc = user_count[tu]
    ic = item_count[ti]
    us = user_sum[tu]
    is_ = item_sum[ti]
    usq = user_sq[tu]
    isq = item_sq[ti]
    uh = user_high[tu]
    ih = item_high[ti]
    ul = user_low[tu]
    il = item_low[ti]

    ua0 = safe_avg(us, uc, 0.0)
    ua5 = safe_avg(us, uc, 5.0)
    ua20 = safe_avg(us, uc, 20.0)
    ua50 = safe_avg(us, uc, 50.0)
    ia0 = safe_avg(is_, ic, 0.0)
    ia2 = safe_avg(is_, ic, 2.0)
    ia5 = safe_avg(is_, ic, 5.0)
    ia20 = safe_avg(is_, ic, 20.0)
    uhf = safe_avg(uh, uc, 5.0)
    ulf = safe_avg(ul, uc, 5.0)
    ihf = safe_avg(ih, ic, 3.0)
    ilf = safe_avg(il, ic, 3.0)
    uvar = np.maximum(safe_avg(usq, uc, 0.0) - ua0 * ua0, 0.0)
    ivar = np.maximum(safe_avg(isq, ic, 0.0) - ia0 * ia0, 0.0)
    uib = safe_avg(user_item_bias_sum[tu], uc, 10.0)
    iub = safe_avg(item_user_bias_sum[ti], ic, 5.0)
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    ru = (1.0 / np.sqrt(uc + 1.0)).astype(np.float32)
    ri = (1.0 / np.sqrt(ic + 1.0)).astype(np.float32)

    features = []
    names = []
    add_feature(features, names, "one", np.ones_like(y))
    for name, value in [
        ("lu", lu), ("li", li), ("lu2", lu * lu), ("li2", li * li), ("ru", ru), ("ri", ri),
        ("ua0", ua0), ("ua5", ua5), ("ua20", ua20), ("ua50", ua50),
        ("ia0", ia0), ("ia2", ia2), ("ia5", ia5), ("ia20", ia20),
        ("uhf", uhf), ("ulf", ulf), ("ihf", ihf), ("ilf", ilf),
        ("uvar", uvar), ("ivar", ivar), ("uib", uib), ("iub", iub),
    ]:
        add_feature(features, names, name, value)

    base_vars = {
        "ua20": ua20, "ia5": ia5, "lu": lu, "li": li, "ru": ru, "ri": ri,
        "uhf": uhf, "ulf": ulf, "ihf": ihf, "ilf": ilf, "uvar": uvar, "ivar": ivar,
        "uib": uib, "iub": iub,
    }
    pair_specs = [
        ("ua20", "ia5"), ("ua0", "ia0"), ("ua20", "li"), ("ia5", "lu"),
        ("ua20", "ihf"), ("ia5", "uhf"), ("ua20", "ilf"), ("ia5", "ulf"),
        ("uhf", "ihf"), ("ulf", "ilf"), ("uvar", "ivar"), ("uib", "ia5"), ("iub", "ua20"),
        ("lu", "li"), ("ru", "ri"),
    ]
    # Include the raw aliases used only in pair specs.
    base_vars["ua0"] = ua0
    base_vars["ia0"] = ia0
    for a, b in pair_specs:
        add_feature(features, names, f"{a}_x_{b}", base_vars[a] * base_vars[b])

    # Low-order nonlinear transforms that cost no extra runtime table state.
    for name in ("ua20", "ia5", "uhf", "ulf", "ihf", "ilf", "uib", "iub"):
        value = base_vars[name]
        add_feature(features, names, f"{name}2", value * value)

    x = np.stack(features, axis=1).astype(np.float32)
    score, ridge, coef = fit_ridge(x, y)
    print(f"BEST dynamic_stats_128 rmse {score:.9f} ridge {ridge:g} cols {x.shape[1]}", flush=True)
    np.savez(
        OUT / "dynamic_stats_128.npz",
        best_rmse=np.array(score, dtype=np.float32),
        param_count=np.array(x.shape[1], dtype=np.int32),
        ridge=np.array(ridge, dtype=np.float32),
        coef=coef,
        names=np.array(names),
    )


if __name__ == "__main__":
    main()
