import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402


PARAM_BUDGET = 999
MIN_GAIN = 1e-5
MAX_ROUNDS_WITHOUT_GAIN = 2


@dataclass(frozen=True)
class Candidate:
    name: str
    bins: tuple[str, ...]
    size: int
    shrink: float
    lr: float


def rmse(y, pred):
    clipped = np.clip(pred, 0.5, 5.0)
    return math.sqrt(float(np.mean((clipped - y) ** 2)))


def uniform_bins(values, n_bins, lo=None, hi=None):
    values = values.astype(np.float32, copy=False)
    if lo is None:
        lo = float(np.min(values))
    if hi is None:
        hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(values.shape, dtype=np.int32)
    scaled = (values - lo) * (float(n_bins) / (hi - lo))
    out = scaled.astype(np.int32)
    np.clip(out, 0, n_bins - 1, out=out)
    return out


def rank_bins(values, n_bins):
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.shape[0], dtype=np.int32)
    out[order] = (np.arange(values.shape[0], dtype=np.int64) * n_bins // values.shape[0]).astype(np.int32)
    return out


def combine_bins(bin_arrays, sizes):
    idx = bin_arrays[0].astype(np.int32, copy=True)
    mult = int(sizes[0])
    for arr, size in zip(bin_arrays[1:], sizes[1:]):
        idx += mult * arr.astype(np.int32, copy=False)
        mult *= int(size)
    return idx


def fit_table(residual, idx, size, shrink):
    sums = np.bincount(idx, weights=residual, minlength=size).astype(np.float64)
    counts = np.bincount(idx, minlength=size).astype(np.float64)
    return (sums / (counts + shrink)).astype(np.float32)


def build_stats_features(users, items):
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))

    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean

    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float32)
    user_sum = np.bincount(inc_u[::2], weights=residual[::2], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[::2], minlength=users).astype(np.float32)

    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)

    uc = user_count[u]
    ic = item_count[i]
    us = user_sum[u]
    is_ = item_sum[i]

    def avg(sum_v, count_v, shrink):
        return np.where(count_v > 0, sum_v / (count_v + shrink), 0.0).astype(np.float32)

    features = {
        "uid": u.astype(np.float32),
        "iid": i.astype(np.float32),
        "ucount": uc.astype(np.float32),
        "icount": ic.astype(np.float32),
        "log_uc": np.log1p(uc).astype(np.float32),
        "log_ic": np.log1p(ic).astype(np.float32),
        "uavg0": avg(us, uc, 0.0),
        "iavg0": avg(is_, ic, 0.0),
        "uavg5": avg(us, uc, 5.0),
        "iavg3": avg(is_, ic, 3.0),
        "uavg30": avg(us, uc, 30.0),
        "iavg8": avg(is_, ic, 8.0),
    }
    return features


def build_bins(values):
    specs: dict[str, tuple[np.ndarray, int]] = {}
    users = int(values["uid"].max()) + 1
    items = int(values["iid"].max()) + 1

    for n in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        if n <= users:
            specs[f"uid{n}"] = ((values["uid"].astype(np.int64) * n // users).astype(np.int32), n)
        if n <= items:
            specs[f"iid{n}"] = ((values["iid"].astype(np.int64) * n // items).astype(np.int32), n)

    fixed_ranges = {
        "base": (0.5, 5.0),
        "uavg0": (-3.0, 3.0),
        "iavg0": (-3.0, 3.0),
        "uavg5": (-2.5, 2.5),
        "iavg3": (-2.5, 2.5),
        "uavg30": (-1.5, 1.5),
        "iavg8": (-1.5, 1.5),
        "log_uc": (0.0, 8.0),
        "log_ic": (0.0, 8.0),
    }
    for key, (lo, hi) in fixed_ranges.items():
        for n in (8, 16, 24, 32, 48, 64, 96, 128):
            specs[f"{key}{n}"] = (uniform_bins(values[key], n, lo, hi), n)
        for n in (8, 16, 32, 64):
            specs[f"{key}q{n}"] = (rank_bins(values[key], n), n)

    return specs


def candidate_list(specs):
    out: list[Candidate] = []
    one_d = [
        "uid",
        "iid",
        "base",
        "uavg0",
        "iavg0",
        "uavg5",
        "iavg3",
        "uavg30",
        "iavg8",
        "log_uc",
        "log_ic",
    ]
    for prefix in one_d:
        for name, (_, size) in specs.items():
            if name.startswith(prefix) and size <= PARAM_BUDGET:
                for shrink in (5.0, 20.0, 100.0):
                    out.append(Candidate(name, (name,), size, shrink, 1.0))

    pair_prefixes = [
        ("uid", "iid"),
        ("uid", "base"),
        ("iid", "base"),
        ("uavg", "iavg"),
        ("uavg", "base"),
        ("iavg", "base"),
        ("log_uc", "log_ic"),
        ("log_uc", "base"),
        ("log_ic", "base"),
        ("uid", "uavg"),
        ("iid", "iavg"),
    ]
    keys = list(specs.keys())
    for a_prefix, b_prefix in pair_prefixes:
        a_keys = [k for k in keys if k.startswith(a_prefix)]
        b_keys = [k for k in keys if k.startswith(b_prefix)]
        for a in a_keys:
            for b in b_keys:
                if a == b:
                    continue
                size = specs[a][1] * specs[b][1]
                if 32 <= size <= PARAM_BUDGET:
                    for shrink in (10.0, 50.0, 200.0):
                        out.append(Candidate(f"{a}x{b}", (a, b), size, shrink, 1.0))
    return out


def evaluate_candidate(y, pred, residual, specs, cand):
    arrays = [specs[name][0] for name in cand.bins]
    sizes = [specs[name][1] for name in cand.bins]
    idx = arrays[0] if len(arrays) == 1 else combine_bins(arrays, sizes)
    table = fit_table(residual, idx, cand.size, cand.shrink)
    trial = pred + cand.lr * table[idx]
    return rmse(y, trial), idx, table


def main():
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    values = build_stats_features(P_m.shape[0], Q_m.shape[0])
    values["base"] = np.clip(base, 0.5, 5.0).astype(np.float32)

    specs = build_bins(values)
    cands = candidate_list(specs)
    print(f"budget {PARAM_BUDGET} bins {len(specs)} candidates {len(cands)}", flush=True)
    pred = np.clip(base, 0.5, 5.0).astype(np.float32)
    best = rmse(y, pred)
    print(f"initial {best:.9f}", flush=True)

    chosen = []
    used = 0
    stale_rounds = 0
    while used < PARAM_BUDGET and stale_rounds < MAX_ROUNDS_WITHOUT_GAIN:
        residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
        best_local = None
        for cand in cands:
            if used + cand.size > PARAM_BUDGET:
                continue
            score, idx, table = evaluate_candidate(y, pred, residual, specs, cand)
            if best_local is None or score < best_local[0]:
                best_local = (score, cand, idx, table)
        if best_local is None:
            break
        score, cand, idx, table = best_local
        gain = best - score
        print(
            f"try round {len(chosen)+1} cand {cand.name} size {cand.size} "
            f"shrink {cand.shrink:g} rmse {score:.9f} gain {gain:.9f}",
            flush=True,
        )
        if gain < MIN_GAIN:
            stale_rounds += 1
            continue
        pred += table[idx]
        used += cand.size
        best = score
        chosen.append((cand, table.copy()))
        stale_rounds = 0
        print(f"accepted params {used} rmse {best:.9f}", flush=True)

    out_path = OUT_DIR / "small_param_tables_under1k.npz"
    save_dict = {
        "best_rmse": np.array(best, dtype=np.float32),
        "param_count": np.array(used, dtype=np.int32),
        "chosen_names": np.array([c.name for c, _ in chosen]),
        "chosen_sizes": np.array([c.size for c, _ in chosen], dtype=np.int32),
        "chosen_shrinks": np.array([c.shrink for c, _ in chosen], dtype=np.float32),
    }
    for n, (cand, table) in enumerate(chosen):
        save_dict[f"table_{n}"] = table
        save_dict[f"bins_{n}"] = np.array(cand.bins)
    np.savez(out_path, **save_dict)
    print(f"saved {out_path} rmse {best:.9f} params {used}", flush=True)


if __name__ == "__main__":
    main()
