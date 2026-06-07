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
import train_small_param_tables as tables  # noqa: E402


PARAM_BUDGET = 999
MIN_DELTA = 2e-6
PATIENCE = 20


@dataclass
class Group:
    name: str
    idx: np.ndarray
    size: int
    shrink: float
    values: np.ndarray


def rmse(y, pred):
    return math.sqrt(float(np.mean((np.clip(pred, 0.5, 5.0) - y) ** 2)))


def hash_bins(ids, n_bins, seed):
    x = ids.astype(np.uint64)
    h = x * np.uint64(0x9E3779B185EBCA87 + seed * 0x100000001B3)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= h >> np.uint64(29)
    return (h % np.uint64(n_bins)).astype(np.int32)


def pair_hash_bins(u, i, n_bins, seed):
    x = u.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
    x ^= i.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F + seed * 1315423911)
    x ^= x >> np.uint64(32)
    return (x % np.uint64(n_bins)).astype(np.int32)


def make_group(name, idx, size, shrink):
    return Group(name, idx.astype(np.int32, copy=False), int(size), float(shrink), np.zeros(int(size), dtype=np.float32))


def combined(specs, names):
    arrays = [specs[name][0] for name in names]
    sizes = [specs[name][1] for name in names]
    return tables.combine_bins(arrays, sizes), int(np.prod(sizes))


def run_backfit(y, base, group_templates, label):
    groups = [
        Group(g.name, g.idx, g.size, g.shrink, np.zeros(g.size, dtype=np.float32))
        for g in group_templates
    ]
    total = sum(g.size for g in groups)
    pred = np.clip(base, 0.5, 5.0).astype(np.float32)
    best = rmse(y, pred)
    best_values = [g.values.copy() for g in groups]
    stale = 0
    epoch = 0
    print(f"\nCONFIG {label} params {total} initial {best:.9f}", flush=True)

    while True:
        epoch += 1
        for g in groups:
            pred -= g.values[g.idx]
            residual = (y - np.clip(pred, 0.5, 5.0)).astype(np.float32)
            sums = np.bincount(g.idx, weights=residual, minlength=g.size).astype(np.float64)
            counts = np.bincount(g.idx, minlength=g.size).astype(np.float64)
            new_values = (sums / (counts + g.shrink)).astype(np.float32)
            g.values = new_values
            pred += g.values[g.idx]

        score = rmse(y, pred)
        improved = score < best - MIN_DELTA
        if improved:
            best = score
            best_values = [g.values.copy() for g in groups]
            stale = 0
        else:
            stale += 1
        if epoch <= 5 or epoch % 5 == 0 or improved:
            print(f"epoch {epoch} rmse {score:.9f} best {best:.9f} stale {stale}", flush=True)
        if stale >= PATIENCE:
            break

    for g, values in zip(groups, best_values):
        g.values = values
    return best, groups


def main():
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    values = tables.build_stats_features(P_m.shape[0], Q_m.shape[0])
    values["base"] = np.clip(base, 0.5, 5.0).astype(np.float32)
    specs = tables.build_bins(values)

    u = values["uid"].astype(np.int32)
    i = values["iid"].astype(np.int32)

    def g1(name, shrink=20.0):
        idx, size = specs[name]
        return make_group(name, idx, size, shrink)

    def gp(names, shrink=50.0):
        idx, size = combined(specs, names)
        return make_group("x".join(names), idx, size, shrink)

    configs: list[tuple[str, list[Group]]] = []

    configs.append(
        (
            "greedy_known",
            [
                gp(("uid96", "uavg0q8"), 10.0),
                g1("iid192", 5.0),
                g1("uavg30q32", 100.0),
            ],
        )
    )
    configs.append(
        (
            "uniform_user_stat_dense",
            [
                gp(("uid64", "uavg08"), 10.0),
                g1("iid256", 5.0),
                g1("iavg0q64", 50.0),
                g1("base64", 50.0),
                g1("log_ic32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_stat_item_stat",
            [
                gp(("uid64", "uavg0q8"), 10.0),
                gp(("iid32", "iavg0q8"), 20.0),
                g1("base64", 50.0),
                g1("uavg30q32", 80.0),
                g1("iavg8q32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "balanced_interactions",
            [
                gp(("uid32", "uavg0q16"), 10.0),
                gp(("iid32", "iavg0q8"), 20.0),
                g1("base96", 50.0),
                g1("log_uc64", 100.0),
                g1("log_ic64", 100.0),
            ],
        )
    )
    configs.append(
        (
            "id_splines",
            [
                g1("uid512", 20.0),
                g1("iid384", 10.0),
                g1("base64", 50.0),
                g1("uavg30q32", 100.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash3_item",
            [
                make_group("uh256_s0", hash_bins(u, 256, 0), 256, 20.0),
                make_group("uh256_s1", hash_bins(u, 256, 1), 256, 20.0),
                make_group("uh256_s2", hash_bins(u, 256, 2), 256, 20.0),
                g1("iid192", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash2_item",
            [
                make_group("uh384_s0", hash_bins(u, 384, 0), 384, 20.0),
                make_group("uh384_s1", hash_bins(u, 384, 1), 384, 20.0),
                g1("iid192", 10.0),
                g1("iavg8q32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash1_big",
            [
                make_group("uh896_s0", hash_bins(u, 896, 0), 896, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash2_big",
            [
                make_group("uh448_s0", hash_bins(u, 448, 0), 448, 20.0),
                make_group("uh448_s1", hash_bins(u, 448, 1), 448, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash3_300",
            [
                make_group("uh300_s0", hash_bins(u, 300, 0), 300, 20.0),
                make_group("uh300_s1", hash_bins(u, 300, 1), 300, 20.0),
                make_group("uh300_s2", hash_bins(u, 300, 2), 300, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash4_224",
            [
                make_group("uh224_s0", hash_bins(u, 224, 0), 224, 20.0),
                make_group("uh224_s1", hash_bins(u, 224, 1), 224, 20.0),
                make_group("uh224_s2", hash_bins(u, 224, 2), 224, 20.0),
                make_group("uh224_s3", hash_bins(u, 224, 3), 224, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "user_hash5_180",
            [
                make_group("uh180_s0", hash_bins(u, 180, 0), 180, 20.0),
                make_group("uh180_s1", hash_bins(u, 180, 1), 180, 20.0),
                make_group("uh180_s2", hash_bins(u, 180, 2), 180, 20.0),
                make_group("uh180_s3", hash_bins(u, 180, 3), 180, 20.0),
                make_group("uh180_s4", hash_bins(u, 180, 4), 180, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "stat_hash_mix_a",
            [
                gp(("uid96", "uavg0q8"), 10.0),
                make_group("uh128_s0", hash_bins(u, 128, 0), 128, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "stat_hash_mix_b",
            [
                gp(("uid64", "uavg0q8"), 10.0),
                make_group("uh192_s0", hash_bins(u, 192, 0), 192, 20.0),
                make_group("uh192_s1", hash_bins(u, 192, 1), 192, 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "stat_hash_mix_c",
            [
                gp(("uid48", "uavg0q8"), 10.0),
                make_group("uh384_s0", hash_bins(u, 384, 0), 384, 20.0),
                g1("iid128", 10.0),
                g1("base64", 50.0),
                g1("uavg30q32", 80.0),
            ],
        )
    )
    configs.append(
        (
            "id_pair_hash",
            [
                g1("uid384", 20.0),
                g1("iid384", 10.0),
                make_group("ph192_s0", pair_hash_bins(u, i, 192, 0), 192, 30.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "hash_pair_mix",
            [
                make_group("uh256_s0", hash_bins(u, 256, 0), 256, 20.0),
                make_group("ih192_s0", hash_bins(i, 192, 10), 192, 10.0),
                make_group("ph256_s0", pair_hash_bins(u, i, 256, 0), 256, 30.0),
                make_group("ph256_s1", pair_hash_bins(u, i, 256, 1), 256, 30.0),
                g1("base32", 50.0),
            ],
        )
    )
    configs.append(
        (
            "stat_pair_hash",
            [
                gp(("uid64", "uavg0q8"), 10.0),
                make_group("ph256_s0", pair_hash_bins(u, i, 256, 0), 256, 30.0),
                g1("iid128", 10.0),
                g1("base64", 50.0),
            ],
        )
    )

    best_score = 99.0
    best_groups = None
    best_label = ""
    for label, groups in configs:
        total = sum(g.size for g in groups)
        if total > PARAM_BUDGET:
            print(f"skip {label} params {total}", flush=True)
            continue
        score, fitted = run_backfit(y, base, groups, label)
        if score < best_score:
            best_score = score
            best_groups = fitted
            best_label = label

    out_path = OUT_DIR / "small_param_additive_under1k.npz"
    save = {
        "best_rmse": np.array(best_score, dtype=np.float32),
        "label": np.array(best_label),
        "param_count": np.array(sum(g.size for g in best_groups), dtype=np.int32),
        "group_names": np.array([g.name for g in best_groups]),
        "group_sizes": np.array([g.size for g in best_groups], dtype=np.int32),
        "group_shrinks": np.array([g.shrink for g in best_groups], dtype=np.float32),
    }
    for idx, group in enumerate(best_groups):
        save[f"idx_{idx}"] = group.idx
        save[f"values_{idx}"] = group.values
    np.savez(out_path, **save)
    print(
        f"\nBEST {best_label} rmse {best_score:.9f} "
        f"params {sum(g.size for g in best_groups)} saved {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
