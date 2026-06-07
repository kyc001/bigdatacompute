import sys
from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
import train_generalized_head as common  # noqa: E402
import train_small_param_additive as add  # noqa: E402
import train_small_param_tables as tables  # noqa: E402


PARAM_BUDGET = 999


def main():
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    _, _, y, base = common.build_base_prediction(P_m.shape, Q_m.shape)
    values = tables.build_stats_features(P_m.shape[0], Q_m.shape[0])
    values["base"] = np.clip(base, 0.5, 5.0).astype(np.float32)
    specs = tables.build_bins(values)

    users = P_m.shape[0]
    items = Q_m.shape[0]
    uid = values["uid"].astype(np.int64)
    iid = values["iid"].astype(np.int64)
    for n in (2, 3, 4, 5, 6, 7, 10, 12, 15, 20, 24, 28, 30, 40):
        specs[f"uid{n}"] = ((uid * n // users).astype(np.int32), n)
        specs[f"iid{n}"] = ((iid * n // items).astype(np.int32), n)
    for key in ("uavg0", "uavg5", "uavg30", "iavg0", "iavg3", "iavg8", "base", "log_uc", "log_ic"):
        for n in (2, 3, 4, 5, 6, 7, 10, 12):
            specs[f"{key}q{n}"] = (tables.rank_bins(values[key], n), n)

    def g1(name, shrink=30.0):
        idx, size = specs[name]
        return add.make_group(name, idx, size, shrink)

    def gp(names, shrink=50.0):
        idx, size = add.combined(specs, names)
        return add.make_group("x".join(names), idx, size, shrink)

    configs = [
        (
            "u32_ub8_i3",
            [gp(("uid32", "uavg0q8", "iid3"), 20.0), g1("iid128", 10.0), g1("base64", 50.0), g1("uavg30q32", 80.0)],
        ),
        (
            "u24_ub8_i5",
            [gp(("uid24", "uavg0q8", "iid5"), 20.0), g1("base32", 50.0)],
        ),
        (
            "u20_ub8_i6",
            [gp(("uid20", "uavg0q8", "iid6"), 20.0), g1("base32", 50.0)],
        ),
        (
            "u16_ub8_i7",
            [gp(("uid16", "uavg0q8", "iid7"), 20.0), g1("base64", 50.0), g1("iavg8q32", 50.0)],
        ),
        (
            "u12_ub16_i5",
            [gp(("uid12", "uavg0q16", "iid5"), 20.0), g1("base32", 50.0)],
        ),
        (
            "u32_ub4_i7",
            [gp(("uid32", "uavg0q4", "iid7"), 20.0), g1("base64", 50.0), g1("iavg8q32", 50.0)],
        ),
        (
            "u48_ub4_i4",
            [gp(("uid48", "uavg0q4", "iid4"), 20.0), g1("iid128", 10.0), g1("base64", 50.0), g1("uavg30q32", 80.0)],
        ),
        (
            "u32_ub8_ib3",
            [gp(("uid32", "uavg0q8", "iavg0q3"), 20.0), g1("iid128", 10.0), g1("base64", 50.0), g1("uavg30q32", 80.0)],
        ),
        (
            "u24_ub8_ib5",
            [gp(("uid24", "uavg0q8", "iavg0q5"), 20.0), g1("base32", 50.0)],
        ),
        (
            "u16_ub8_ib7",
            [gp(("uid16", "uavg0q8", "iavg0q7"), 20.0), g1("iid64", 10.0), g1("base32", 50.0)],
        ),
        (
            "hash_user_stat_item",
            [
                add.make_group("uh512_s0", add.hash_bins(values["uid"].astype(np.int32), 512, 0), 512, 20.0),
                gp(("uid24", "uavg0q8", "iid2"), 20.0),
                g1("iid64", 10.0),
                g1("base32", 50.0),
                g1("uavg30q32", 80.0),
            ],
        ),
    ]

    best_score = 99.0
    best_groups = None
    best_label = ""
    for label, groups in configs:
        params = sum(g.size for g in groups)
        if params > PARAM_BUDGET:
            print(f"skip {label} params {params}", flush=True)
            continue
        score, fitted = add.run_backfit(y, base, groups, label)
        if score < best_score:
            best_score = score
            best_groups = fitted
            best_label = label

    out_path = OUT_DIR / "small_param_3d_under1k.npz"
    save = {
        "best_rmse": np.array(best_score, dtype=np.float32),
        "label": np.array(best_label),
        "param_count": np.array(sum(g.size for g in best_groups), dtype=np.int32),
        "group_names": np.array([g.name for g in best_groups]),
        "group_sizes": np.array([g.size for g in best_groups], dtype=np.int32),
        "group_shrinks": np.array([g.shrink for g in best_groups], dtype=np.float32),
    }
    for idx, group in enumerate(best_groups):
        save[f"values_{idx}"] = group.values
    np.savez(out_path, **save)
    print(
        f"\nBEST 3d {best_label} rmse {best_score:.9f} "
        f"params {sum(g.size for g in best_groups)} saved {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
