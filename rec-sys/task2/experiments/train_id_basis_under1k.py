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

    u = values["uid"].astype(np.int64)
    i = values["iid"].astype(np.int64)
    users = P_m.shape[0]
    items = Q_m.shape[0]

    def group(name, idx, size, shrink):
        return add.make_group(name, idx.astype(np.int32), size, shrink)

    def uid_div(n):
        return group(f"uid_div{n}", u * n // users, n, 20.0)

    def uid_mod(n):
        return group(f"uid_mod{n}", u % n, n, 20.0)

    def iid_div(n):
        return group(f"iid_div{n}", i * n // items, n, 10.0)

    def iid_mod(n):
        return group(f"iid_mod{n}", i % n, n, 10.0)

    def g1(name, shrink=30.0):
        idx, size = specs[name]
        return add.make_group(name, idx, size, shrink)

    configs = [
        ("mod896", [uid_mod(896), iid_div(64), g1("base32", 50.0)]),
        ("mod768_item", [uid_mod(768), iid_div(128), g1("base64", 50.0), g1("uavg30q32", 80.0)]),
        ("div_mod_512_384", [uid_div(384), uid_mod(512), iid_div(64), g1("base32", 50.0)]),
        ("div_mod_384_512", [uid_div(512), uid_mod(384), iid_div(64), g1("base32", 50.0)]),
        ("div_mod_512_256_item", [uid_div(512), uid_mod(256), iid_div(128), g1("base64", 50.0), g1("uavg30q32", 80.0)]),
        ("div_mod_448_384", [uid_div(448), uid_mod(384), iid_div(96), g1("base64", 50.0)]),
        ("div_mod_384_384_item", [uid_div(384), uid_mod(384), iid_div(128), g1("base64", 50.0), g1("iavg8q32", 50.0)]),
        ("triple_mod", [uid_mod(257), uid_mod(263), uid_mod(269), iid_div(128), g1("base64", 50.0)]),
        ("mod_plus_hash", [uid_mod(512), add.make_group("uh256_s0", add.hash_bins(u.astype(np.int32), 256, 0), 256, 20.0), iid_div(128), g1("base64", 50.0)]),
        ("div_plus_hash", [uid_div(512), add.make_group("uh256_s0", add.hash_bins(u.astype(np.int32), 256, 0), 256, 20.0), iid_div(128), g1("base64", 50.0)]),
        ("item_mod_mix", [uid_mod(512), uid_div(256), iid_mod(128), iid_div(64), g1("base32", 50.0)]),
        ("item_two_basis", [uid_mod(512), iid_mod(192), iid_div(192), g1("base64", 50.0), g1("uavg30q32", 80.0)]),
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

    out_path = OUT_DIR / "id_basis_under1k.npz"
    save = {
        "best_rmse": np.array(best_score, dtype=np.float32),
        "label": np.array(best_label),
        "param_count": np.array(sum(g.size for g in best_groups), dtype=np.int32),
        "group_names": np.array([g.name for g in best_groups]),
        "group_sizes": np.array([g.size for g in best_groups], dtype=np.int32),
        "group_shrinks": np.array([g.shrink for g in best_groups], dtype=np.float32),
    }
    for idx, group_obj in enumerate(best_groups):
        save[f"values_{idx}"] = group_obj.values
    np.savez(out_path, **save)
    print(
        f"\nBEST id_basis {best_label} rmse {best_score:.9f} "
        f"params {sum(g.size for g in best_groups)} saved {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
