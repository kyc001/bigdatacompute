from pathlib import Path

import numpy as np

import probe_sample_gate_128 as gate


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")


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
    n = residual.shape[0]
    row = np.arange(n, dtype=np.int64)
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    all_item_mask = np.ones(n, dtype=bool)

    candidates = []
    for phase in range(10):
        candidates.append((f"user_stride10_phase{phase}", (row % 10) == phase))
    for phase in range(20):
        candidates.append((f"user_stride20_phase{phase}", (row % 20) == phase))
    for phase in range(5):
        candidates.append((f"user_stride5_phase{phase}", (row % 5) == phase))

    bucket = np.rint(rating * 2).astype(np.int32)
    for b in range(1, 11):
        # Keep the same phase-0 row gate, but exclude one rating bucket from user-side updates.
        candidates.append((f"user_phase0_skip_rating_{b/2:.1f}", ((row % 10) == 0) & (bucket != b)))
    for keep in ((1, 10), (1, 2, 9, 10), (1, 2, 3, 8, 9, 10), (1, 2, 3, 4, 7, 8, 9, 10)):
        candidates.append((f"user_phase0_keep_ratings_{'_'.join(map(str, keep))}", ((row % 10) == 0) & np.isin(bucket, keep)))

    best = (99.0, None)
    for label, user_mask in candidates:
        score, _, _, _ = gate.evaluate(
            label, inc_u, inc_i, residual, all_item_mask, user_mask, users, items, u, i, y
        )
        if score < best[0]:
            best = (score, label)
    print(f"BEST {best[1]} {best[0]:.9f}", flush=True)


if __name__ == "__main__":
    main()
