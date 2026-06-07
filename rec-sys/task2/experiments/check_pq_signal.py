import math
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")


def main():
    p = np.load(ROOT / "P.npy", mmap_mode="r")
    q = np.load(ROOT / "Q.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))

    u = torch.tensor(test[:, 0].astype(np.int64), device="cuda")
    i = torch.tensor(test[:, 1].astype(np.int64), device="cuda")
    y = torch.tensor(test[:, 2].astype(np.float32), device="cuda")
    pt = torch.tensor(np.asarray(p), device="cuda")
    qt = torch.tensor(np.asarray(q), device="cuda")

    for k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        t0 = time.time()
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, y.numel(), 65536):
                sl = slice(start, min(start + 65536, y.numel()))
                if k == 0:
                    pred = torch.full((u[sl].numel(),), mean, device="cuda")
                else:
                    pred = mean + (pt[u[sl], :k] * qt[i[sl], :k]).sum(dim=1)
                err = torch.clamp(pred, 0.5, 5.0) - y[sl]
                se += torch.sum(err * err).item()
                count += err.numel()
        print(f"K {k:4d} rmse {math.sqrt(se / count):.9f} sec {time.time() - t0:.2f}", flush=True)


if __name__ == "__main__":
    main()
