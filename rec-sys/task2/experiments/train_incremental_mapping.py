import math
import time
from pathlib import Path

import numpy as np
import torch

from train_mapping_mlp_item_base import item_component


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")


class SmallPair(torch.nn.Module):
    def __init__(self, k, rank):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.user_linear = torch.nn.Parameter(torch.zeros(k))
        self.item_linear = torch.nn.Parameter(torch.zeros(k))
        self.user_proj = torch.nn.Linear(k, rank, bias=False)
        self.item_proj = torch.nn.Linear(k, rank, bias=False)
        torch.nn.init.normal_(self.user_proj.weight, std=0.001)
        torch.nn.init.normal_(self.item_proj.weight, std=0.001)

    def forward(self, p, q):
        return self.bias + p.matmul(self.user_linear) + q.matmul(self.item_linear) + (
            self.user_proj(p) * self.item_proj(q)
        ).sum(dim=1)


def rmse_np(y, pred):
    pred = np.clip(pred, 0.5, 5.0)
    return math.sqrt(float(np.mean((y - pred) ** 2)))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    comp = item_component()
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)

    k = 1024
    P = np.asarray(P_m[:, :k], dtype=np.float32)
    Q = np.asarray(Q_m[:, :k], dtype=np.float32)
    p_mu = P.mean(axis=0).astype(np.float32)
    p_std = P.std(axis=0).astype(np.float32)
    q_mu = Q.mean(axis=0).astype(np.float32)
    q_std = Q.std(axis=0).astype(np.float32)
    p_std[p_std < 1e-6] = 1.0
    q_std[q_std < 1e-6] = 1.0
    P = (P - p_mu) / p_std
    Q = (Q - q_mu) / q_std

    train_u = inc[:, 0].astype(np.int64)
    train_i = inc[:, 1].astype(np.int64)
    train_y = inc[:, 2].astype(np.float32)
    train_base = comp[train_i].astype(np.float32)
    test_u = test[:, 0].astype(np.int64)
    test_i = test[:, 1].astype(np.int64)
    test_y = test[:, 2].astype(np.float32)
    test_base = comp[test_i].astype(np.float32)
    print("base inc", rmse_np(train_y, train_base), "base test", rmse_np(test_y, test_base), flush=True)

    device = torch.device("cuda")
    Pt = torch.tensor(P, device=device)
    Qt = torch.tensor(Q, device=device)
    tu = torch.tensor(train_u, device=device, dtype=torch.long)
    ti = torch.tensor(train_i, device=device, dtype=torch.long)
    ty = torch.tensor(train_y, device=device)
    tb = torch.tensor(train_base, device=device)
    eu = torch.tensor(test_u, device=device, dtype=torch.long)
    ei = torch.tensor(test_i, device=device, dtype=torch.long)
    ey = torch.tensor(test_y, device=device)
    eb = torch.tensor(test_base, device=device)
    train_idx = torch.arange(train_y.shape[0], device=device, dtype=torch.long)
    eval_idx = torch.arange(test_y.shape[0], device=device, dtype=torch.long)

    for rank in (0, 4, 8, 16, 32):
        model = SmallPair(k, rank).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-6)
        print(f"rank {rank}", flush=True)

        def eval_rmse():
            se = 0.0
            n = 0
            with torch.no_grad():
                for s in range(0, eval_idx.numel(), 131072):
                    ids = eval_idx[s : s + 131072]
                    pred = torch.clamp(eb[ids] + model(Pt[eu[ids]], Qt[ei[ids]]), 0.5, 5.0)
                    err = pred - ey[ids]
                    se += torch.sum(err * err).item()
                    n += ids.numel()
            return math.sqrt(se / n)

        best = eval_rmse()
        print("initial test", best, flush=True)
        for epoch in range(1, 81):
            if epoch == 35:
                for g in opt.param_groups:
                    g["lr"] = 0.00025
            if epoch == 60:
                for g in opt.param_groups:
                    g["lr"] = 0.00008
            order = train_idx[torch.randperm(train_idx.numel(), device=device)]
            t0 = time.time()
            loss_sum = 0.0
            batches = 0
            for s in range(0, order.numel(), 65536):
                ids = order[s : s + 65536]
                loss = torch.mean((tb[ids] + model(Pt[tu[ids]], Qt[ti[ids]]) - ty[ids]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_sum += loss.item()
                batches += 1
            if epoch <= 5 or epoch % 5 == 0:
                r = eval_rmse()
                best = min(best, r)
                print(
                    f"epoch {epoch} train {math.sqrt(loss_sum / batches):.9f} test {r:.9f} best {best:.9f} sec {time.time()-t0:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
