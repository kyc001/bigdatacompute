import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import train_generalized_head as common


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ResidualPair(torch.nn.Module):
    def __init__(self, k, hidden, rank):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.user_linear = torch.nn.Parameter(torch.zeros(k))
        self.item_linear = torch.nn.Parameter(torch.zeros(k))
        self.user_fc1 = torch.nn.Linear(k, hidden)
        self.item_fc1 = torch.nn.Linear(k, hidden)
        self.user_fc2 = torch.nn.Linear(hidden, 1)
        self.item_fc2 = torch.nn.Linear(hidden, 1)
        self.user_proj = torch.nn.Linear(k, rank, bias=False)
        self.item_proj = torch.nn.Linear(k, rank, bias=False)

        torch.nn.init.kaiming_uniform_(self.user_fc1.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.item_fc1.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.user_fc2.weight)
        torch.nn.init.zeros_(self.user_fc2.bias)
        torch.nn.init.zeros_(self.item_fc2.weight)
        torch.nn.init.zeros_(self.item_fc2.bias)
        torch.nn.init.normal_(self.user_proj.weight, std=0.001)
        torch.nn.init.normal_(self.item_proj.weight, std=0.001)

    def forward(self, pu, qi):
        user = pu.matmul(self.user_linear) + self.user_fc2(torch.relu(self.user_fc1(pu))).squeeze(1)
        item = qi.matmul(self.item_linear) + self.item_fc2(torch.relu(self.item_fc1(qi))).squeeze(1)
        pair = (self.user_proj(pu) * self.item_proj(qi)).sum(dim=1)
        return self.bias + user + item + pair


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    k = int(os.environ.get("TASK2_MAP_K", "512"))
    hidden = int(os.environ.get("TASK2_MAP_HIDDEN", "64"))
    rank = int(os.environ.get("TASK2_MAP_RANK", "32"))
    max_epochs = int(os.environ.get("TASK2_MAP_EPOCHS", "600"))
    eval_every = 5
    patience = 30
    min_delta = 1e-5

    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    u_all, i_all, y_np, base_np = common.build_base_prediction(P_m.shape, Q_m.shape)
    n = y_np.shape[0]
    print(f"config K={k} hidden={hidden} rank={rank} n={n}", flush=True)

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

    device = torch.device("cuda")
    Pt = torch.tensor(P, device=device)
    Qt = torch.tensor(Q, device=device)
    u_t = torch.tensor(u_all, device=device, dtype=torch.long)
    i_t = torch.tensor(i_all, device=device, dtype=torch.long)
    y_t = torch.tensor(y_np, device=device)
    base_t = torch.tensor(base_np, device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)
    model = ResidualPair(k, hidden, rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-7)

    def residual(ids):
        return model(Pt[u_t[ids]], Qt[i_t[ids]])

    def rmse_all():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 131072):
                ids = all_idx[start : start + 131072]
                pred = torch.clamp(base_t[ids] + residual(ids), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = rmse_all()
    best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    stale = 0
    print(f"initial_rmse {best:.9f}", flush=True)

    for epoch in range(1, max_epochs + 1):
        if epoch == 180:
            for group in opt.param_groups:
                group["lr"] = 1e-4
        if epoch == 360:
            for group in opt.param_groups:
                group["lr"] = 3e-5

        order = all_idx[torch.randperm(n, device=device)]
        loss_sum = 0.0
        batches = 0
        t0 = time.time()
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            pred = base_t[ids] + residual(ids)
            loss = torch.mean((pred - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item()
            batches += 1

        if epoch <= 5 or epoch % eval_every == 0:
            current = rmse_all()
            improved = current < best - min_delta
            if improved:
                best = current
                stale = 0
                best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
            else:
                stale += 1
            print(
                f"epoch {epoch} train {math.sqrt(loss_sum / batches):.9f} "
                f"rmse {current:.9f} best {best:.9f} stale {stale} sec {time.time() - t0:.2f}",
                flush=True,
            )
            if stale >= patience:
                print(f"early_stop epoch={epoch} best={best:.9f}", flush=True)
                break

    out_path = OUT_DIR / f"residual_pair_base_stats_k{k}_h{hidden}_r{rank}.npz"
    np.savez(
        out_path,
        K=np.array(k, dtype=np.int32),
        hidden=np.array(hidden, dtype=np.int32),
        rank=np.array(rank, dtype=np.int32),
        p_mu=p_mu,
        p_std=p_std,
        q_mu=q_mu,
        q_std=q_std,
        best_rmse=np.array(best, dtype=np.float32),
        **best_state,
    )
    print(f"saved {out_path} best {best:.9f}", flush=True)


if __name__ == "__main__":
    main()
