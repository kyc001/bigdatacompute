import math
import time
from pathlib import Path

import numpy as np
import torch

import train_generalized_head as common


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT_DIR = Path("rec-sys/task2/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PolySmall(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.wp = torch.nn.Parameter(torch.zeros(k))
        self.wq = torch.nn.Parameter(torch.zeros(k))
        self.wpq = torch.nn.Parameter(torch.zeros(k))
        self.wp2 = torch.nn.Parameter(torch.zeros(k))
        self.wq2 = torch.nn.Parameter(torch.zeros(k))
        self.wabs = torch.nn.Parameter(torch.zeros(k))

    def forward(self, p, q):
        pq = p * q
        return (
            self.bias
            + p.matmul(self.wp)
            + q.matmul(self.wq)
            + pq.matmul(self.wpq)
            + (p * p - 1.0).matmul(self.wp2)
            + (q * q - 1.0).matmul(self.wq2)
            + torch.abs(pq).matmul(self.wabs)
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    k = 768
    print(f"torch {torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)
    P_m = np.load(ROOT / "P.npy", mmap_mode="r")
    Q_m = np.load(ROOT / "Q.npy", mmap_mode="r")
    u_all, i_all, y_np, base_np = common.build_base_prediction(P_m.shape, Q_m.shape)
    n = y_np.shape[0]
    print(f"config K={k} params={1 + 6 * k} n={n}", flush=True)

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
    model = PolySmall(k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-8)
    lr = 5e-4
    min_lr = 1e-6
    eval_every = 5
    patience = 25
    min_delta = 1e-5
    stale = 0

    def rmse_all():
        se = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, n, 131072):
                ids = all_idx[start : start + 131072]
                pred = torch.clamp(base_t[ids] + model(Pt[u_t[ids]], Qt[i_t[ids]]), 0.5, 5.0)
                err = pred - y_t[ids]
                se += torch.sum(err * err).item()
                count += ids.numel()
        return math.sqrt(se / count)

    best = rmse_all()
    best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    print(f"initial {best:.9f}", flush=True)

    epoch = 0
    while True:
        epoch += 1
        order = all_idx[torch.randperm(n, device=device)]
        loss_sum = 0.0
        batches = 0
        t0 = time.time()
        for start in range(0, n, 65536):
            ids = order[start : start + 65536]
            loss = torch.mean((base_t[ids] + model(Pt[u_t[ids]], Qt[i_t[ids]]) - y_t[ids]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            loss_sum += loss.item()
            batches += 1
        if epoch <= 5 or epoch % eval_every == 0:
            current = rmse_all()
            if current < best - min_delta:
                best = current
                stale = 0
                best_state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
            else:
                stale += 1
            print(
                f"epoch {epoch} train {math.sqrt(loss_sum / batches):.9f} "
                f"rmse {current:.9f} best {best:.9f} stale {stale} lr {lr:.2g} sec {time.time()-t0:.2f}",
                flush=True,
            )
            if stale >= patience:
                if lr > min_lr:
                    lr *= 0.3
                    for group in opt.param_groups:
                        group["lr"] = lr
                    stale = 0
                    print(f"reduce_lr {lr:.3g}", flush=True)
                else:
                    break

    out_path = OUT_DIR / "poly_small_k768.npz"
    np.savez(
        out_path,
        K=np.array(k, dtype=np.int32),
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
