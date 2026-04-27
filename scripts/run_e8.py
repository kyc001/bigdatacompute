"""E8：dead-end 处理策略对比实验。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as pagerank_main


def top10_signature(ranks: np.ndarray, node_ids: np.ndarray) -> tuple[str, np.ndarray]:
    """生成 Top-10 签名与节点列表。"""

    order = np.lexsort((node_ids, -ranks))[:10]
    top10_ids = node_ids[order]
    signature = ",".join(str(int(node_id)) for node_id in top10_ids)
    return signature, top10_ids


def jaccard_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个 Top-10 集合的 Jaccard 相似度。"""

    left_set = set(map(int, left.tolist()))
    right_set = set(map(int, right.tolist()))
    return len(left_set & right_set) / len(left_set | right_set)


def pagerank_ignore_dead_ends(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    n_nodes: int,
    beta: float,
    eps: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """忽略 dead-end 的对照实现。"""

    r = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float32)
    r_new = np.empty(n_nodes, dtype=np.float32)
    delta_buffer = np.empty(n_nodes, dtype=np.float32)
    live_mask = out_deg > 0
    live_nodes = np.flatnonzero(live_mask).astype(np.int32, copy=False)
    inv_out_deg = np.zeros(n_nodes, dtype=np.float32)
    inv_out_deg[live_mask] = np.float32(1.0) / out_deg[live_mask]
    base = np.float32((1.0 - beta) / n_nodes)
    beta_cast = np.float32(beta)
    last_delta = float("inf")

    for num_iter in range(1, max_iter + 1):
        r_new.fill(base)
        for src in live_nodes:
            src_index = int(src)
            contrib = beta_cast * r[src_index] * inv_out_deg[src_index]
            start = int(row_ptr[src_index])
            stop = int(row_ptr[src_index + 1])
            np.add.at(r_new, col_idx[start:stop], contrib)

        delta_buffer[:] = r_new
        delta_buffer -= r
        np.abs(delta_buffer, out=delta_buffer)
        last_delta = float(np.sum(delta_buffer, dtype=np.float64))
        if last_delta < eps:
            return r_new.copy(), num_iter, last_delta
        r, r_new = r_new, r

    return r.copy(), max_iter, last_delta


def prune_dead_ends(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """递归删除 dead-end，返回保留节点和压缩后的 CSR。"""

    active = np.ones(n_nodes, dtype=bool)
    while True:
        active_out_deg = np.zeros(n_nodes, dtype=np.int32)
        for src in range(n_nodes):
            if not active[src]:
                continue
            start = int(row_ptr[src])
            stop = int(row_ptr[src + 1])
            count = 0
            for offset in range(start, stop):
                if active[int(col_idx[offset])]:
                    count += 1
            active_out_deg[src] = count

        dead_nodes = active & (active_out_deg == 0)
        if not np.any(dead_nodes):
            break
        active[dead_nodes] = False

    active_nodes = np.flatnonzero(active).astype(np.int32, copy=False)
    mapping = np.full(n_nodes, -1, dtype=np.int32)
    mapping[active_nodes] = np.arange(active_nodes.shape[0], dtype=np.int32)

    src_list: list[int] = []
    dst_list: list[int] = []
    for src in active_nodes:
        src_index = int(src)
        mapped_src = int(mapping[src_index])
        start = int(row_ptr[src_index])
        stop = int(row_ptr[src_index + 1])
        for offset in range(start, stop):
            dst = int(col_idx[offset])
            if active[dst]:
                src_list.append(mapped_src)
                dst_list.append(int(mapping[dst]))

    reduced_n = int(active_nodes.shape[0])
    reduced_out_deg = np.zeros(reduced_n, dtype=np.int32)
    for src in src_list:
        reduced_out_deg[src] += 1

    reduced_row_ptr = np.empty(reduced_n + 1, dtype=np.int32)
    reduced_row_ptr[0] = 0
    cursor = 0
    for node in range(reduced_n):
        cursor += int(reduced_out_deg[node])
        reduced_row_ptr[node + 1] = cursor

    reduced_col_idx = np.empty(len(dst_list), dtype=np.int32)
    write_ptr = reduced_row_ptr[:-1].copy()
    for src, dst in zip(src_list, dst_list):
        position = int(write_ptr[src])
        reduced_col_idx[position] = dst
        write_ptr[src] = position + 1

    return active_nodes, reduced_row_ptr, reduced_col_idx, reduced_out_deg


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """写出 E8 CSV。"""

    fieldnames = [
        "strategy",
        "wall_sec",
        "iters",
        "delta",
        "rank_sum",
        "active_nodes",
        "removed_nodes",
        "top10_signature",
        "jaccard_vs_compensation",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run E8 dead-end strategy comparison")
    parser.add_argument("--data", default="Data.txt")
    parser.add_argument("--out", default="experiments/E8.csv")
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    row_ptr, col_idx, out_deg, n_nodes = pagerank_main.load_graph(args.data)
    node_ids = getattr(pagerank_main, "_LAST_NODE_IDS", None)
    if node_ids is None or int(node_ids.shape[0]) != n_nodes:
        node_ids = np.arange(n_nodes, dtype=np.int64)

    rows: list[dict[str, object]] = []

    start = time.perf_counter()
    compensation_ranks, compensation_iters, compensation_delta = pagerank_main.power_iteration(
        row_ptr,
        col_idx,
        out_deg,
        n_nodes,
        beta=args.beta,
        eps=args.eps,
        max_iter=args.max_iter,
    )
    compensation_wall = time.perf_counter() - start
    compensation_signature, compensation_top10 = top10_signature(compensation_ranks, node_ids)
    rows.append(
        {
            "strategy": "compensation",
            "wall_sec": compensation_wall,
            "iters": compensation_iters,
            "delta": compensation_delta,
            "rank_sum": float(np.sum(compensation_ranks, dtype=np.float64)),
            "active_nodes": n_nodes,
            "removed_nodes": 0,
            "top10_signature": compensation_signature,
            "jaccard_vs_compensation": 1.0,
        }
    )

    start = time.perf_counter()
    ignore_ranks, ignore_iters, ignore_delta = pagerank_ignore_dead_ends(
        row_ptr,
        col_idx,
        out_deg,
        n_nodes,
        beta=args.beta,
        eps=args.eps,
        max_iter=args.max_iter,
    )
    ignore_wall = time.perf_counter() - start
    ignore_signature, ignore_top10 = top10_signature(ignore_ranks, node_ids)
    rows.append(
        {
            "strategy": "ignore",
            "wall_sec": ignore_wall,
            "iters": ignore_iters,
            "delta": ignore_delta,
            "rank_sum": float(np.sum(ignore_ranks, dtype=np.float64)),
            "active_nodes": n_nodes,
            "removed_nodes": 0,
            "top10_signature": ignore_signature,
            "jaccard_vs_compensation": jaccard_similarity(compensation_top10, ignore_top10),
        }
    )

    start = time.perf_counter()
    active_nodes, reduced_row_ptr, reduced_col_idx, reduced_out_deg = prune_dead_ends(row_ptr, col_idx, n_nodes)
    if active_nodes.size == 0:
        delete_ranks = np.zeros(n_nodes, dtype=np.float32)
        delete_iters = 0
        delete_delta = 0.0
    else:
        reduced_ranks, delete_iters, delete_delta = pagerank_main.power_iteration(
            reduced_row_ptr,
            reduced_col_idx,
            reduced_out_deg,
            int(active_nodes.shape[0]),
            beta=args.beta,
            eps=args.eps,
            max_iter=args.max_iter,
        )
        delete_ranks = np.zeros(n_nodes, dtype=np.float32)
        delete_ranks[active_nodes] = reduced_ranks
        total = float(np.sum(delete_ranks, dtype=np.float64))
        if total > 0.0:
            delete_ranks /= np.float32(total)
    delete_wall = time.perf_counter() - start
    delete_signature, delete_top10 = top10_signature(delete_ranks, node_ids)
    rows.append(
        {
            "strategy": "delete",
            "wall_sec": delete_wall,
            "iters": delete_iters,
            "delta": delete_delta,
            "rank_sum": float(np.sum(delete_ranks, dtype=np.float64)),
            "active_nodes": int(active_nodes.shape[0]),
            "removed_nodes": int(n_nodes - active_nodes.shape[0]),
            "top10_signature": delete_signature,
            "jaccard_vs_compensation": jaccard_similarity(compensation_top10, delete_top10),
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out_path, rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
