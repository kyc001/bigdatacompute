"""E1 稠密基线。

说明：
- 该文件只服务于 E1 对照实验，不作为最终主提交方案；
- 为了避免重复实现，图读取、Top-k 输出和 RSS 采样直接复用 `main.py`；
- 核心算法仍然是手写的稠密幂迭代，不调用任何现成 PageRank API。
"""

from __future__ import annotations

import argparse
import json
import time

from main import PeakRssSampler, dense_power_iteration, dump_top_k, load_graph, _normalize_dtype


def run_dense_baseline(
    data_path: str,
    out_path: str,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype_name: str = "float32",
    max_iter: int = 200,
) -> dict[str, object]:
    """运行一次 E1 稠密基线。

    时间复杂度：
    - 建图阶段：O(M + N)
    - 每轮迭代：O(N^2)

    空间复杂度：
    - 主要由稠密转移矩阵主导，为 O(N^2)
    - 因此该实现只用于建立和 CSR 版本的时间/内存对照
    """

    dtype = _normalize_dtype("float32" if dtype_name == "float32" else "float64")
    sampler = PeakRssSampler(interval=0.05)
    wall_start = time.perf_counter()
    wall_sec = 0.0
    n_nodes = 0
    n_edges = 0
    iters = 0
    delta = float("inf")
    top10_signature = ""

    sampler.start()
    try:
        row_ptr, col_idx, out_deg, n_nodes = load_graph(data_path)
        n_edges = int(col_idx.shape[0])
        ranks, iters, delta = dense_power_iteration(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )
        top10_signature = dump_top_k(out_path, ranks, k=100)
        wall_sec = time.perf_counter() - wall_start
    finally:
        sampler.stop()

    return {
        "peak_rss_mb": float(sampler.peak_rss / (1024 * 1024)),
        "wall_sec": float(wall_sec),
        "iters": int(iters),
        "mode": "dense",
        "dtype": dtype_name,
        "delta": float(delta),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "top10_signature": top10_signature,
    }


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Dense PageRank baseline for E1")
    parser.add_argument("--data", default="Data.txt", help="输入 Data.txt 路径")
    parser.add_argument("--beta", type=float, default=0.85, help="阻尼系数")
    parser.add_argument("--eps", type=float, default=1e-8, help="收敛阈值")
    parser.add_argument("--out", default="Res_dense.txt", help="输出结果路径")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    summary = run_dense_baseline(
        data_path=args.data,
        out_path=args.out,
        beta=args.beta,
        eps=args.eps,
        dtype_name=args.dtype,
        max_iter=args.max_iter,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
