"""main.py

课程作业主入口：
- 提供 `load_graph / power_iteration / dump_top_k` 的兼容实现；
- 默认主提交模式为 `csr_block`；
- 标准输出最后一行固定为 JSON，供 benchmark.py / sweep.py 解析。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time

import numpy as np
import psutil

from blocks import build_blocks, iter_edges_from_csr, iterate_by_block


def load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """读取边表并转换为 CSR。

    这里保留一个本地兼容实现，A 的真实版本可在保持签名不变的前提下替换。
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as file_obj:
        text = file_obj.read().strip()

    if not text:
        raise ValueError("输入图为空")

    flat = np.fromstring(text, sep=" ", dtype=np.int64)
    if flat.size == 0 or flat.size % 2 != 0:
        raise ValueError("输入文件格式错误：整数个数必须为 2 的倍数")

    edges = flat.reshape(-1, 2)
    src = edges[:, 0].astype(np.int32, copy=False)
    dst = edges[:, 1].astype(np.int32, copy=False)

    if np.any(src < 0) or np.any(dst < 0):
        raise ValueError("节点编号必须为非负整数")

    n_nodes = int(max(int(src.max()), int(dst.max())) + 1)
    order = np.lexsort((dst, src))
    src_sorted = src[order]
    dst_sorted = dst[order].astype(np.int32, copy=True)

    out_deg = np.bincount(src_sorted, minlength=n_nodes).astype(np.int32, copy=False)
    row_ptr = np.empty(n_nodes + 1, dtype=np.int32)
    row_ptr[0] = 0
    np.cumsum(out_deg, out=row_ptr[1:])
    return row_ptr, dst_sorted, out_deg, n_nodes


def power_iteration(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    N: int,
    beta: float,
    eps: float,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
) -> tuple[np.ndarray, int, float]:
    """纯 CSR 幂迭代。

    dead-end 处理严格使用 INTERFACE.md 冻结的质量补偿公式。
    """

    if N <= 0:
        raise ValueError("N 必须为正整数")
    if row_ptr.shape[0] != N + 1 or out_deg.shape[0] != N:
        raise ValueError("CSR 数组长度不匹配")
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype 仅支持 np.float32 或 np.float64")

    r = np.full(N, 1.0 / N, dtype=dtype)
    r_new = np.empty(N, dtype=dtype)
    live_mask = out_deg > 0
    dead_mask = ~live_mask
    inv_out_deg = np.zeros(N, dtype=dtype)
    inv_out_deg[live_mask] = 1.0 / out_deg[live_mask]

    for num_iter in range(1, max_iter + 1):
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype((1.0 - beta) / N + beta * dangling_mass / N)
        r_new.fill(base)

        for src in range(N):
            degree = int(out_deg[src])
            if degree == 0:
                continue
            contrib = beta * r[src] * inv_out_deg[src]
            start = int(row_ptr[src])
            stop = int(row_ptr[src + 1])
            np.add.at(r_new, col_idx[start:stop], contrib)

        delta = float(np.sum(np.abs(r_new - r), dtype=np.float64))
        if delta < eps:
            return r_new.astype(dtype, copy=True), num_iter, delta

        r, r_new = r_new, r

    return r.astype(dtype, copy=True), max_iter, delta


def dense_power_iteration(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    N: int,
    beta: float,
    eps: float,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
    max_dense_nodes: int = 2000,
) -> tuple[np.ndarray, int, float]:
    """小图 dense 版本，只用于 mock / 对照实验。"""

    if N > max_dense_nodes:
        raise ValueError("dense 模式只保留给小图与对照实验，不用于大图主提交流程")

    transition = np.zeros((N, N), dtype=dtype)
    for src in range(N):
        degree = int(out_deg[src])
        if degree == 0:
            continue
        start = int(row_ptr[src])
        stop = int(row_ptr[src + 1])
        transition[src, col_idx[start:stop]] = dtype(1.0 / degree)

    r = np.full(N, 1.0 / N, dtype=dtype)
    r_new = np.empty(N, dtype=dtype)
    temp = np.empty(N, dtype=dtype)
    dead_mask = out_deg == 0

    for num_iter in range(1, max_iter + 1):
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype((1.0 - beta) / N + beta * dangling_mass / N)
        np.dot(transition.T, r, out=temp)
        temp *= beta
        r_new[:] = temp
        r_new += base

        delta = float(np.sum(np.abs(r_new - r), dtype=np.float64))
        if delta < eps:
            return r_new.astype(dtype, copy=True), num_iter, delta

        r, r_new = r_new, r

    return r.astype(dtype, copy=True), max_iter, delta


def dump_top_k(path: str, ranks: np.ndarray, k: int = 100) -> str:
    """输出 Top-k 结果，并返回 top10_signature。"""

    total = ranks.shape[0]
    top_k = min(k, total)
    node_ids = np.arange(total, dtype=np.int32)
    order = np.lexsort((node_ids, -ranks))
    top_idx = order[:top_k]

    with open(path, "w", encoding="utf-8") as file_obj:
        for node_id in top_idx:
            file_obj.write(f"{int(node_id)} {float(ranks[node_id]):.10f}\n")

    signature = ",".join(str(int(node_id)) for node_id in top_idx[:10])
    return signature


class PeakRssSampler:
    """在主进程内部做轻量 RSS 采样，供 stdout JSON 回报。"""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = int(self.process.memory_info().rss)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))
            except psutil.Error:
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))
        except psutil.Error:
            pass


def run_selected_mode(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    n_nodes: int,
    mode: str,
    K: int,
    beta: float,
    eps: float,
    dtype: np.dtype,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """按 mode 执行对应版本。"""

    if mode == "dense":
        return dense_power_iteration(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )

    if mode == "csr":
        return power_iteration(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            beta=beta,
            eps=eps,
            dtype=dtype,
            max_iter=max_iter,
        )

    if mode in ("block", "csr_block"):
        with tempfile.TemporaryDirectory(prefix="pagerank_blocks_") as tmp_dir:
            block_meta = build_blocks(iter_edges_from_csr(row_ptr, col_idx), K, tmp_dir)
            return iterate_by_block(
                block_meta,
                out_deg,
                n_nodes,
                beta=beta,
                eps=eps,
                dtype=dtype,
                max_iter=max_iter,
                tmp_dir=tmp_dir,
            )

    raise ValueError("mode 仅支持 dense / csr / block / csr_block")


def run_pagerank(
    data_path: str,
    out_path: str,
    mode: str = "csr_block",
    K: int = 8,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype_name: str = "float32",
    max_iter: int = 200,
) -> dict:
    """执行一次 PageRank，并返回可 JSON 序列化的摘要。"""

    dtype = np.float32 if dtype_name == "float32" else np.float64
    sampler = PeakRssSampler(interval=0.05)
    sampler.start()
    wall_start = time.perf_counter()
    try:
        row_ptr, col_idx, out_deg, n_nodes = load_graph(data_path)
        ranks, iters, delta = run_selected_mode(
            row_ptr,
            col_idx,
            out_deg,
            n_nodes,
            mode=mode,
            K=K,
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
        "mode": mode,
        "K": int(K),
        "dtype": dtype_name,
        "delta": float(delta),
        "n_nodes": int(n_nodes),
        "n_edges": int(col_idx.shape[0]),
        "top10_signature": top10_signature,
        "out": out_path,
    }


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="PageRank main entry")
    parser.add_argument("--data", default="Data.txt")
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--out", default="Res.txt")
    parser.add_argument("--mode", choices=("dense", "csr", "block", "csr_block"), default="csr_block")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    summary = run_pagerank(
        data_path=args.data,
        out_path=args.out,
        mode=args.mode,
        K=args.K,
        beta=args.beta,
        eps=args.eps,
        dtype_name=args.dtype,
        max_iter=args.max_iter,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
