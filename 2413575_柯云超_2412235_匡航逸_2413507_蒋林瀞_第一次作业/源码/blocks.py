"""blocks.py

分块 PageRank 的核心实现，供 B 角色独立开发与后续集成。
"""

from __future__ import annotations

import gc
import os
import struct
import tempfile
from typing import Iterable, Iterator

import numpy as np


# 每条边记录 8 字节：src(int32) + local_dst(int32)。
BLOCK_RECORD_DTYPE = np.dtype([("src", np.int32), ("local_dst", np.int32)])
BLOCK_RECORD_STRUCT = struct.Struct("<ii")


def iter_edges_from_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> tuple[Iterator[tuple[int, int]], int]:
    """把 CSR 视图转换为边流，供 build_blocks 直接消费。

    时间复杂度：O(M)
    空间复杂度：O(1) 额外空间（不把整张图重新展开到内存）
    """

    if row_ptr.ndim != 1 or col_idx.ndim != 1:
        raise ValueError("row_ptr 与 col_idx 必须是一维数组")
    if row_ptr.size == 0:
        raise ValueError("row_ptr 不能为空")

    n_nodes = int(row_ptr.size - 1)

    def edge_iter() -> Iterator[tuple[int, int]]:
        for src in range(n_nodes):
            start = int(row_ptr[src])
            stop = int(row_ptr[src + 1])
            for offset in range(start, stop):
                yield src, int(col_idx[offset])

    return edge_iter(), n_nodes


def _normalize_edges_input(
    edges,
) -> tuple[Iterable[tuple[int, int]], int]:
    """统一解析 build_blocks 的输入。

    支持：
    - `(edge_iterable, N)`：推荐，用于主流程；
    - `np.ndarray(M, 2)`：用于 mock / tests。
    """

    if isinstance(edges, tuple) and len(edges) == 2:
        edge_iterable, n_nodes = edges
        n_nodes = int(n_nodes)
        if n_nodes <= 0:
            raise ValueError("N 必须为正整数")
        return edge_iterable, n_nodes

    if isinstance(edges, np.ndarray):
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges 数组必须是形如 (M, 2) 的二维数组")
        if edges.size == 0:
            raise ValueError("edges 不能为空")
        n_nodes = int(edges.max()) + 1
        return ((int(src), int(dst)) for src, dst in edges), n_nodes

    raise ValueError("edges 必须是 (edge_iterable, N) 或 np.ndarray(M, 2)")


def build_blocks(edges, K: int, tmp_dir: str) -> list[dict]:
    """按目标节点分桶，把边写入 K 个二进制临时文件。

    设计意图：
    - 用目标节点分块，而不是源节点分块；
    - 每个块单独落盘，后续可由 np.memmap 逐块读取；
    - 文件中存储的是 `(src, local_dst)`，避免迭代阶段重复做全局下标减法。

    时间复杂度：O(M)
    空间复杂度：O(K)，不依赖边数 M 的额外常驻内存
    """

    if K < 1:
        raise ValueError("K 必须大于等于 1")

    edge_iterable, n_nodes = _normalize_edges_input(edges)
    os.makedirs(tmp_dir, exist_ok=True)

    block_paths: list[str] = []
    block_files = []
    edge_counts = np.zeros(K, dtype=np.int32)
    block_starts = np.empty(K, dtype=np.int32)
    block_stops = np.empty(K, dtype=np.int32)
    node_to_block = np.empty(n_nodes, dtype=np.int32)

    for block_id in range(K):
        block_starts[block_id] = (block_id * n_nodes) // K
        block_stops[block_id] = ((block_id + 1) * n_nodes) // K
        node_to_block[block_starts[block_id] : block_stops[block_id]] = block_id

    for block_id in range(K):
        path = os.path.join(tmp_dir, f"block_{block_id:03d}.bin")
        block_paths.append(path)
        block_files.append(open(path, "wb"))

    try:
        for src, dst in edge_iterable:
            if src < 0 or dst < 0 or dst >= n_nodes or src >= n_nodes:
                raise ValueError("检测到非法节点编号，必须落在 [0, N) 区间")

            block_id = int(node_to_block[dst])
            node_start = int(block_starts[block_id])
            local_dst = dst - node_start
            block_files[block_id].write(BLOCK_RECORD_STRUCT.pack(src, local_dst))
            edge_counts[block_id] += 1
    finally:
        for file_obj in block_files:
            file_obj.close()

    metadata: list[dict] = []
    for block_id, path in enumerate(block_paths):
        node_start = int(block_starts[block_id])
        node_stop = int(block_stops[block_id])
        metadata.append(
            {
                "block_id": int(block_id),
                "path": path,
                "node_start": int(node_start),
                "node_stop": int(node_stop),
                "node_count": int(node_stop - node_start),
                "edge_count": int(edge_counts[block_id]),
            }
        )
    return metadata


def iterate_by_block(
    blocks: list[dict],
    out_deg: np.ndarray,
    N: int,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
    tmp_dir: str | None = None,
    chunk_size: int = 1 << 15,
) -> tuple[np.ndarray, int, float]:
    """使用磁盘映射和真分块执行 PageRank 幂迭代。

    核心策略：
    - 完整 `r` 与 `r_new` 都放到 memmap 文件中；
    - 每轮只取当前块的 `r_new[node_start:node_stop]` 视图做更新；
    - 块内边记录通过 memmap 顺序读取，避免把整张反向邻接表留在内存。

    时间复杂度：O(T * M)，T 为收敛轮数
    空间复杂度：O(N + max_block_edges_chunk)，其中 `r_new` 的活动工作集仅为当前块片段
    """

    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype 仅支持 np.float32 或 np.float64")
    if N <= 0:
        raise ValueError("N 必须为正整数")
    if out_deg.shape[0] != N:
        raise ValueError("out_deg 长度必须等于 N")
    if not blocks:
        raise ValueError("blocks 不能为空")

    own_tmp_dir = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="pagerank_rank_")
        own_tmp_dir = True
    else:
        os.makedirs(tmp_dir, exist_ok=True)

    rank_a_path = os.path.join(tmp_dir, f"rank_a_{dtype.__name__}.bin")
    rank_b_path = os.path.join(tmp_dir, f"rank_b_{dtype.__name__}.bin")

    live_mask = out_deg > 0
    dead_mask = ~live_mask
    inv_out_deg = np.zeros(N, dtype=dtype)
    inv_out_deg[live_mask] = 1.0 / out_deg[live_mask]
    max_block_nodes = max(int(block["node_count"]) for block in blocks)

    src_rank_buffer = np.empty(chunk_size, dtype=dtype)
    contrib_buffer = np.empty(chunk_size, dtype=dtype)
    delta_buffer = np.empty(max_block_nodes, dtype=dtype)

    r = None
    r_new = None
    try:
        r = np.memmap(rank_a_path, dtype=dtype, mode="w+", shape=(N,))
        r_new = np.memmap(rank_b_path, dtype=dtype, mode="w+", shape=(N,))
        r.fill(dtype(1.0 / N))

        last_delta = np.inf
        for num_iter in range(1, max_iter + 1):
            dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
            base = dtype((1.0 - beta) / N + beta * dangling_mass / N)
            l1_delta = 0.0

            for block in blocks:
                node_start = int(block["node_start"])
                node_stop = int(block["node_stop"])
                node_count = int(block["node_count"])
                edge_count = int(block["edge_count"])

                block_view = r_new[node_start:node_stop]
                block_view.fill(base)

                if edge_count > 0:
                    edge_map = np.memmap(
                        block["path"],
                        dtype=BLOCK_RECORD_DTYPE,
                        mode="r",
                        shape=(edge_count,),
                    )
                    src_idx = None
                    local_dst = None
                    try:
                        for offset in range(0, edge_count, chunk_size):
                            chunk_stop = min(offset + chunk_size, edge_count)
                            current = chunk_stop - offset
                            src_idx = edge_map["src"][offset:chunk_stop]
                            local_dst = edge_map["local_dst"][offset:chunk_stop]

                            # 先把 rank[src] 和 1/out_deg[src] 拉到固定缓冲区，再原地乘起来。
                            np.take(r, src_idx, out=src_rank_buffer[:current])
                            np.take(inv_out_deg, src_idx, out=contrib_buffer[:current])
                            np.multiply(
                                src_rank_buffer[:current],
                                contrib_buffer[:current],
                                out=contrib_buffer[:current],
                            )
                            contrib_buffer[:current] *= beta
                            np.add.at(block_view, local_dst, contrib_buffer[:current])
                    finally:
                        del src_idx
                        del local_dst
                        mmap_obj = getattr(edge_map, "_mmap", None)
                        if mmap_obj is not None:
                            mmap_obj.close()
                        del edge_map

                # 按块累计 L1 范数，避免一次性构造完整差值向量。
                block_delta = delta_buffer[:node_count]
                block_delta[:] = block_view
                block_delta -= r[node_start:node_stop]
                np.abs(block_delta, out=block_delta)
                l1_delta += float(np.sum(block_delta, dtype=np.float64))
                del block_delta
                del block_view

            last_delta = l1_delta
            if l1_delta < eps:
                result = np.array(r_new, dtype=dtype, copy=True)
                return result, num_iter, last_delta

            r, r_new = r_new, r

        result = np.array(r, dtype=dtype, copy=True)
        return result, max_iter, float(last_delta)
    finally:
        if r is not None:
            try:
                r.flush()
            except ValueError:
                pass
            mmap_obj = getattr(r, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        if r_new is not None:
            try:
                r_new.flush()
            except ValueError:
                pass
            mmap_obj = getattr(r_new, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        del r
        del r_new
        gc.collect()

        for path in (rank_a_path, rank_b_path):
            if os.path.exists(path):
                os.remove(path)
        if own_tmp_dir and os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
