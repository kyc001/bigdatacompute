"""mock_graph.py

mock 先行开发使用的图数据生成与参考实现：

1. 5 节点小图：用于把每轮 PageRank 数值写成手算断言；
2. 100 节点、500 边随机图：用于回归测试与分块结果对拍；
3. 提供朴素 O(N^2) dense reference，不依赖任何现成 PageRank API。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


SMALL_GRAPH_NODES = 5
RANDOM_GRAPH_NODES = 100
RANDOM_GRAPH_EDGES = 500
RANDOM_SEED = 20260420


@dataclass(frozen=True)
class GraphCase:
    """统一描述 mock 图数据。"""

    name: str
    edges: np.ndarray
    row_ptr: np.ndarray
    col_idx: np.ndarray
    out_deg: np.ndarray
    n_nodes: int


def edges_to_csr(edges: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把边表转换为 CSR。

    这里保留纯 numpy 实现，方便测试和 mock 场景独立运行。
    """

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges 必须是形如 (M, 2) 的二维数组")

    if edges.shape[0] == 0:
        row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
        col_idx = np.empty(0, dtype=np.int32)
        out_deg = np.zeros(n_nodes, dtype=np.int32)
        return row_ptr, col_idx, out_deg

    src = edges[:, 0].astype(np.int32, copy=False)
    dst = edges[:, 1].astype(np.int32, copy=False)
    order = np.lexsort((dst, src))
    src_sorted = src[order]
    dst_sorted = dst[order].astype(np.int32, copy=True)

    out_deg = np.bincount(src_sorted, minlength=n_nodes).astype(np.int32, copy=False)
    row_ptr = np.empty(n_nodes + 1, dtype=np.int32)
    row_ptr[0] = 0
    np.cumsum(out_deg, out=row_ptr[1:])
    return row_ptr, dst_sorted, out_deg


def build_small_handcalc_graph() -> GraphCase:
    """返回一个 5 节点小图。

    边集合：
    - 0 -> 1
    - 1 -> 2
    - 2 -> 0
    - 3 -> 2
    - 4 为 dead-end

    该图的前两轮结果可手工写出，测试中会直接断言：
    - r0 = [0.2, 0.2, 0.2, 0.2, 0.2]
    - r1 = [0.234, 0.234, 0.404, 0.064, 0.064]
    - r2 = [0.38428, 0.23978, 0.29418, 0.04088, 0.04088]
    """

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [3, 2],
        ],
        dtype=np.int32,
    )
    row_ptr, col_idx, out_deg = edges_to_csr(edges, SMALL_GRAPH_NODES)
    return GraphCase(
        name="small_handcalc",
        edges=edges,
        row_ptr=row_ptr,
        col_idx=col_idx,
        out_deg=out_deg,
        n_nodes=SMALL_GRAPH_NODES,
    )


def build_random_mock_graph(
    n_nodes: int = RANDOM_GRAPH_NODES,
    n_edges: int = RANDOM_GRAPH_EDGES,
    seed: int = RANDOM_SEED,
) -> GraphCase:
    """构造 100 节点、500 边的随机图。

    设计目标：
    - 95~99 作为 dead-ends，没有出边；
    - 92/93/94 构成 spider-trap，只向集合内部连边；
    - 其余边使用固定随机种子生成，保证可复现。
    """

    if n_nodes != RANDOM_GRAPH_NODES:
        raise ValueError("当前随机 mock 图固定为 100 个节点")
    if n_edges != RANDOM_GRAPH_EDGES:
        raise ValueError("当前随机 mock 图固定为 500 条边")

    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = {
        (92, 93),
        (92, 94),
        (93, 92),
        (93, 94),
        (94, 92),
        (94, 93),
    }

    while len(edges) < n_edges:
        src = int(rng.integers(0, 95))
        if src >= 92:
            dst = int(rng.choice(np.array([92, 93, 94], dtype=np.int32)))
        else:
            dst = int(rng.integers(0, n_nodes))
        if src == dst:
            continue
        edges.add((src, dst))

    edges_arr = np.array(sorted(edges), dtype=np.int32)
    row_ptr, col_idx, out_deg = edges_to_csr(edges_arr, n_nodes)
    return GraphCase(
        name="random_mock",
        edges=edges_arr,
        row_ptr=row_ptr,
        col_idx=col_idx,
        out_deg=out_deg,
        n_nodes=n_nodes,
    )


def dense_reference_pagerank(
    edges: np.ndarray,
    n_nodes: int,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype: np.dtype = np.float64,
    max_iter: int = 400,
) -> tuple[np.ndarray, int, float]:
    """使用朴素 O(N^2) dense 公式作为 reference。

    这里的“朴素”不是现成 PageRank API，而是直接构造 dense 转移矩阵：

        T[u, v] = 1 / out_deg[u], if u -> v

    然后每轮原地执行：

        temp = T^T @ r
        r_new[:] = beta * temp + base
    """

    row_ptr, col_idx, out_deg = edges_to_csr(edges, n_nodes)
    transition = np.zeros((n_nodes, n_nodes), dtype=dtype)
    live_mask = out_deg > 0
    dead_mask = ~live_mask

    for src in range(n_nodes):
        degree = int(out_deg[src])
        if degree == 0:
            continue
        start = int(row_ptr[src])
        stop = int(row_ptr[src + 1])
        transition[src, col_idx[start:stop]] = dtype(1.0 / degree)

    r = np.full(n_nodes, 1.0 / n_nodes, dtype=dtype)
    r_new = np.empty(n_nodes, dtype=dtype)
    temp = np.empty(n_nodes, dtype=dtype)

    for num_iter in range(1, max_iter + 1):
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype((1.0 - beta) / n_nodes + beta * dangling_mass / n_nodes)
        np.dot(transition.T, r, out=temp)
        temp *= beta
        r_new[:] = temp
        r_new += base

        delta = float(np.sum(np.abs(r_new - r), dtype=np.float64))
        if delta < eps:
            return r_new.astype(dtype, copy=True), num_iter, delta
        r, r_new = r_new, r

    return r.astype(dtype, copy=True), max_iter, delta


def edge_formula_steps(
    edges: np.ndarray,
    n_nodes: int,
    steps: int,
    beta: float = 0.85,
    dtype: np.dtype = np.float64,
) -> list[np.ndarray]:
    """按定义式逐边累加，返回前若干轮迭代结果。

    该函数专门给小图手算断言使用。
    """

    _, _, out_deg = edges_to_csr(edges, n_nodes)
    r = np.full(n_nodes, 1.0 / n_nodes, dtype=dtype)
    history = [r.copy()]
    inv_out_deg = np.zeros(n_nodes, dtype=dtype)
    live_mask = out_deg > 0
    inv_out_deg[live_mask] = 1.0 / out_deg[live_mask]
    dead_mask = ~live_mask

    for _ in range(steps):
        r_new = np.empty(n_nodes, dtype=dtype)
        dangling_mass = float(np.sum(r[dead_mask], dtype=np.float64))
        base = dtype((1.0 - beta) / n_nodes + beta * dangling_mass / n_nodes)
        r_new.fill(base)
        for src, dst in edges:
            if out_deg[src] == 0:
                continue
            r_new[dst] += beta * r[src] * inv_out_deg[src]
        history.append(r_new.copy())
        r = r_new
    return history


def main() -> None:
    """命令行入口：打印 mock 图摘要。"""

    parser = argparse.ArgumentParser(description="Print mock graph summaries")
    parser.add_argument("--show", choices=("small", "random", "all"), default="all")
    args = parser.parse_args()

    outputs = []
    if args.show in ("small", "all"):
        small = build_small_handcalc_graph()
        history = edge_formula_steps(small.edges, small.n_nodes, steps=2)
        outputs.append(
            {
                "name": small.name,
                "n_nodes": small.n_nodes,
                "n_edges": int(small.edges.shape[0]),
                "step_0": history[0].tolist(),
                "step_1": history[1].tolist(),
                "step_2": history[2].tolist(),
            }
        )

    if args.show in ("random", "all"):
        random_case = build_random_mock_graph()
        ref, num_iter, delta = dense_reference_pagerank(random_case.edges, random_case.n_nodes)
        outputs.append(
            {
                "name": random_case.name,
                "n_nodes": random_case.n_nodes,
                "n_edges": int(random_case.edges.shape[0]),
                "num_iter": num_iter,
                "delta": delta,
                "sum": float(ref.sum()),
            }
        )

    for item in outputs:
        print(item)


if __name__ == "__main__":
    main()
