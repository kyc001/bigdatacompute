"""A 侧算法测试。"""

from __future__ import annotations

import os

import numpy as np

from main import dump_top_k, load_graph, power_iteration
from mock_graph import (
    build_small_handcalc_graph,
    dense_reference_pagerank,
    edge_formula_steps,
    edges_to_csr,
)


def test_small_graph_matches_handcalc_and_dense_reference() -> None:
    """5 节点小图：前两轮手算值与最终收敛结果都应正确。"""

    graph = build_small_handcalc_graph()
    history = edge_formula_steps(graph.edges, graph.n_nodes, steps=2, beta=0.85)
    expected_step_1 = np.array([0.234, 0.234, 0.404, 0.064, 0.064], dtype=np.float64)
    expected_step_2 = np.array([0.38428, 0.23978, 0.29418, 0.04088, 0.04088], dtype=np.float64)

    assert np.allclose(history[1], expected_step_1, atol=1e-10)
    assert np.allclose(history[2], expected_step_2, atol=1e-10)

    ranks, iters, delta = power_iteration(
        graph.row_ptr,
        graph.col_idx,
        graph.out_deg,
        graph.n_nodes,
        beta=0.85,
        eps=1e-10,
        dtype=np.float64,
    )
    reference, _, _ = dense_reference_pagerank(graph.edges, graph.n_nodes, beta=0.85, eps=1e-10)

    assert np.allclose(ranks, reference, atol=1e-10)
    assert iters > 0
    assert delta < 1e-10


def test_dead_end_graph_top_k_matches_reference_order() -> None:
    """含 dead-end 的小图：Top-k 顺序应与稠密参考实现一致。"""

    edges = np.array(
        [
            [0, 2],
            [1, 2],
            [2, 3],
            [3, 2],
        ],
        dtype=np.int32,
    )
    row_ptr, col_idx, out_deg = edges_to_csr(edges, 5)
    ranks, _, _ = power_iteration(row_ptr, col_idx, out_deg, 5, beta=0.85, eps=1e-10)
    reference, _, _ = dense_reference_pagerank(edges, 5, beta=0.85, eps=1e-10)

    os.makedirs(".tmp_test", exist_ok=True)
    out_path = os.path.join(".tmp_test", "algorithm_dead_end_res.txt")
    dump_top_k(out_path, ranks, k=5)
    with open(out_path, "r", encoding="utf-8") as file_obj:
        written_ids = [int(line.split()[0]) for line in file_obj.read().splitlines()]

    expected_order = np.lexsort((np.arange(5, dtype=np.int32), -reference))[:5]
    assert written_ids == expected_order.tolist()
    assert written_ids[0] == 2


def test_spider_trap_graph_converges_within_100_iterations() -> None:
    """含 spider-trap 的小图：应在 100 轮内收敛，并保持概率和为 1。"""

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 2],
            [3, 2],
        ],
        dtype=np.int32,
    )
    row_ptr, col_idx, out_deg = edges_to_csr(edges, 4)
    ranks, iters, delta = power_iteration(
        row_ptr,
        col_idx,
        out_deg,
        4,
        beta=0.85,
        eps=1e-8,
        max_iter=100,
    )

    top2 = np.argsort(-ranks)[:2]
    assert set(map(int, top2)) == {1, 2}
    assert iters <= 100
    assert delta < 1e-8
    assert abs(float(np.sum(ranks, dtype=np.float64)) - 1.0) <= 1e-5


def test_beta_zero_and_beta_one_boundaries() -> None:
    """边界参数：beta=0 与 beta=1 都应得到稳定且合法的概率向量。"""

    edges = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int32,
    )
    row_ptr, col_idx, out_deg = edges_to_csr(edges, 4)
    uniform = np.full(4, 0.25, dtype=np.float32)

    beta_zero_ranks, beta_zero_iters, _ = power_iteration(
        row_ptr,
        col_idx,
        out_deg,
        4,
        beta=0.0,
        eps=1e-12,
        max_iter=20,
    )
    assert np.allclose(beta_zero_ranks, uniform, atol=1e-7)
    assert beta_zero_iters == 1

    dead_row_ptr = np.zeros(5, dtype=np.int32)
    dead_col_idx = np.empty(0, dtype=np.int32)
    dead_out_deg = np.zeros(4, dtype=np.int32)
    beta_one_ranks, beta_one_iters, _ = power_iteration(
        dead_row_ptr,
        dead_col_idx,
        dead_out_deg,
        4,
        beta=1.0,
        eps=1e-12,
        max_iter=20,
    )
    assert np.allclose(beta_one_ranks, uniform, atol=1e-7)
    assert beta_one_iters == 1


def test_non_contiguous_ids_are_restored_in_dump_top_k() -> None:
    """冻结接口未显式返回 id_map 时，dump_top_k 也必须恢复原始 NodeID。"""

    os.makedirs(".tmp_test", exist_ok=True)
    data_path = os.path.join(".tmp_test", "algorithm_non_contiguous.txt")
    out_path = os.path.join(".tmp_test", "algorithm_non_contiguous_res.txt")
    with open(data_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("10 20\n20 30\n30 10\n")

    row_ptr, col_idx, out_deg, n_nodes = load_graph(data_path)
    ranks, _, _ = power_iteration(row_ptr, col_idx, out_deg, n_nodes, beta=0.85, eps=1e-10)
    signature = dump_top_k(out_path, ranks, k=3)
    with open(out_path, "r", encoding="utf-8") as file_obj:
        written_ids = [int(line.split()[0]) for line in file_obj.read().splitlines()]

    assert written_ids == [10, 20, 30]
    assert signature == "10,20,30"
