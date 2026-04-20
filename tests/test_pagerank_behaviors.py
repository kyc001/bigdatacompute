import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from blocks import build_blocks, iterate_by_block
from main import power_iteration
from mock_graph import (
    build_random_mock_graph,
    build_small_handcalc_graph,
    dense_reference_pagerank,
    edge_formula_steps,
    edges_to_csr,
)


def test_small_graph_block_and_csr_match_handcalc_reference():
    """5 节点小图：逐轮手算值与最终 block / csr 结果都应一致。"""

    graph = build_small_handcalc_graph()
    history = edge_formula_steps(graph.edges, graph.n_nodes, steps=2, beta=0.85)
    expected_step_1 = np.array([0.234, 0.234, 0.404, 0.064, 0.064], dtype=np.float64)
    expected_step_2 = np.array([0.38428, 0.23978, 0.29418, 0.04088, 0.04088], dtype=np.float64)

    assert np.allclose(history[1], expected_step_1, atol=1e-10)
    assert np.allclose(history[2], expected_step_2, atol=1e-10)

    reference, _, _ = dense_reference_pagerank(graph.edges, graph.n_nodes, eps=1e-12)
    csr_rank, _, _ = power_iteration(
        graph.row_ptr,
        graph.col_idx,
        graph.out_deg,
        graph.n_nodes,
        beta=0.85,
        eps=1e-10,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = build_blocks((graph.edges, graph.n_nodes), 2, tmpdir)
        block_rank, _, _ = iterate_by_block(
            metadata,
            graph.out_deg,
            graph.n_nodes,
            beta=0.85,
            eps=1e-10,
            tmp_dir=tmpdir,
        )

    assert np.allclose(csr_rank, reference.astype(np.float32), atol=1e-6)
    assert np.allclose(block_rank, reference.astype(np.float32), atol=1e-6)


def test_random_graph_block_matches_dense_reference():
    """100 节点随机图：分块版应与朴素 O(N^2) dense reference 一致。"""

    graph = build_random_mock_graph()
    reference, _, _ = dense_reference_pagerank(graph.edges, graph.n_nodes, eps=1e-10)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = build_blocks(graph.edges, 8, tmpdir)
        block_rank, _, _ = iterate_by_block(
            metadata,
            graph.out_deg,
            graph.n_nodes,
            beta=0.85,
            eps=1e-8,
            tmp_dir=tmpdir,
        )

    assert np.allclose(block_rank, reference.astype(np.float32), atol=1e-6)


def test_results_are_stable_across_k_values():
    """K 取 1 / 4 / 8 / 16 时，分块结果应一致。"""

    graph = build_random_mock_graph()
    results = []

    for k_value in (1, 4, 8, 16):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_blocks(graph.edges, k_value, tmpdir)
            ranks, _, _ = iterate_by_block(
                metadata,
                graph.out_deg,
                graph.n_nodes,
                beta=0.85,
                eps=1e-8,
                tmp_dir=tmpdir,
            )
            results.append(ranks)

    assert np.allclose(results[0], results[1], atol=1e-6)
    assert np.allclose(results[1], results[2], atol=1e-6)
    assert np.allclose(results[2], results[3], atol=1e-6)


def test_all_dead_end_graph_is_uniform_after_compensation():
    """全 dead-end 病态图应在补偿公式下收敛到均匀分布。"""

    n_nodes = 4
    row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    col_idx = np.empty(0, dtype=np.int32)
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    expected = np.full(n_nodes, 0.25, dtype=np.float32)

    csr_rank, _, _ = power_iteration(row_ptr, col_idx, out_deg, n_nodes, 0.85, 1e-10)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = build_blocks((iter(()), n_nodes), 2, tmpdir)
        block_rank, _, _ = iterate_by_block(
            metadata,
            out_deg,
            n_nodes,
            beta=0.85,
            eps=1e-10,
            tmp_dir=tmpdir,
        )

    assert np.allclose(csr_rank, expected, atol=1e-7)
    assert np.allclose(block_rank, expected, atol=1e-7)


def test_spider_trap_graph_converges_normally():
    """spider-trap 图应正常收敛，且 trap 节点占据更高权重。"""

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 2],
            [4, 2],
        ],
        dtype=np.int32,
    )
    row_ptr, col_idx, out_deg = edges_to_csr(edges, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = build_blocks(edges, 2, tmpdir)
        block_rank, _, _ = iterate_by_block(
            metadata,
            out_deg,
            5,
            beta=0.85,
            eps=1e-10,
            tmp_dir=tmpdir,
        )

    top2 = np.argsort(-block_rank)[:2]
    assert set(map(int, top2)) == {2, 3}
    assert abs(float(block_rank.sum()) - 1.0) < 1e-5


def test_benchmark_parses_stdout_json_and_writes_csv(tmp_path: Path):
    """benchmark.py 应能解析 stdout 最后一行 JSON，并写出固定 schema CSV。"""

    probe_script = tmp_path / "json_probe.py"
    probe_script.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--data')",
                "parser.add_argument('--beta')",
                "parser.add_argument('--eps')",
                "parser.add_argument('--out')",
                "parser.add_argument('--mode')",
                "parser.add_argument('--K')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--max-iter')",
                "args = parser.parse_args()",
                "with open(args.out, 'w', encoding='utf-8') as f:",
                "    f.write('0 1.0000000000\\n')",
                "print('warmup line')",
                "print(json.dumps({",
                "    'peak_rss_mb': 12.5,",
                "    'wall_sec': 0.123,",
                "    'iters': 7,",
                "    'mode': args.mode,",
                "    'K': int(args.K),",
                "    'dtype': args.dtype,",
                "    'top10_signature': '1,2,3,4,5,6,7,8,9,10'",
                "}))",
            ]
        ),
        encoding="utf-8",
    )

    data_path = tmp_path / "dummy.txt"
    data_path.write_text("0 1\n", encoding="utf-8")
    csv_path = tmp_path / "bench.csv"

    command = [
        sys.executable,
        "benchmark.py",
        "--main",
        str(probe_script),
        "--data",
        str(data_path),
        "--out",
        str(csv_path),
        "--interval",
        "0.01",
        "--runs",
        "1",
        "--modes",
        "csr_block",
        "--K",
        "8",
        "--dtype",
        "float32",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)

    assert result.returncode == 0, result.stderr

    with csv_path.open("r", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert rows[0]["mode"] == "csr_block"
    assert rows[0]["K"] == "8"
    assert rows[0]["dtype"] == "float32"
    assert rows[0]["iters"] == "7"
    assert rows[0]["top10_signature"] == "1,2,3,4,5,6,7,8,9,10"
