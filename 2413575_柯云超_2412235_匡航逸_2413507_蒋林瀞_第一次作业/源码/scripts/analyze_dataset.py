"""数据集画像脚本。

职责：
- 读取作业给定的 Data.txt；
- 统计节点数、边数、入/出度、dead-end、弱连通分量；
- 输出 JSON、度分布 CSV 和 PNG 图，供报告第 1 章使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def read_edges(path: str) -> np.ndarray:
    """读取边表为 int32 二维数组。"""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    edges = np.loadtxt(path, dtype=np.int32)
    if edges.size == 0:
        raise ValueError("输入数据为空")
    if edges.ndim == 1:
        if edges.size != 2:
            raise ValueError("输入文件格式错误：每行应包含两个整数")
        edges = edges.reshape(1, 2)
    if edges.shape[1] != 2:
        raise ValueError("输入文件格式错误：每行应包含 FromNodeID ToNodeID")
    if np.any(edges < 0):
        raise ValueError("节点编号必须为非负整数")
    return edges


class DisjointSet:
    """用于统计弱连通分量的并查集。"""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        """路径压缩查找。"""

        root = x
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[x]) != x:
            parent = int(self.parent[x])
            self.parent[x] = root
            x = parent
        return root

    def union(self, a: int, b: int) -> None:
        """按集合大小合并两个节点。"""

        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if int(self.size[root_a]) < int(self.size[root_b]):
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]

    def component_sizes(self) -> list[int]:
        """返回所有弱连通分量大小。"""

        counter: Counter[int] = Counter()
        for node in range(self.parent.shape[0]):
            counter[self.find(node)] += 1
        return sorted(counter.values(), reverse=True)


def compute_stats(edges: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """计算数据集统计指标。"""

    src = edges[:, 0]
    dst = edges[:, 1]
    n_nodes = int(max(int(src.max()), int(dst.max())) + 1)
    n_edges = int(edges.shape[0])

    out_deg = np.bincount(src, minlength=n_nodes).astype(np.int32, copy=False)
    in_deg = np.bincount(dst, minlength=n_nodes).astype(np.int32, copy=False)
    total_deg = in_deg + out_deg
    dead_end_count = int(np.count_nonzero(out_deg == 0))

    dsu = DisjointSet(n_nodes)
    for from_node, to_node in edges:
        dsu.union(int(from_node), int(to_node))
    component_sizes = dsu.component_sizes()

    stats: dict[str, object] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_out_degree": float(out_deg.mean()),
        "avg_in_degree": float(in_deg.mean()),
        "max_out_degree": int(out_deg.max()),
        "max_in_degree": int(in_deg.max()),
        "max_total_degree": int(total_deg.max()),
        "dead_end_count": dead_end_count,
        "dead_end_ratio": float(dead_end_count / n_nodes),
        "weak_component_count": int(len(component_sizes)),
        "largest_weak_component_size": int(component_sizes[0]) if component_sizes else 0,
        "largest_weak_component_ratio": float(component_sizes[0] / n_nodes) if component_sizes else 0.0,
        "isolated_node_count": int(np.count_nonzero(total_deg == 0)),
        "self_loop_count": int(np.count_nonzero(src == dst)),
        "duplicate_edge_count": int(n_edges - np.unique(edges, axis=0).shape[0]),
    }
    return stats, in_deg, out_deg


def write_degree_distribution(csv_path: str, in_deg: np.ndarray, out_deg: np.ndarray) -> None:
    """输出入度/出度频数表。"""

    max_degree = int(max(int(in_deg.max()), int(out_deg.max())))
    in_counts = np.bincount(in_deg, minlength=max_degree + 1)
    out_counts = np.bincount(out_deg, minlength=max_degree + 1)

    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["degree", "in_count", "out_count"])
        for degree in range(max_degree + 1):
            writer.writerow([degree, int(in_counts[degree]), int(out_counts[degree])])


def plot_degree_distribution(fig_path: str, in_deg: np.ndarray, out_deg: np.ndarray) -> None:
    """绘制度分布图。matplotlib 只在需要画图时导入。"""

    import matplotlib.pyplot as plt

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for ax, degrees, title, color in (
        (axes[0], out_deg, "Out-degree distribution", "#2F6B8F"),
        (axes[1], in_deg, "In-degree distribution", "#C9653B"),
    ):
        nonzero_bins = np.bincount(degrees)
        xs = np.arange(nonzero_bins.shape[0])
        ax.bar(xs, nonzero_bins, width=0.9, color=color, alpha=0.86)
        ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count (log)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Analyze PageRank dataset")
    parser.add_argument("--data", default="Data.txt", help="输入边表路径")
    parser.add_argument("--out-json", default="experiments/dataset_stats.json", help="统计 JSON 输出路径")
    parser.add_argument(
        "--degree-csv",
        default="experiments/degree_distribution.csv",
        help="度分布 CSV 输出路径",
    )
    parser.add_argument(
        "--fig",
        default="report/fig/degree_distribution.png",
        help="度分布 PNG 输出路径",
    )
    args = parser.parse_args()

    edges = read_edges(args.data)
    stats, in_deg, out_deg = compute_stats(edges)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.degree_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")

    write_degree_distribution(args.degree_csv, in_deg, out_deg)
    plot_degree_distribution(args.fig, in_deg, out_deg)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
