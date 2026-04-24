"""报告图表生成脚本。

职责：
- 从真实 CSV 读取实验数据；
- 生成报告要求的核心 PNG；
- 对尚未由 A/B 暴露的数据只标注待补，不手填、不估算。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path: str) -> list[dict[str, str]]:
    """读取 CSV；文件不存在时返回空列表，方便阶段性交付。"""

    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def to_float(value: str, default: float = 0.0) -> float:
    """安全转浮点数。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def average_by(rows: list[dict[str, str]], keys: tuple[str, ...], metric: str) -> dict[tuple[str, ...], float]:
    """按多个键分组求均值。"""

    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = tuple(row.get(item, "") for item in keys)
        grouped[key].append(to_float(row.get(metric, "")))
    return {key: mean(values) for key, values in grouped.items() if values}


def plot_memory_bar(e4_rows: list[dict[str, str]], out_dir: Path) -> None:
    """绘制 E1/E2/E3/E4 内存峰值柱状图。

    当前仓库尚未提供 A 的 E1 数据；脚本只绘制真实 CSV 中已有的 E2/E4，
    并在图内标注 E1 待补。
    """

    import matplotlib.pyplot as plt

    memory_avg = average_by(e4_rows, ("mode", "K", "dtype"), "peak_rss_mb")
    labels = []
    values = []
    colors = []

    csr_key = ("csr", "8", "float32")
    block_key = ("csr_block", "8", "float32")
    if csr_key in memory_avg:
        labels.append("E2 CSR")
        values.append(memory_avg[csr_key])
        colors.append("#2F6B8F")
    if block_key in memory_avg:
        labels.append("E4 CSR+Block")
        values.append(memory_avg[block_key])
        colors.append("#C9653B")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    if values:
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.bar_label(bars, fmt="%.2f MB", padding=4, fontsize=9)
    ax.text(
        0.02,
        0.94,
        "E1 dense / E3 standalone block: pending real CSV from A/B",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory peak by implementation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_peak_bar.png", dpi=300)
    plt.close(fig)


def plot_k_tradeoff(e3_rows: list[dict[str, str]], out_dir: Path) -> None:
    """绘制 K 与内存/时间双轴图。"""

    import matplotlib.pyplot as plt

    mem_avg = average_by(e3_rows, ("K",), "peak_rss_mb")
    time_avg = average_by(e3_rows, ("K",), "wall_sec")
    k_values = sorted({int(key[0]) for key in mem_avg.keys() | time_avg.keys()})
    mem_values = [mem_avg[(str(k),)] for k in k_values]
    time_values = [time_avg[(str(k),)] for k in k_values]

    fig, ax_left = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax_right = ax_left.twinx()
    ax_left.plot(k_values, mem_values, marker="o", color="#2F6B8F", label="Peak RSS")
    ax_right.plot(k_values, time_values, marker="s", color="#C9653B", label="Wall time")
    ax_left.set_xlabel("Block count K")
    ax_left.set_ylabel("Peak RSS (MB)", color="#2F6B8F")
    ax_right.set_ylabel("Wall time (s)", color="#C9653B")
    ax_left.set_xticks(k_values)
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_title("Block count trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "k_memory_time.png", dpi=300)
    plt.close(fig)


def plot_beta_similarity(sweep_rows: list[dict[str, str]], out_dir: Path) -> None:
    """绘制 beta 与 Top-10 相似度关系。"""

    import matplotlib.pyplot as plt

    rows = [row for row in sweep_rows if row.get("experiment") == "E5_beta" and row.get("status") == "ok"]
    rows.sort(key=lambda row: to_float(row["beta"]))
    betas = [to_float(row["beta"]) for row in rows]
    jaccard = [to_float(row["jaccard_vs_beta085"]) for row in rows]
    tau = [to_float(row["kendall_tau_intersection_vs_beta085"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.plot(betas, jaccard, marker="o", color="#2F6B8F", label="Jaccard")
    ax.plot(betas, tau, marker="s", color="#7A6A9B", label="Kendall tau")
    ax.axvline(0.85, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Top-10 similarity vs beta=0.85")
    ax.set_ylim(0, 1.05)
    ax.set_title("Teleport beta sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "beta_top10_similarity.png", dpi=300)
    plt.close(fig)


def plot_epsilon_residual(sweep_rows: list[dict[str, str]], out_dir: Path) -> None:
    """绘制 E6 的迭代轮数与最终残差。

    INTERFACE.md 当前只要求 main.py 输出最终 delta，尚未暴露逐轮 residual。
    因此这里画的是真实采集到的 eps -> (iters, final delta) 汇总图。
    """

    import matplotlib.pyplot as plt

    rows = [row for row in sweep_rows if row.get("experiment") == "E6_eps" and row.get("status") == "ok"]
    rows.sort(key=lambda row: to_float(row["eps"]), reverse=True)
    labels = [row["eps"] for row in rows]
    iters = [to_float(row["iters"]) for row in rows]
    deltas = [to_float(row["delta"]) for row in rows]

    fig, ax_left = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax_right = ax_left.twinx()
    xs = range(len(rows))
    ax_left.bar(xs, iters, color="#2F6B8F", alpha=0.82, width=0.55, label="Iterations")
    ax_right.plot(xs, deltas, marker="o", color="#C9653B", label="Final delta")
    ax_right.set_yscale("log")
    ax_left.set_xticks(list(xs))
    ax_left.set_xticklabels(labels)
    ax_left.set_xlabel("Epsilon")
    ax_left.set_ylabel("Iterations")
    ax_right.set_ylabel("Final L1 delta (log)")
    ax_left.set_title("Convergence threshold sensitivity")
    ax_left.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_residual_summary.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Generate report figures from CSV")
    parser.add_argument("--e3", default="experiments/E3.csv")
    parser.add_argument("--e4", default="experiments/E4.csv")
    parser.add_argument("--sweep", default="experiments/sweep_results.csv")
    parser.add_argument("--out-dir", default="report/fig")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e3_rows = read_csv(args.e3)
    e4_rows = read_csv(args.e4)
    sweep_rows = read_csv(args.sweep)

    plot_memory_bar(e4_rows, out_dir)
    plot_k_tradeoff(e3_rows, out_dir)
    plot_beta_similarity(sweep_rows, out_dir)
    plot_epsilon_residual(sweep_rows, out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
