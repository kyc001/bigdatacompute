"""论文风格实验图表生成脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "dense": "#4C566A",
    "csr": "#2E5EAA",
    "csr_block": "#C96A3D",
    "float32": "#2E5EAA",
    "float64": "#5B8C5A",
    "compensation": "#2E5EAA",
    "ignore": "#B23A48",
    "delete": "#C96A3D",
    "jaccard": "#2E5EAA",
    "kendall": "#C96A3D",
}


def setup_style() -> None:
    """设置统一的论文风格。"""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D8DEE9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 150,
            "mathtext.fontset": "dejavuserif",
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    """读取 CSV。"""

    return pd.read_csv(path)


def mean_std(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """按列分组计算均值与标准差。"""

    grouped = df.groupby(group_col, as_index=False)[value_col].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def plot_degree_distribution(degree_csv: Path, out_dir: Path) -> None:
    """绘制度分布图。"""

    df = load_csv(degree_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    for ax, column, title, color in (
        (axes[0], "out_count", "Out-degree distribution", "#2E5EAA"),
        (axes[1], "in_count", "In-degree distribution", "#C96A3D"),
    ):
        ax.bar(df["degree"], df[column], color=color, width=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count (log scale)")
        ax.set_title(title)
        ax.grid(axis="y")

    fig.savefig(out_dir / "degree_distribution.png")
    plt.close(fig)


def plot_implementation_overview(e1_path: Path, e4_path: Path, out_dir: Path) -> None:
    """绘制 E1/E2/E4 总览图。"""

    dense = load_csv(e1_path)
    sparse = load_csv(e4_path)
    df = pd.concat([dense, sparse], ignore_index=True)
    order = ["dense", "csr", "csr_block"]
    labels = ["Dense (E1)", "CSR (E2)", "CSR+Block (E4)"]

    mem = df.groupby("mode")["peak_rss_mb"].agg(["mean", "std"]).reindex(order)
    time = df.groupby("mode")["wall_sec"].agg(["mean", "std"]).reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    x = np.arange(len(order))
    colors = [PALETTE[item] for item in order]

    axes[0].bar(x, mem["mean"], yerr=mem["std"], capsize=4, color=colors, width=0.62)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Peak RSS (MB)")
    axes[0].set_title("Memory footprint by implementation")
    axes[0].grid(axis="y")

    axes[1].bar(x, time["mean"], yerr=time["std"], capsize=4, color=colors, width=0.62)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Wall-clock time (s)")
    axes[1].set_title("Runtime by implementation")
    axes[1].grid(axis="y")

    fig.savefig(out_dir / "implementation_overview.png")
    plt.close(fig)


def plot_k_tradeoff(e3_path: Path, out_dir: Path) -> None:
    """绘制 E3：K 值敏感性。"""

    df = load_csv(e3_path)
    grouped = df.groupby("K").agg(
        rss_mean=("peak_rss_mb", "mean"),
        rss_std=("peak_rss_mb", "std"),
        time_mean=("wall_sec", "mean"),
        time_std=("wall_sec", "std"),
    )
    grouped = grouped.sort_index()
    ks = grouped.index.to_numpy(dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)

    axes[0].plot(ks, grouped["rss_mean"], marker="o", color=PALETTE["csr"], linewidth=1.8)
    axes[0].fill_between(
        ks,
        grouped["rss_mean"] - grouped["rss_std"].fillna(0.0),
        grouped["rss_mean"] + grouped["rss_std"].fillna(0.0),
        color=PALETTE["csr"],
        alpha=0.15,
    )
    axes[0].set_xlabel("Block count K")
    axes[0].set_ylabel("Peak RSS (MB)")
    axes[0].set_title("Memory sensitivity to K")
    axes[0].set_xticks(ks)
    axes[0].grid(axis="y")

    axes[1].plot(ks, grouped["time_mean"], marker="s", color=PALETTE["csr_block"], linewidth=1.8)
    axes[1].fill_between(
        ks,
        grouped["time_mean"] - grouped["time_std"].fillna(0.0),
        grouped["time_mean"] + grouped["time_std"].fillna(0.0),
        color=PALETTE["csr_block"],
        alpha=0.15,
    )
    axes[1].set_xlabel("Block count K")
    axes[1].set_ylabel("Wall-clock time (s)")
    axes[1].set_title("Runtime sensitivity to K")
    axes[1].set_xticks(ks)
    axes[1].grid(axis="y")

    fig.savefig(out_dir / "k_tradeoff.png")
    plt.close(fig)


def plot_beta_sensitivity(sweep_path: Path, out_dir: Path) -> None:
    """绘制 E5：beta 敏感性。"""

    df = load_csv(sweep_path)
    beta_df = df[df["experiment"] == "E5_beta"].sort_values("beta")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)

    axes[0].plot(
        beta_df["beta"],
        beta_df["jaccard_vs_beta085"],
        marker="o",
        linewidth=1.8,
        color=PALETTE["jaccard"],
        label="Jaccard",
    )
    axes[0].plot(
        beta_df["beta"],
        beta_df["kendall_tau_intersection_vs_beta085"],
        marker="s",
        linewidth=1.8,
        color=PALETTE["kendall"],
        label="Kendall tau",
    )
    axes[0].axvline(0.85, color="#7B8794", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel(r"Damping factor $\beta$")
    axes[0].set_ylabel("Top-10 similarity")
    axes[0].set_ylim(0.75, 1.02)
    axes[0].set_title("Ranking stability under beta sweep")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="y")

    axes[1].plot(beta_df["beta"], beta_df["iters"], marker="o", color="#5B8C5A", linewidth=1.8)
    axes[1].set_xlabel(r"Damping factor $\beta$")
    axes[1].set_ylabel("Iterations to convergence")
    axes[1].set_title("Convergence cost under beta sweep")
    axes[1].grid(axis="y")

    fig.savefig(out_dir / "beta_sensitivity.png")
    plt.close(fig)


def plot_epsilon_sensitivity(sweep_path: Path, out_dir: Path) -> None:
    """绘制 E6：epsilon 敏感性。"""

    df = load_csv(sweep_path)
    eps_df = df[df["experiment"] == "E6_eps"].copy()
    eps_df["eps_value"] = eps_df["eps"].astype(float)
    eps_df = eps_df.sort_values("eps_value")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)

    axes[0].plot(eps_df["eps_value"], eps_df["iters"], marker="o", color=PALETTE["csr"], linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"Convergence threshold $\varepsilon$")
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Iteration count vs. epsilon")
    axes[0].grid(axis="y")

    axes[1].plot(eps_df["eps_value"], eps_df["delta"], marker="s", color=PALETTE["csr_block"], linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Convergence threshold $\varepsilon$")
    axes[1].set_ylabel("Final L1 delta")
    axes[1].set_title("Terminal residual vs. epsilon")
    axes[1].grid(axis="y")

    fig.savefig(out_dir / "epsilon_sensitivity.png")
    plt.close(fig)


def plot_dtype_comparison(e7_path: Path, out_dir: Path) -> None:
    """绘制 E7：dtype 对比。"""

    df = load_csv(e7_path)
    order = ["float32", "float64"]
    labels = ["float32", "float64"]
    mem = df.groupby("dtype")["peak_rss_mb"].agg(["mean", "std"]).reindex(order)
    time = df.groupby("dtype")["wall_sec"].agg(["mean", "std"]).reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    x = np.arange(len(order))
    colors = [PALETTE[item] for item in order]

    axes[0].bar(x, mem["mean"], yerr=mem["std"], capsize=4, color=colors, width=0.62)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Peak RSS (MB)")
    axes[0].set_title("Memory cost by floating-point type")
    axes[0].grid(axis="y")

    axes[1].bar(x, time["mean"], yerr=time["std"], capsize=4, color=colors, width=0.62)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Wall-clock time (s)")
    axes[1].set_title("Runtime by floating-point type")
    axes[1].grid(axis="y")

    fig.savefig(out_dir / "dtype_comparison.png")
    plt.close(fig)


def plot_deadend_comparison(e8_path: Path, out_dir: Path) -> None:
    """绘制 E8：dead-end 策略对比。"""

    df = load_csv(e8_path)
    order = ["compensation", "ignore", "delete"]
    labels = ["Compensation", "Ignore", "Delete"]
    colors = [PALETTE[item] for item in order]
    df = df.set_index("strategy").loc[order].reset_index()
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)

    axes[0].bar(x, df["wall_sec"], color=colors, width=0.62)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Runtime")
    axes[0].grid(axis="y")
    for idx, row in df.iterrows():
        axes[0].text(idx, row["wall_sec"] + 0.02, f"iters={int(row['iters'])}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, df["rank_sum"], color=colors, width=0.62)
    axes[1].axhline(1.0, color="#7B8794", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylabel("Sum of PageRank vector")
    axes[1].set_title("Probability mass conservation")
    axes[1].grid(axis="y")

    axes[2].bar(x, df["jaccard_vs_compensation"], color=colors, width=0.62)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(x, labels, rotation=12)
    axes[2].set_ylabel("Jaccard vs. compensation")
    axes[2].set_title("Top-10 consistency")
    axes[2].grid(axis="y")

    fig.savefig(out_dir / "deadend_strategy_comparison.png")
    plt.close(fig)


def write_report_summary(
    stats_path: Path,
    e1_path: Path,
    e4_path: Path,
    e3_path: Path,
    e7_path: Path,
    e8_path: Path,
    sweep_path: Path,
    out_path: Path,
) -> None:
    """把报告常用均值写成 JSON，便于 LaTeX/人工核对。"""

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    e1 = load_csv(e1_path)
    e4 = load_csv(e4_path)
    e3 = load_csv(e3_path)
    e7 = load_csv(e7_path)
    e8 = load_csv(e8_path)
    sweep = load_csv(sweep_path)

    summary = {
        "dataset": stats,
        "implementation": pd.concat([e1, e4], ignore_index=True).groupby("mode")[["peak_rss_mb", "wall_sec", "iters"]].mean().round(6).to_dict(),
        "k_tradeoff": e3.groupby("K")[["peak_rss_mb", "wall_sec", "iters"]].mean().round(6).to_dict(),
        "dtype": e7.groupby("dtype")[["peak_rss_mb", "wall_sec", "iters"]].mean().round(6).to_dict(),
        "deadend": e8.set_index("strategy")[["wall_sec", "iters", "delta", "rank_sum", "jaccard_vs_compensation"]].round(6).to_dict(),
        "beta": sweep[sweep["experiment"] == "E5_beta"].set_index("beta")[["iters", "jaccard_vs_beta085", "kendall_tau_intersection_vs_beta085"]].round(6).to_dict(),
        "epsilon": sweep[sweep["experiment"] == "E6_eps"].set_index("eps")[["iters", "delta"]].round(12).to_dict(),
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Generate publication-style figures for the PageRank report")
    parser.add_argument("--dataset-stats", default="experiments/dataset_stats.json")
    parser.add_argument("--degree-csv", default="experiments/degree_distribution.csv")
    parser.add_argument("--e1", default="experiments/E1_dense.csv")
    parser.add_argument("--e3", default="experiments/E3.csv")
    parser.add_argument("--e4", default="experiments/E4.csv")
    parser.add_argument("--e7", default="experiments/E7.csv")
    parser.add_argument("--e8", default="experiments/E8.csv")
    parser.add_argument("--sweep", default="experiments/sweep_results.csv")
    parser.add_argument("--out-dir", default="report/fig")
    parser.add_argument("--summary-json", default="experiments/report_summary.json")
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_degree_distribution(Path(args.degree_csv), out_dir)
    plot_implementation_overview(Path(args.e1), Path(args.e4), out_dir)
    plot_k_tradeoff(Path(args.e3), out_dir)
    plot_beta_sensitivity(Path(args.sweep), out_dir)
    plot_epsilon_sensitivity(Path(args.sweep), out_dir)
    plot_dtype_comparison(Path(args.e7), out_dir)
    plot_deadend_comparison(Path(args.e8), out_dir)
    write_report_summary(
        Path(args.dataset_stats),
        Path(args.e1),
        Path(args.e4),
        Path(args.e3),
        Path(args.e7),
        Path(args.e8),
        Path(args.sweep),
        Path(args.summary_json),
    )
    print(f"wrote figures to {out_dir}")
    print(f"wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
