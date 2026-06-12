import csv
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#5B78A7"
ORANGE = "#E68B32"
GREEN = "#5F8F73"
RED = "#B85C5A"
GRAY = "#6E6E6E"
BLACK = "#111111"
GRID = "#D8D8D8"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 8.2,
    "axes.titlesize": 11.2,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.95,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


LABELS = {
    "final": "最终方法",
    "dense_item": "完整物品统计",
    "stride16": "激进采样",
    "item_only": "Item-only基线",
    "constant": "无增量预测",
    "no_user_prior": "去用户分段先验",
    "no_count_terms": "去计数特征",
    "no_user_residual": "去用户残差",
    "no_item_residual": "去物品残差",
    "no_prior_no_count": "去先验和计数",
    "stride2": "物品stride=2",
    "stride8": "物品stride=8",
}


def load_rows():
    rows = []
    with (ROOT / "ablation_benchmark_results.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "method": row["method"],
                "label": LABELS.get(row["method"], row["label"]),
                "group": row["group"],
                "repeat": int(row["repeat"]),
                "total": float(row["total"]),
                "pre_rmse": float(row["pre_rmse"]),
                "post_rmse": float(row["post_rmse"]),
                "valid": row["valid"].lower() == "true",
            })
    return rows


ROWS = load_rows()
BASE_RMSE = ROWS[0]["pre_rmse"]


def method_rows(method):
    return sorted([r for r in ROWS if r["method"] == method], key=lambda r: r["repeat"])


def method_stats(method):
    rs = method_rows(method)
    vals = [r["total"] for r in rs]
    return {
        "method": method,
        "label": rs[0]["label"],
        "group": rs[0]["group"],
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
        "rmse": rs[0]["post_rmse"],
        "valid": rs[0]["valid"],
        "values": vals,
    }


STATS = {m: method_stats(m) for m in sorted({r["method"] for r in ROWS})}


def polish(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.2, width=0.9, color=BLACK)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def main_method_runtime():
    methods = ["final", "dense_item", "stride2", "stride8", "stride16", "item_only", "constant"]
    labels = [STATS[m]["label"].replace("物品", "物品\n").replace("Item-only", "Item-only\n") for m in methods]
    means = [STATS[m]["mean"] for m in methods]
    medians = [STATS[m]["median"] for m in methods]
    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.35))
    ax.bar(x - width / 2, means, width, label="平均值", color=BLUE, edgecolor=BLUE)
    ax.bar(x + width / 2, medians, width, label="中位数", color=ORANGE, edgecolor=ORANGE)
    ax.set_title("主要方法的 5 次重复测量耗时")
    ax.set_ylabel("10-run 总耗时 / 秒")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 0.16)
    ax.legend(frameon=False, loc="upper right")
    polish(ax)
    save(fig, "main_method_runtime.png")


def tradeoff_scatter():
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    methods = ["item_only", "stride16", "stride8", "final", "stride2", "dense_item"]
    style = {
        "final": (BLUE, "o"),
        "dense_item": (GREEN, "s"),
        "stride2": (ORANGE, "^"),
        "stride8": (GREEN, "^"),
        "stride16": (ORANGE, "s"),
        "item_only": (GRAY, "D"),
    }
    label_offsets = {
        "item_only": (0.0020, 0.0018, "left", "bottom"),
        "stride16": (-0.0022, 0.0016, "right", "bottom"),
        "stride8": (0.0020, -0.0012, "left", "top"),
        "final": (0.0020, -0.0018, "left", "top"),
        "stride2": (0.0020, 0.0012, "left", "bottom"),
        "dense_item": (-0.0020, -0.0020, "right", "top"),
    }
    for method in methods:
        s = STATS[method]
        color, marker = style[method]
        ax.scatter(s["mean"], s["rmse"], s=56, marker=marker, color=color,
                   edgecolor=BLACK, linewidth=0.72, zorder=3)
        dx, dy, ha, va = label_offsets.get(method, (0.002, 0.001, "left", "bottom"))
        ax.text(s["mean"] + dx, s["rmse"] + dy, s["label"], fontsize=6.9, ha=ha, va=va)
    ax.set_title("有效方法的精度-时间分布")
    ax.set_xlabel("5 次平均 10-run 总耗时 / 秒")
    ax.set_ylabel("更新后 RMSE")
    ax.set_xlim(0.114, 0.143)
    ax.set_ylim(0.912, 0.946)
    polish(ax)
    save(fig, "tradeoff_scatter.png")


def sampling_sweep():
    methods = ["dense_item", "stride2", "final", "stride8", "stride16"]
    strides = np.array([1, 2, 4, 8, 16], dtype=float)
    rmse = np.array([STATS[m]["rmse"] for m in methods])
    runtime = np.array([STATS[m]["mean"] for m in methods])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.3))
    ax2 = ax1.twinx()
    ax1.plot(strides, rmse, color=BLUE, marker="o", linewidth=1.6, label="RMSE")
    ax2.plot(strides, runtime, color=ORANGE, marker="s", linewidth=1.6, label="平均耗时")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(strides, ["完整", "2", "4", "8", "16"])
    ax1.set_title("物品侧采样强度扫描")
    ax1.set_xlabel("物品侧采样间隔")
    ax1.set_ylabel("更新后 RMSE", color=BLUE)
    ax2.set_ylabel("10-run 平均耗时 / 秒", color=ORANGE)
    ax1.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax1.set_ylim(0.912, 0.933)
    ax2.set_ylim(0.110, 0.145)
    ax1.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.55)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")
    save(fig, "sampling_sweep.png")


def ablation_rmse_delta():
    methods = ["no_item_residual", "no_prior_no_count", "no_count_terms", "no_user_prior", "no_user_residual"]
    deltas = [STATS[m]["rmse"] - STATS["final"]["rmse"] for m in methods]
    labels = [STATS[m]["label"] for m in methods]
    colors = [ORANGE if m == "no_item_residual" else BLUE for m in methods]
    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.barh(y, deltas, color=colors, edgecolor=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title("关键组件移除后的 RMSE 损失")
    ax.set_xlabel("相对最终方法的 RMSE 增量")
    ax.set_xlim(0.0, 0.076)
    for yi, delta in zip(y, deltas):
        ax.text(delta + 0.0012, yi, f"+{delta:.4f}", va="center", fontsize=7.6)
    polish(ax, grid_axis="x")
    save(fig, "ablation_rmse_delta.png")


def repeat_sequence():
    methods = ["final", "dense_item", "stride2", "stride16", "item_only"]
    styles = [
        (BLUE, "o", "-"),
        (GREEN, "s", "-"),
        (ORANGE, "^", "--"),
        (GRAY, "D", ":"),
        (RED, "v", "-."),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    x = np.arange(1, 6)
    for method, (color, marker, line) in zip(methods, styles):
        vals = [r["total"] for r in method_rows(method)]
        ax.plot(x, vals, line, color=color, marker=marker, markersize=4.0,
                linewidth=1.35, label=STATS[method]["label"])
    ax.set_title("5 次重复测量的时间序列")
    ax.set_xlabel("重复测量编号")
    ax.set_ylabel("10-run 总耗时 / 秒")
    ax.set_xticks(x)
    ax.set_ylim(0.080, 0.172)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    polish(ax)
    save(fig, "repeat_sequence.png")


def complexity_reduction():
    groups = ["预测算术量", "物品统计写入"]
    direct = np.array([2048.0, 100.0])
    final = np.array([3.0, 25.0])
    y = np.arange(len(groups))
    height = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.95))
    ax.barh(y - height / 2, direct, height, label="直接路径", color=BLUE, edgecolor=BLUE)
    ax.barh(y + height / 2, final, height, label="最终路径", color=ORANGE, edgecolor=ORANGE)
    ax.set_xscale("log")
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_title("关键热路径工作量下降")
    ax.set_xlabel("近似操作次数 / 对数坐标")
    ax.set_xlim(1.0, 4096.0)
    for yi, d, f in zip(y, direct, final):
        ax.text(d * 1.08, yi - height / 2, f"{int(d)}", va="center", fontsize=7.5)
        ax.text(f * 1.12, yi + height / 2, f"{int(f)}", va="center", fontsize=7.5)
    ax.legend(frameon=False, loc="lower right")
    polish(ax, grid_axis="x")
    save(fig, "complexity_reduction.png")


if __name__ == "__main__":
    main_method_runtime()
    tradeoff_scatter()
    sampling_sweep()
    ablation_rmse_delta()
    repeat_sequence()
    complexity_reduction()
