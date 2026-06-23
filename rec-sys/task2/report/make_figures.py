import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#5B78A7"
ORANGE = "#E68B32"
GREEN = "#5F8F73"
RED = "#B85C5A"
PURPLE = "#7A6EA8"
GRAY = "#707070"
LIGHT = "#E9E9E9"
BLACK = "#111111"
GRID = "#D8D8D8"

LABEL = {
    "global_mean": "全局均值",
    "constant": "全局均值",
    "k2_count": "K2 计数校准",
    "factor128": "128 参因子化",
    "thread_local": "线程本地统计",
    "segment_prior": "用户分段先验",
    "touched_refresh": "触达刷新",
    "online_no_table": "在线采样",
    "final": "最终 stride=4",
    "dense_item": "完整物品统计",
    "stride2": "物品 stride=2",
    "stride4": "物品 stride=4",
    "stride16": "物品 stride=16",
    "no_segment_prior": "无用户分段先验",
    "no_count_terms": "无计数形状项",
    "no_user_residual": "无用户残差",
    "no_item_residual": "无物品残差",
    "item_only": "仅物品残差",
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 8.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.95,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def read_csv(name):
    with (ROOT / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_optional(name):
    path = ROOT / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


METHOD = {r["method"]: r for r in read_csv("method_benchmark_results.csv")}
PROCESS = {r["method"]: r for r in read_csv("process_benchmark_results.csv")}
PROFILE = {r["method"]: r for r in read_csv("stage_profile_results.csv")}
THREAD = {int(r["threads"]): r for r in read_csv("thread_benchmark_results.csv")}
THREAD_ROBUST_ROWS = read_csv_optional("thread_benchmark_outlier_results.csv")
THREAD_ROBUST = {int(r["threads"]): r for r in THREAD_ROBUST_ROWS}


def polish(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.72, alpha=0.58)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.2, width=0.9, color=BLACK)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def optimization_path():
    order = [
        "global_mean", "k2_count", "factor128", "thread_local",
        "segment_prior", "touched_refresh", "online_no_table", "final",
    ]
    x = np.arange(len(order))
    rmse = np.array([as_float(PROCESS[m], "post_rmse") for m in order])
    total = np.array([as_float(METHOD["final"], "total") if m == "final" else as_float(PROCESS[m], "total") for m in order])
    labels = [LABEL[m].replace(" ", "\n") for m in order]

    fig, ax1 = plt.subplots(figsize=(7.6, 4.35))
    ax2 = ax1.twinx()
    ax1.plot(x, rmse, color=BLUE, marker="o", linewidth=1.8, markersize=4.5, label="RMSE")
    ax2.plot(x, total, color=ORANGE, marker="s", linewidth=1.65, markersize=4.2, label="5-run 耗时")
    ax1.scatter([len(order) - 1], [rmse[-1]], s=100, marker="*", color=RED, zorder=4, label="最终")
    ax1.set_title("优化路径：先找有效信号，再压缩热点")
    ax1.set_ylabel("更新后 RMSE", color=BLUE)
    ax2.set_ylabel("5-run 总耗时 / s", color=ORANGE)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0.910, 1.032)
    ax2.set_ylim(0.0, max(total) * 1.18)
    ax1.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax1.grid(axis="y", color=GRID, linewidth=0.72, alpha=0.58)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.annotate("精度显著改善\n但刷新成本偏高", xy=(4, rmse[4]), xytext=(3.2, 0.944),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY), fontsize=7.4)
    ax1.annotate("在线采样 + 查表\n保留精度并降成本", xy=(7, rmse[7]), xytext=(5.7, 0.955),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY), fontsize=7.4)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    save(fig, "optimization_path.png")


def pareto_frontier():
    rows = []
    for key, row in METHOD.items():
        if key == "stride4":
            continue
        rows.append((key, as_float(row, "total"), as_float(row, "post_rmse"), row["valid"].lower() == "true", "method"))
    for key, row in PROCESS.items():
        if key not in {"global_mean", "final"}:
            rows.append((key, as_float(row, "total"), as_float(row, "post_rmse"), row["valid"].lower() == "true", "process"))

    fig, ax = plt.subplots(figsize=(7.75, 4.70))
    for key, x, y, valid, group in rows:
        if not valid:
            color, marker, size, alpha = "#AAAAAA", "x", 64, 0.9
        elif key == "final":
            color, marker, size, alpha = RED, "*", 125, 1.0
        elif key in {"dense_item", "stride2", "stride4", "stride16"}:
            color, marker, size, alpha = ORANGE, "D", 50, 0.95
        elif key.startswith("no_") or key == "item_only":
            color, marker, size, alpha = GRAY, "o", 42, 0.72
        else:
            color, marker, size, alpha = BLUE, "o", 48, 0.9
        if marker == "x":
            ax.scatter(x, y, s=size, marker=marker, color=color, linewidth=0.85, alpha=alpha, zorder=3)
        else:
            edge = color if marker == "*" else BLACK
            ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor=edge,
                       linewidth=0.65, alpha=alpha, zorder=3)

    valid_points = sorted([(x, y, key) for key, x, y, valid, _ in rows if valid], key=lambda v: v[0])
    frontier = []
    best = float("inf")
    for x, y, key in valid_points:
        if y < best:
            frontier.append((x, y, key))
            best = y
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], color=GREEN, linewidth=1.25,
                linestyle="--", label="本地 Pareto 前沿", zorder=2)

    callouts = {
        "constant": (0.0170, 1.0200),
        "stride16": (0.0150, 0.9324),
        "final": (0.0275, 0.9188),
        "k2_count": (0.0360, 0.9358),
        "dense_item": (0.0450, 0.9178),
        "touched_refresh": (0.0555, 0.9130),
        "segment_prior": (0.0740, 0.9205),
    }
    point_map = {key: (x, y) for key, x, y, _, _ in rows}
    for key, text_pos in callouts.items():
        if key not in point_map:
            continue
        arrow = dict(arrowstyle="-", lw=0.65, color=GRAY, shrinkA=2, shrinkB=3)
        ax.annotate(LABEL[key], xy=point_map[key], xytext=text_pos, fontsize=6.8,
                    ha="left", va="center", arrowprops=arrow,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.05))
    ax.set_title("候选架构的本地 Pareto 分布")
    ax.set_xlabel("5-run 总耗时 / s")
    ax.set_ylabel("更新后 RMSE")
    ax.set_xlim(0.0115, 0.0830)
    ax.set_ylim(0.906, 1.034)
    ax.legend(frameon=False, loc="upper right")
    polish(ax)
    save(fig, "pareto_frontier.png")


def update_predict_breakdown():
    order = ["dense_item", "final", "stride16", "no_count_terms"]
    labels = [LABEL[m].replace("物品 ", "物品\n").replace("最终 ", "最终\n") for m in order]
    update = np.array([as_float(PROFILE[m], "update_total") for m in order])
    predict = np.array([as_float(PROFILE[m], "predict_total") for m in order])
    rmse = np.array([as_float(PROFILE[m], "post_rmse") for m in order])
    x = np.arange(len(order))

    fig, ax1 = plt.subplots(figsize=(7.45, 4.2))
    ax2 = ax1.twinx()
    ax1.bar(x, update, color=BLUE, edgecolor=BLUE, label="update")
    ax1.bar(x, predict, bottom=update, color=ORANGE, edgecolor=ORANGE, label="predict")
    ax2.plot(x, rmse, color=RED, marker="o", linewidth=1.35, label="RMSE")
    ax1.set_title("端到端耗时拆分：优化主要发生在 update")
    ax1.set_ylabel("5-run 分段耗时 / s")
    ax2.set_ylabel("更新后 RMSE", color=RED)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0.0, max(update + predict) * 1.20)
    ax2.set_ylim(0.910, 0.982)
    ax2.tick_params(axis="y", colors=RED)
    for xi, u, p in zip(x, update, predict):
        ax1.text(xi, u + p + 0.0012, f"{u/(u+p):.0%}", ha="center", va="bottom", fontsize=7.2)
    lines = ax1.patches[:1] + ax1.patches[len(order):len(order)+1] + ax2.get_lines()
    ax1.legend(lines, ["update", "predict", "RMSE"], frameon=False, loc="upper right")
    polish(ax1)
    ax2.spines["top"].set_visible(False)
    save(fig, "update_predict_breakdown.png")


def sampling_efficiency():
    methods = ["dense_item", "stride2", "final", "stride16"]
    ratio = np.array([1.0, 0.5, 0.25, 0.0625])
    rmse = np.array([as_float(METHOD[m], "post_rmse") for m in methods])
    total = np.array([as_float(METHOD[m], "total") for m in methods])
    sizes = 360 * total / total.max() + 65
    fig, ax = plt.subplots(figsize=(7.05, 4.25))
    ax.scatter(ratio, rmse, s=sizes, color=BLUE, edgecolor=BLACK, linewidth=0.65, alpha=0.9)
    ax.plot(ratio, rmse, color=BLUE, linewidth=1.25, alpha=0.75)
    offsets = {
        "dense_item": (-0.10, 0.0016, "right", "bottom"),
        "stride2": (-0.02, -0.0015, "right", "top"),
        "final": (-0.01, 0.0014, "right", "bottom"),
        "stride16": (0.02, 0.0015, "left", "bottom"),
    }
    for m, x, y in zip(methods, ratio, rmse):
        dx, dy, ha, va = offsets[m]
        ax.text(x + dx * x, y + dy, LABEL[m], ha=ha, va=va, fontsize=7.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratio, ["完整", "1/2", "1/4", "1/16"])
    ax.set_title("物品侧采样强度：更少写入与 RMSE 余量的交换")
    ax.set_xlabel("相对物品更新写入量")
    ax.set_ylabel("更新后 RMSE")
    ax.set_xlim(0.052, 1.30)
    ax.set_ylim(0.9115, 0.934)
    polish(ax)
    save(fig, "sampling_efficiency.png")


def component_loss():
    methods = ["no_item_residual", "no_count_terms", "no_segment_prior", "no_user_residual", "item_only"]
    base = as_float(METHOD["final"], "post_rmse")
    deltas = np.array([as_float(METHOD[m], "post_rmse") - base for m in methods])
    labels = [LABEL[m] for m in methods]
    order = np.argsort(deltas)
    deltas = deltas[order]
    labels = [labels[i] for i in order]
    colors = [ORANGE if "物品" in lab else BLUE for lab in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(labels))
    ax.hlines(y, 0, deltas, color=colors, linewidth=5.5, alpha=0.9)
    ax.scatter(deltas, y, s=38, color=colors, edgecolor=BLACK, linewidth=0.55, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_title("组件贡献：移除后 RMSE 损失")
    ax.set_xlabel("相对最终方案的 RMSE 增量")
    ax.set_xlim(0.0, max(deltas) * 1.18)
    for yi, d in zip(y, deltas):
        ax.text(d + 0.002, yi, f"+{d:.4f}", va="center", fontsize=7.4)
    polish(ax, grid_axis="x")
    save(fig, "component_loss.png")


def thread_scaling():
    robust = bool(THREAD_ROBUST)
    rows = THREAD_ROBUST if robust else THREAD
    xs = np.array(sorted(rows))
    times = np.array([
        as_float(rows[int(x)], "filtered_mean" if robust else "total")
        for x in xs
    ])
    speedup = times[0] / times
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(xs, speedup, color=BLUE, marker="o", linewidth=1.65, label="相对 1 线程加速")
    ax.axhline(1.0, color=BLACK, linewidth=0.85)
    ax.fill_between(xs, speedup, 1.0, where=speedup >= 1.0, color=BLUE, alpha=0.12)
    ax.fill_between(xs, speedup, 1.0, where=speedup < 1.0, color=RED, alpha=0.10)
    for x, s, t in zip(xs, speedup, times):
        text = f"{t * 1000:.1f} ms" if robust else f"{t:.3f}s"
        place_above = s >= 1.0 or s < 0.60
        ax.text(x, s + (0.06 if place_above else -0.08), text, ha="center",
                va="bottom" if place_above else "top", fontsize=7.0)
    ax.set_title("线程数不是单调收益：短预测路径被调度开销主导")
    ax.set_xlabel("预测线程数")
    ax.set_ylabel("相对 1 线程速度")
    ax.set_xticks(xs)
    ax.set_ylim(0.45, max(speedup) * 1.18)
    polish(ax)
    save(fig, "thread_scaling.png")


def complexity_sources():
    categories = ["预测热路径", "物品更新写入", "用户更新写入", "计数函数"]
    direct = np.array([2048.0, 100.0, 100.0, 100.0])
    final = np.array([3.0, 25.0, 10.0, 1.0])
    y = np.arange(len(categories))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.25, 4.0))
    ax.barh(y - height / 2, direct, height, color=BLUE, edgecolor=BLUE, label="直接/早期路径")
    ax.barh(y + height / 2, final, height, color=ORANGE, edgecolor=ORANGE, label="最终路径")
    ax.set_xscale("log")
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set_title("复杂度来源：减少高频路径，而不是只调参数")
    ax.set_xlabel("近似操作或写入次数 / 对数坐标")
    ax.set_xlim(0.8, 4096.0)
    for yi, d, f in zip(y, direct, final):
        ax.text(d * 1.08, yi - height / 2, f"{d:g}", va="center", fontsize=7.2)
        ax.text(f * 1.12, yi + height / 2, f"{f:g}", va="center", fontsize=7.2)
    ax.legend(frameon=False, loc="lower right")
    polish(ax, grid_axis="x")
    save(fig, "complexity_sources.png")


if __name__ == "__main__":
    optimization_path()
    pareto_frontier()
    update_predict_breakdown()
    sampling_efficiency()
    component_loss()
    thread_scaling()
    complexity_sources()
