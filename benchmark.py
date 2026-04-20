"""benchmark.py

开发期性能测量工具：
- 用 subprocess 启动 main.py；
- 用 psutil 轮询采样 RSS 峰值；
- 解析 main.py stdout 最后一行 JSON；
- 输出固定 schema 的 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import psutil


CSV_FIELDS = [
    "run_id",
    "mode",
    "K",
    "dtype",
    "peak_rss_mb",
    "wall_sec",
    "iters",
    "top10_signature",
]


def collect_tree_rss(root: psutil.Process) -> int:
    """统计进程树 RSS，单位字节。"""

    total = 0
    try:
        processes = [root] + root.children(recursive=True)
    except psutil.Error:
        return 0

    for proc in processes:
        try:
            total += int(proc.memory_info().rss)
        except psutil.Error:
            continue
    return total


def parse_last_json(stdout_text: str) -> dict[str, object] | None:
    """解析 stdout 最后一条非空行 JSON。"""

    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def run_once(
    main_path: str,
    data_path: str,
    mode: str,
    K: int,
    dtype: str,
    interval: float,
    result_path: str,
    beta: float = 0.85,
    eps: float = 1e-8,
    max_iter: int = 200,
) -> dict[str, object]:
    """运行一次 main.py，返回一条 CSV 行。"""

    command = [
        sys.executable,
        main_path,
        "--data",
        data_path,
        "--beta",
        str(beta),
        "--eps",
        str(eps),
        "--out",
        result_path,
        "--mode",
        mode,
        "--K",
        str(K),
        "--dtype",
        dtype,
        "--max-iter",
        str(max_iter),
    ]

    start_time = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ps_process = psutil.Process(process.pid)
    peak_rss = 0

    while process.poll() is None:
        peak_rss = max(peak_rss, collect_tree_rss(ps_process))
        time.sleep(interval)

    stdout_bytes, stderr_bytes = process.communicate()
    peak_rss = max(peak_rss, collect_tree_rss(ps_process))
    elapsed = time.perf_counter() - start_time

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    _stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    summary = parse_last_json(stdout_text)

    if process.returncode != 0 or summary is None:
        return {
            "run_id": 0,
            "mode": mode,
            "K": int(K),
            "dtype": dtype,
            "peak_rss_mb": float(peak_rss / (1024 * 1024)),
            "wall_sec": float(elapsed),
            "iters": -1,
            "top10_signature": "ERROR",
        }

    json_peak = float(summary.get("peak_rss_mb", 0.0))
    sampled_peak = float(peak_rss / (1024 * 1024))
    return {
        "run_id": 0,
        "mode": mode,
        "K": int(summary.get("K", K)),
        "dtype": str(summary.get("dtype", dtype)),
        "peak_rss_mb": max(sampled_peak, json_peak),
        "wall_sec": float(summary.get("wall_sec", elapsed)),
        "iters": int(summary.get("iters", -1)),
        "top10_signature": str(summary.get("top10_signature", "ERROR")),
    }


def write_rows(csv_path: str, rows: list[dict[str, object]]) -> None:
    """写出 benchmark CSV。"""

    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Benchmark PageRank runtime and RSS peak")
    parser.add_argument("--main", required=True, help="被测 main.py 路径")
    parser.add_argument("--data", required=True, help="输入数据路径")
    parser.add_argument("--out", required=True, help="CSV 输出路径")
    parser.add_argument("--interval", type=float, default=0.05, help="RSS 采样间隔，单位秒")
    parser.add_argument("--runs", type=int, default=3, help="每个 mode 重复运行次数")
    parser.add_argument("--modes", default="csr_block", help="逗号分隔，例如 csr,csr_block")
    parser.add_argument("--K", type=int, default=8, help="块数")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows: list[dict[str, object]] = []
    run_id = 1

    with tempfile.TemporaryDirectory(prefix="pagerank_benchmark_") as tmp_dir:
        for mode in modes:
            for _ in range(args.runs):
                result_path = os.path.join(tmp_dir, f"{mode}_{run_id:03d}.txt")
                row = run_once(
                    main_path=args.main,
                    data_path=args.data,
                    mode=mode,
                    K=args.K,
                    dtype=args.dtype,
                    interval=args.interval,
                    result_path=result_path,
                    beta=args.beta,
                    eps=args.eps,
                    max_iter=args.max_iter,
                )
                row["run_id"] = run_id
                rows.append(row)
                run_id += 1

    write_rows(args.out, rows)

    for row in rows:
        print(
            f"run_id={row['run_id']} "
            f"mode={row['mode']} "
            f"peak_rss_mb={float(row['peak_rss_mb']):.3f} "
            f"wall_sec={float(row['wall_sec']):.6f} "
            f"iters={row['iters']}"
        )


if __name__ == "__main__":
    main()
