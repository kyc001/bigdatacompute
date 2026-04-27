"""参数扫描脚本。

职责：
- 严格通过 subprocess 调用 main.py，不 import 主程序；
- 按 INTERFACE.md 的 CLI 扫描 beta / eps / K / dtype；
- 解析 stdout 最后一行 JSON，保存 sweep_results.csv；
- 保存每次运行的 Top-100 结果，便于报告复查。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


CSV_FIELDS = [
    "run_id",
    "experiment",
    "beta",
    "eps",
    "mode",
    "K",
    "dtype",
    "peak_rss_mb",
    "wall_sec",
    "iters",
    "delta",
    "top10_signature",
    "jaccard_vs_beta085",
    "kendall_tau_intersection_vs_beta085",
    "out_path",
    "status",
]


def parse_float_list(raw: str) -> list[float]:
    """解析逗号分隔浮点数列表。"""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    """解析逗号分隔整数列表。"""

    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_last_json(stdout_text: str) -> dict[str, object] | None:
    """解析 stdout 最后一条非空行 JSON。"""

    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def signature_to_list(signature: str) -> list[int]:
    """把 top10_signature 转成节点列表。"""

    if not signature:
        return []
    return [int(item) for item in signature.split(",") if item.strip()]


def jaccard_similarity(a: Iterable[int], b: Iterable[int]) -> float:
    """计算 Top-10 集合 Jaccard 相似度。"""

    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def kendall_tau_on_intersection(reference: list[int], current: list[int]) -> float | None:
    """在两个 Top-10 的交集上计算 Kendall tau。

    若交集少于 2 个节点，返回 None。这样避免为缺失节点捏造顺序。
    """

    common = [node for node in reference if node in set(current)]
    n = len(common)
    if n < 2:
        return None

    ref_pos = {node: idx for idx, node in enumerate(reference)}
    cur_pos = {node: idx for idx, node in enumerate(current)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            left = common[i]
            right = common[j]
            ref_order = ref_pos[left] - ref_pos[right]
            cur_order = cur_pos[left] - cur_pos[right]
            if ref_order * cur_order > 0:
                concordant += 1
            elif ref_order * cur_order < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs


def run_main(
    main_path: str,
    data_path: str,
    out_path: str,
    beta: float,
    eps: float,
    mode: str,
    k_value: int,
    dtype: str,
    max_iter: int,
) -> tuple[dict[str, object] | None, str, str, int]:
    """调用 main.py 一次。"""

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
        out_path,
        "--mode",
        mode,
        "--K",
        str(k_value),
        "--dtype",
        dtype,
        "--max-iter",
        str(max_iter),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    return parse_last_json(result.stdout), result.stdout, result.stderr, int(result.returncode)


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    """写出扫描结果 CSV。"""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Sweep PageRank parameters through main.py CLI")
    parser.add_argument("--main", default="main.py", help="main.py 路径")
    parser.add_argument("--data", default="Data.txt", help="输入 Data.txt 路径")
    parser.add_argument("--out", default="experiments/sweep_results.csv", help="CSV 输出路径")
    parser.add_argument("--result-dir", default="experiments/sweep_outputs", help="Top-100 结果保存目录")
    parser.add_argument("--mode", default="csr_block", choices=("dense", "csr", "block", "csr_block"))
    parser.add_argument("--K", default="8", help="块数列表，逗号分隔")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--betas", default="0.70,0.80,0.85,0.90,0.95", help="E5 beta 列表")
    parser.add_argument("--eps-list", default="1e-6,1e-8,1e-10", help="E6 eps 列表")
    parser.add_argument("--baseline-beta", type=float, default=0.85)
    parser.add_argument("--baseline-eps", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_id = 1
    baseline_top10: list[int] | None = None

    # E5：固定 eps / K / dtype，扫描 beta。
    for beta in parse_float_list(args.betas):
        out_path = result_dir / f"E5_beta_{beta:.2f}.txt"
        summary, stdout_text, stderr_text, returncode = run_main(
            args.main,
            args.data,
            str(out_path),
            beta=beta,
            eps=args.baseline_eps,
            mode=args.mode,
            k_value=parse_int_list(args.K)[0],
            dtype=args.dtype,
            max_iter=args.max_iter,
        )
        status = "ok" if returncode == 0 and summary is not None else "error"
        top10 = signature_to_list(str(summary.get("top10_signature", ""))) if summary else []
        if abs(beta - args.baseline_beta) < 1e-12:
            baseline_top10 = top10
        rows.append(
            {
                "run_id": run_id,
                "experiment": "E5_beta",
                "beta": beta,
                "eps": args.baseline_eps,
                "mode": args.mode,
                "K": parse_int_list(args.K)[0],
                "dtype": args.dtype,
                "peak_rss_mb": summary.get("peak_rss_mb", "") if summary else "",
                "wall_sec": summary.get("wall_sec", "") if summary else "",
                "iters": summary.get("iters", "") if summary else "",
                "delta": summary.get("delta", "") if summary else "",
                "top10_signature": summary.get("top10_signature", "") if summary else "ERROR",
                "jaccard_vs_beta085": "",
                "kendall_tau_intersection_vs_beta085": "",
                "out_path": str(out_path),
                "status": status,
            }
        )
        if status != "ok":
            (result_dir / f"E5_beta_{beta:.2f}.stderr.txt").write_text(stderr_text + stdout_text, encoding="utf-8")
        run_id += 1

    if baseline_top10 is None:
        baseline_top10 = signature_to_list(str(rows[0]["top10_signature"])) if rows else []

    for row in rows:
        if row["experiment"] != "E5_beta" or row["status"] != "ok":
            continue
        top10 = signature_to_list(str(row["top10_signature"]))
        row["jaccard_vs_beta085"] = jaccard_similarity(top10, baseline_top10)
        tau = kendall_tau_on_intersection(baseline_top10, top10)
        row["kendall_tau_intersection_vs_beta085"] = "" if tau is None else tau

    # E6：固定 beta / K / dtype，扫描 eps。
    for eps in parse_float_list(args.eps_list):
        out_path = result_dir / f"E6_eps_{eps:.0e}.txt"
        summary, stdout_text, stderr_text, returncode = run_main(
            args.main,
            args.data,
            str(out_path),
            beta=args.baseline_beta,
            eps=eps,
            mode=args.mode,
            k_value=parse_int_list(args.K)[0],
            dtype=args.dtype,
            max_iter=args.max_iter,
        )
        status = "ok" if returncode == 0 and summary is not None else "error"
        top10 = signature_to_list(str(summary.get("top10_signature", ""))) if summary else []
        rows.append(
            {
                "run_id": run_id,
                "experiment": "E6_eps",
                "beta": args.baseline_beta,
                "eps": eps,
                "mode": args.mode,
                "K": parse_int_list(args.K)[0],
                "dtype": args.dtype,
                "peak_rss_mb": summary.get("peak_rss_mb", "") if summary else "",
                "wall_sec": summary.get("wall_sec", "") if summary else "",
                "iters": summary.get("iters", "") if summary else "",
                "delta": summary.get("delta", "") if summary else "",
                "top10_signature": summary.get("top10_signature", "") if summary else "ERROR",
                "jaccard_vs_beta085": jaccard_similarity(top10, baseline_top10) if status == "ok" else "",
                "kendall_tau_intersection_vs_beta085": (
                    kendall_tau_on_intersection(baseline_top10, top10) if status == "ok" else ""
                ),
                "out_path": str(out_path),
                "status": status,
            }
        )
        if status != "ok":
            (result_dir / f"E6_eps_{eps:.0e}.stderr.txt").write_text(stderr_text + stdout_text, encoding="utf-8")
        run_id += 1

    write_rows(args.out, rows)
    print(f"wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
