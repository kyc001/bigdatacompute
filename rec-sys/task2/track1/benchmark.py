#!/usr/bin/env python3
"""
增量SVD矩阵分解 - 学生自测脚本

用于本地测量 solution.py / solution.cpp 的内存峰值占用和运行耗时。
测量结果仅供本机优化参考，与 OJ 最终评测结果可能存在差异。

用法:
    python benchmark.py                              # 自动选择 students/solution.py 或 solution.cpp
    python benchmark.py --solution my_solution.py    # 指定 Python 解法文件
    python benchmark.py --solution my_solution.cpp   # 指定 C++ 解法文件
    python benchmark.py --language cpp               # 强制按 C++ 解法测试
    python benchmark.py --data-dir /path/to/data     # 指定数据目录
    python benchmark.py --benchmark-runs 3           # 跑多轮
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:
    resource = None


ALLOWED_IMPORT_ROOTS = {
    "numpy",
    "scipy",
    "math",
    "random",
    "threading",
    "concurrent",
    "collections",
    "itertools",
    "functools",
    "operator",
    "typing",
}
DANGEROUS_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
DANGEROUS_ATTRS = {"system", "popen", "fork", "execv", "execve", "spawn", "remove", "unlink", "rmdir"}
FILE_IO_ATTRS = {
    "load", "save", "savez", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "memmap", "loadmat", "savemat",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def scan_solution(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                raise ValueError(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in DANGEROUS_ATTRS:
                raise ValueError(f"forbidden attribute call: {node.func.attr}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FILE_IO_ATTRS:
                raise ValueError(f"forbidden file I/O helper: {node.func.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "/data" in lowered or ".npy" in lowered or ".bin" in lowered:
                raise ValueError("forbidden data path literal")


def load_solution(path: Path):
    scan_solution(path)
    spec = importlib.util.spec_from_file_location("student_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_solution"] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "IncrementalSVD", None)
    if cls is None:
        raise AttributeError("solution.py must define IncrementalSVD")
    return cls


def get_peak_memory_mb() -> float:
    """获取当前进程的峰值物理内存占用 (RSS, 单位 MB)。"""
    if resource is None:
        return 0.0
    # Linux: ru_maxrss 单位是 KB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_kb = usage.ru_maxrss
    return peak_kb / 1024.0


def get_child_peak_memory_mb() -> float:
    """获取已结束子进程的峰值物理内存占用 (RSS, 单位 MB)。"""
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_maxrss / 1024.0


def infer_language(solution_path: Path, language: str) -> str:
    if language != "auto":
        return language
    suffix = solution_path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".cpp", ".cc", ".cxx"}:
        return "cpp"
    raise ValueError("无法从解法文件后缀判断语言，请使用 --language python 或 --language cpp")


def default_solution_path(language: str) -> Path:
    students_dir = Path(__file__).parent
    if language == "python":
        return students_dir / "solution.py"
    if language == "cpp":
        return students_dir / "solution.cpp"

    py_path = students_dir / "solution.py"
    cpp_path = students_dir / "solution.cpp"
    if py_path.exists():
        return py_path
    if cpp_path.exists():
        return cpp_path
    return py_path


def required_data_files(language: str) -> tuple[str, ...]:
    if language == "cpp":
        return ("judge_data.bin",)
    return ("meta.json", "P.npy", "Q.npy", "global_mean.npy", "incremental.npy", "test.npy")


def default_data_dir(language: str) -> Path:
    candidates = [
        Path(__file__).parent / "secure_data_full_1024",
    ]
    required = required_data_files(language)
    for candidate in candidates:
        if all((candidate / fname).exists() for fname in required):
            return candidate
    return candidates[0]


def cpp_compile_command(source: str, output: Path) -> list[str]:
    cxx = os.environ.get("TASK2_CPP_CXX", "g++")
    march = os.environ.get("TASK2_CPP_MARCH", "native").strip()
    extra_flags = shlex.split(os.environ.get("TASK2_CPP_FLAGS", ""))

    command = [
        cxx,
        "-O3",
        "-std=c++17",
    ]
    if march:
        command.append(f"-march={march}")
    command.extend(extra_flags)
    command.extend([
        "-fopenmp",
        source,
        "-o",
        str(output),
    ])
    return command


def check_files(data_dir: Path, filenames: tuple[str, ...]) -> None:
    for fname in filenames:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(f"数据目录缺少文件: {data_dir / fname}")


def summarize_data_dir(data_dir: Path) -> None:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    latent_dim = int(meta["latent_dim"])
    batch_size = int(meta.get("batch_size", 100_000))
    print(f"    用户数: {num_users}, 物品数: {num_items}, 隐维度: {latent_dim}")
    print(f"    每批大小: {batch_size}")


def run_cpp_benchmark(args: argparse.Namespace, data_dir: Path, solution_path: Path) -> None:
    check_files(data_dir, ("judge_data.bin",))
    runner_path = REPO_ROOT / "runner" / "cpp" / "main.cpp"
    scanner_path = REPO_ROOT / "scripts" / "scan_cpp.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"找不到 C++ runner: {runner_path}")
    if not scanner_path.exists():
        raise FileNotFoundError(f"找不到 C++ 扫描器: {scanner_path}")

    print(f"[1/4] 检查数据 ({data_dir}) ...")
    summarize_data_dir(data_dir)
    judge_data_mb = (data_dir / "judge_data.bin").stat().st_size / (1024 ** 2)
    print(f"    judge_data.bin: {judge_data_mb:.1f} MB")
    print()

    print(f"[2/4] 检查并编译 C++ 解法 ({solution_path}) ...")
    scan_proc = subprocess.run(
        [sys.executable, str(scanner_path), str(solution_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if scan_proc.returncode != 0:
        detail = (scan_proc.stdout or scan_proc.stderr).strip()
        raise RuntimeError(f"C++ 安全扫描失败: {detail}")

    with tempfile.TemporaryDirectory(prefix="svd_benchmark_cpp_") as tmp_name:
        tmp_dir = Path(tmp_name)
        shutil.copy2(solution_path, tmp_dir / "solution.cpp")
        shutil.copy2(runner_path, tmp_dir / "main.cpp")
        exe_path = tmp_dir / "benchmark_cpp"
        compile_cmd = cpp_compile_command("main.cpp", exe_path)
        compile_start = time.perf_counter()
        compile_proc = subprocess.run(
            compile_cmd,
            cwd=tmp_dir,
            text=True,
            capture_output=True,
            timeout=args.compile_timeout,
            check=False,
        )
        compile_elapsed = time.perf_counter() - compile_start
        if compile_proc.returncode != 0:
            detail = (compile_proc.stderr or compile_proc.stdout).strip()
            raise RuntimeError(f"C++ 编译失败:\n{detail}")
        print(f"    编译完成: {compile_elapsed:.3f} 秒")
        print()

        benchmark_runs = max(1, args.benchmark_runs)
        if args.skip_rmse:
            print("    提示: C++ runner 与 OJ 一致，会始终计算 RMSE；--skip-rmse 仅对 Python 生效。")
        print(f"[3/4] 运行 C++ 基准测试 ({benchmark_runs} 轮) ...")
        print("-" * 60)

        run_cmd = [
            str(exe_path),
            str(data_dir / "judge_data.bin"),
            str(args.epsilon),
            str(benchmark_runs),
        ]
        run_start = time.perf_counter()
        run_proc = subprocess.run(
            run_cmd,
            cwd=tmp_dir,
            text=True,
            capture_output=True,
            timeout=args.run_timeout,
            check=False,
        )
        wall_elapsed = time.perf_counter() - run_start
        output_lines = [line for line in run_proc.stdout.splitlines() if line.strip()]
        if not output_lines:
            detail = run_proc.stderr.strip()
            raise RuntimeError(f"C++ runner 未输出结果:\n{detail}")
        try:
            payload = json.loads(output_lines[-1])
        except json.JSONDecodeError as exc:
            detail = (run_proc.stdout + "\n" + run_proc.stderr).strip()
            raise RuntimeError(f"C++ runner 输出不是合法 JSON: {exc}\n{detail}") from exc
        if run_proc.returncode != 0 or payload.get("status") != "success":
            raise RuntimeError(f"C++ runner 失败: {payload.get('error', payload)}")

        elapsed_runs = [float(x) for x in payload.get("time_runs", [])]
        total_time = float(payload.get("time_sec", sum(elapsed_runs)))
        if not elapsed_runs:
            elapsed_runs = [total_time]
        best_time = min(elapsed_runs)
        avg_time = total_time / len(elapsed_runs)
        rmse_base = float(payload["rmse_base"])
        rmse_new = float(payload["rmse"])
        valid = bool(payload["valid"])
        peak_mem_mb = get_child_peak_memory_mb()

        for idx, elapsed in enumerate(elapsed_runs, start=1):
            print(f"    第 {idx} 轮: {elapsed:.3f} 秒")

    print()
    print("[4/4] 测试结果汇总")
    print("=" * 60)
    print()

    print(f"  运行语言:        C++")
    print(f"  运行轮数:        {len(elapsed_runs)}")
    print(f"  总耗时:          {total_time:.3f} 秒")
    if len(elapsed_runs) > 1:
        print(f"  最优单轮:        {best_time:.3f} 秒")
        print(f"  平均单轮:        {avg_time:.3f} 秒")
    print(f"  各轮耗时:        {', '.join(f'{t:.3f}' for t in elapsed_runs)}")
    print(f"  墙钟耗时:        {wall_elapsed:.3f} 秒 (含 runner 额外开销)")
    print()

    print(f"  子进程峰值内存:  {peak_mem_mb:.1f} MB (RSS，可能包含编译器峰值)")
    print()

    improvement = rmse_base - rmse_new
    print(f"  更新前 RMSE:     {rmse_base:.6f}")
    print(f"  更新后 RMSE:     {rmse_new:.6f}")
    print(f"  RMSE 下降:       {improvement:.6f} (阈值: {args.epsilon})")
    print(f"  结果有效性:      {'有效 PASS' if valid else '无效 FAIL'}")
    if not math.isfinite(rmse_new):
        print(f"    原因: RMSE 为非有限值 (NaN/Inf)")
    elif improvement < args.epsilon:
        print(f"    原因: RMSE 下降不足 {args.epsilon}")
    print()

    baseline_cpp = 54.0
    baseline_python = 900.0
    print(f"  参考基线 (C++):    {baseline_cpp:.0f} 秒")
    print(f"  参考基线 (Python): {baseline_python:.0f} 秒")
    ratio = best_time / baseline_cpp * 100
    print(f"  你的最优耗时:      {best_time:.3f} 秒 (C++ 基线的 {ratio:.1f}%)")
    print()

    print("=" * 60)
    print("  注意事项:")
    print("  - 以上数据仅供本机优化参考")
    print("  - C++ 自测会在临时目录复制 solution.cpp 并本地编译执行")
    print("  - OJ 评测在 Docker 容器中运行 (最多 16 核, 4GB 内存)")
    print("  - OJ 评测取 10 轮总耗时，本脚本默认只跑 1 轮，注意参考基线是10轮结果")
    print("  - 如需更接近 OJ 结果，可用 --benchmark-runs 10")
    print("=" * 60)


def rmse(model, ratings: np.ndarray) -> float:
    if ratings.size == 0:
        return float("inf")
    sqerr = 0.0
    for u_f, i_f, r_f in ratings:
        pred = float(model.predict(int(u_f), int(i_f)))
        if not math.isfinite(pred):
            return float("nan")
        err = float(r_f) - pred
        sqerr += err * err
    return math.sqrt(sqerr / ratings.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="增量SVD本地自测脚本 - 测量内存峰值和运行耗时"
    )
    parser.add_argument(
        "--solution",
        default=None,
        help="解法文件路径，支持 solution.py / solution.cpp (默认: 自动选择 students/solution.py 或 solution.cpp)",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "python", "cpp"),
        default="auto",
        help="解法语言 (默认: auto，根据文件后缀判断)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="数据目录路径 (默认: secure_data_full_1024/)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="基准测试轮数 (默认: 1，设为多轮可观察稳定性)",
    )
    parser.add_argument(
        "--skip-rmse",
        action="store_true",
        help="跳过 RMSE 计算（加速测试，仅看速度和内存）",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="RMSE 下降阈值 (默认: 0.001)",
    )
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=60.0,
        help="C++ 编译超时时间，单位秒 (默认: 60)",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=600.0,
        help="C++ 运行超时时间，单位秒 (默认: 600)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  增量SVD矩阵分解 - 本地自测工具")
    print("=" * 60)
    print()

    solution_path = (
        Path(args.solution).resolve() if args.solution else default_solution_path(args.language).resolve()
    )
    language = infer_language(solution_path, args.language)
    data_dir = (Path(args.data_dir) if args.data_dir else default_data_dir(language)).resolve()

    if not solution_path.exists():
        print(f"[错误] 找不到解法文件: {solution_path}")
        sys.exit(1)

    try:
        if language == "cpp":
            run_cpp_benchmark(args, data_dir, solution_path)
            return

        check_files(data_dir, required_data_files(language))

        print(f"[1/4] 加载数据 ({data_dir}) ...")
        with (data_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        base_p = np.load(data_dir / "P.npy").astype(np.float32, copy=False)
        base_q = np.load(data_dir / "Q.npy").astype(np.float32, copy=False)
        global_mean = float(np.load(data_dir / "global_mean.npy"))
        incremental = np.load(data_dir / "incremental.npy", mmap_mode="r")
        test = np.load(data_dir / "test.npy", mmap_mode="r")

        num_users = int(meta["num_users"])
        num_items = int(meta["num_items"])
        latent_dim = int(meta["latent_dim"])
        batch_size = int(meta.get("batch_size", 100_000))

        p_mem_mb = base_p.nbytes / (1024 ** 2)
        q_mem_mb = base_q.nbytes / (1024 ** 2)
        inc_mem_mb = incremental.nbytes / (1024 ** 2)
        test_mem_mb = test.nbytes / (1024 ** 2)
        print(f"    用户数: {num_users}, 物品数: {num_items}, 隐维度: {latent_dim}")
        print(f"    增量批次数: {math.ceil(incremental.shape[0] / batch_size)}, 每批大小: {batch_size}")
        print(f"    矩阵 P: {p_mem_mb:.1f} MB, Q: {q_mem_mb:.1f} MB")
        print(f"    增量数据: {inc_mem_mb:.1f} MB, 测试数据: {test_mem_mb:.1f} MB")
        print()

        print(f"[2/4] 加载解法 ({solution_path}) ...")
        cls = load_solution(solution_path)
        print(f"    解法类: {cls.__name__}")
        print()

        benchmark_runs = max(1, args.benchmark_runs)
        print(f"[3/4] 运行基准测试 ({benchmark_runs} 轮) ...")
        print("-" * 60)

        # 记录加载数据后的基线内存
        mem_before_load = get_peak_memory_mb()

        elapsed_runs = []
        rmse_base = None
        rmse_new = None

        for run_id in range(benchmark_runs):
            p = base_p.copy()
            q = base_q.copy()
            model = cls(num_users, num_items, latent_dim, global_mean)
            model.load_base_model(p, q)

            if run_id == 0 and not args.skip_rmse:
                print(f"    计算更新前 RMSE ...")
                rmse_base = rmse(model, test)
                print(f"    更新前 RMSE = {rmse_base:.6f}")

            t_start = time.perf_counter()
            for offset in range(0, incremental.shape[0], batch_size):
                model.update(incremental[offset : offset + batch_size])
            t_elapsed = time.perf_counter() - t_start

            elapsed_runs.append(t_elapsed)
            print(f"    第 {run_id + 1} 轮: {t_elapsed:.3f} 秒")

            if run_id == 0 and not args.skip_rmse:
                print(f"    计算更新后 RMSE ...")
                rmse_new = rmse(model, test)
                print(f"    更新后 RMSE = {rmse_new:.6f}")

        # 获取峰值内存
        peak_mem_mb = get_peak_memory_mb()

        print()
        print("[4/4] 测试结果汇总")
        print("=" * 60)
        print()

        # 时间统计
        total_time = sum(elapsed_runs)
        best_time = min(elapsed_runs)
        avg_time = total_time / len(elapsed_runs)

        print(f"  运行语言:        Python")
        print(f"  运行轮数:        {benchmark_runs}")
        print(f"  总耗时:          {total_time:.3f} 秒")
        if benchmark_runs > 1:
            print(f"  最优单轮:        {best_time:.3f} 秒")
            print(f"  平均单轮:        {avg_time:.3f} 秒")
        print(f"  各轮耗时:        {', '.join(f'{t:.3f}' for t in elapsed_runs)}")
        print()

        # 内存统计
        print(f"  进程峰值内存:    {peak_mem_mb:.1f} MB (RSS)")
        print(f"  矩阵 P+Q 占用:  {p_mem_mb + q_mem_mb:.1f} MB")
        print()

        # RMSE 统计
        if rmse_base is not None and rmse_new is not None:
            improvement = rmse_base - rmse_new
            valid = math.isfinite(rmse_new) and improvement >= args.epsilon
            print(f"  更新前 RMSE:     {rmse_base:.6f}")
            print(f"  更新后 RMSE:     {rmse_new:.6f}")
            print(f"  RMSE 下降:       {improvement:.6f} (阈值: {args.epsilon})")
            print(f"  结果有效性:      {'有效 PASS' if valid else '无效 FAIL'}")
            if not math.isfinite(rmse_new):
                print(f"    原因: RMSE 为非有限值 (NaN/Inf)")
            elif improvement < args.epsilon:
                print(f"    原因: RMSE 下降不足 {args.epsilon}")
            print()

        # 与基线对比
        baseline_cpp = 54.0
        baseline_python = 900.0
        print(f"  参考基线 (C++):    {baseline_cpp:.0f} 秒")
        print(f"  参考基线 (Python): {baseline_python:.0f} 秒")
        if best_time < baseline_python:
            ratio = best_time / baseline_python * 100
            print(f"  你的最优耗时:      {best_time:.3f} 秒 (Python 基线的 {ratio:.1f}%)")
        else:
            print(f"  你的最优耗时:      {best_time:.3f} 秒 (超过 Python 基线)")
        print()

        print("=" * 60)
        print("  注意事项:")
        print("  - 以上数据仅供本机优化参考")
        print("  - 因处理器型号、核心数、缓存等硬件差异，")
        print("    与 OJ 最终评测服务器的运行时间存在差异")
        print("  - OJ 评测在 Docker 容器中运行 (最多 16 核, 4GB 内存)")
        print("  - OJ 评测取 10 轮总耗时，本脚本默认只跑 1 轮，注意参考基线是10轮结果")
        print("  - 如需更接近 OJ 结果，可用 --benchmark-runs 10")
        print("=" * 60)

    except Exception as exc:
        print(f"\n[错误] {exc}")
        traceback.print_exc(limit=5)
        sys.exit(1)


if __name__ == "__main__":
    main()
