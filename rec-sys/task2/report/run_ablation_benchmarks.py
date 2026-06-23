import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "rec-sys" / "task2" / "report"
SRC_DIR = REPORT / "ablation_sources"
DATA = ROOT / "rec-sys" / "task2" / "track1" / "secure_data_full_1024" / "judge_data.bin"
RUNNER = ROOT / "rec-sys" / "task2" / "runner" / "cpp" / "main.cpp"
SCANNER = ROOT / "rec-sys" / "task2" / "scripts" / "scan_cpp.py"
METHOD_OUT = REPORT / "method_benchmark_results.csv"
THREAD_OUT = REPORT / "thread_benchmark_results.csv"

METHODS = [
    ("final", "最终 stride=4", "final.cpp", "main"),
    ("dense_item", "完整物品统计", "dense_item.cpp", "sampling"),
    ("stride2", "物品 stride=2", "stride2.cpp", "sampling"),
    ("stride16", "物品 stride=16", "stride16.cpp", "sampling"),
    ("no_segment_prior", "无用户分段先验", "no_segment_prior.cpp", "ablation"),
    ("no_count_terms", "无计数形状项", "no_count_terms.cpp", "ablation"),
    ("no_user_residual", "无用户残差", "no_user_residual.cpp", "ablation"),
    ("no_item_residual", "无物品残差", "no_item_residual.cpp", "ablation"),
    ("item_only", "仅物品残差", "item_only.cpp", "baseline"),
    ("constant", "全局均值", "constant.cpp", "baseline"),
]

THREADS = [1, 2, 4, 8, 16]


def run(cmd, cwd=None, timeout=120, env=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(map(str, cmd))}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def compile_source(source: Path, work: Path, extra_flags=None) -> Path:
    run(["python3", str(SCANNER), str(source)], timeout=60)
    shutil.copy2(source, work / "solution.cpp")
    shutil.copy2(RUNNER, work / "main.cpp")
    exe = work / "bench"
    cmd = ["g++", "-O3", "-std=c++17", "-march=native", "-fopenmp"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["main.cpp", "-o", str(exe)])
    run(cmd, cwd=work, timeout=120)
    return exe


def parse_payload(stdout: str):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"runner did not emit JSON:\n{stdout}")


def run_payload(exe: Path, work: Path, runs=5):
    proc = run([str(exe), str(DATA), "0.001", str(runs)], cwd=work, timeout=900)
    return parse_payload(proc.stdout)


def payload_row(payload):
    return {
        "total": f"{float(payload['time_sec']):.6f}",
        "pre_rmse": f"{float(payload['rmse_base']):.6f}",
        "post_rmse": f"{float(payload['rmse']):.6f}",
        "valid": str(bool(payload["valid"])),
        "time_runs": ";".join(f"{float(x):.6f}" for x in payload.get("time_runs", [])),
    }


def benchmark_methods():
    rows = []
    for method_id, label, filename, group in METHODS:
        source = SRC_DIR / filename
        print(f"[method] {method_id}: {label}")
        with tempfile.TemporaryDirectory(prefix=f"report_{method_id}_") as tmp:
            work = Path(tmp)
            exe = compile_source(source, work)
            payload = run_payload(exe, work, runs=5)
            row = {
                "method": method_id,
                "label": label,
                "group": group,
                **payload_row(payload),
            }
            rows.append(row)
            print(f"  total={row['total']} rmse={row['post_rmse']} valid={row['valid']}")
    with METHOD_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "label", "group", "total", "pre_rmse", "post_rmse", "valid", "time_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {METHOD_OUT}")


def benchmark_threads():
    rows = []
    source = SRC_DIR / "final.cpp"
    for threads in THREADS:
        print(f"[thread] {threads}")
        with tempfile.TemporaryDirectory(prefix=f"report_thread_{threads}_") as tmp:
            work = Path(tmp)
            exe = compile_source(source, work, extra_flags=[f"-DTASK2_PREDICTION_THREADS={threads}"])
            payload = run_payload(exe, work, runs=5)
            row = {
                "threads": str(threads),
                "label": f"{threads} 线程",
                **payload_row(payload),
            }
            rows.append(row)
            print(f"  total={row['total']} rmse={row['post_rmse']} valid={row['valid']}")
    with THREAD_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threads", "label", "total", "pre_rmse", "post_rmse", "valid", "time_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {THREAD_OUT}")


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    benchmark_methods()
    benchmark_threads()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
