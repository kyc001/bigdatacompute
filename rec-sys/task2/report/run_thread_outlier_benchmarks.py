import csv
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "rec-sys" / "task2" / "report"
SOURCE = ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp"
DATA = ROOT / "rec-sys" / "task2" / "track1" / "secure_data_full_1024" / "judge_data.bin"
RUNNER = ROOT / "rec-sys" / "task2" / "runner" / "cpp" / "main.cpp"
SCANNER = ROOT / "rec-sys" / "task2" / "scripts" / "scan_cpp.py"
OUT = REPORT / "thread_benchmark_outlier_results.csv"

THREADS = [1, 2, 4, 8, 16]
RUNS_PER_THREAD = 10


def run(cmd, cwd=None, timeout=120):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
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


def parse_payload(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"runner did not emit JSON:\n{stdout}")


def compile_source(work, threads):
    run(["python3", str(SCANNER), str(SOURCE)], timeout=60)
    shutil.copy2(SOURCE, work / "solution.cpp")
    shutil.copy2(RUNNER, work / "main.cpp")
    exe = work / "bench"
    cmd = [
        "g++", "-O3", "-std=c++17", "-march=native", "-fopenmp",
        f"-DTASK2_PREDICTION_THREADS={threads}", "main.cpp", "-o", str(exe),
    ]
    run(cmd, cwd=work, timeout=120)
    return exe


def run_payload(exe, work):
    proc = run([str(exe), str(DATA), "0.001", str(RUNS_PER_THREAD)], cwd=work, timeout=1800)
    return parse_payload(proc.stdout)


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def outlier_mask(values):
    med = statistics.median(values)
    deviations = [abs(x - med) for x in values]
    mad = statistics.median(deviations)
    if mad > 0.0:
        return [abs(0.6745 * (x - med) / mad) <= 3.5 for x in values], "mad_z<=3.5"

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    if iqr > 0.0:
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return [lo <= x <= hi for x in values], "iqr_1.5"

    return [True for _ in values], "all_equal"


def summarize(threads, payload):
    runs = [float(x) for x in payload.get("time_runs", [])]
    if len(runs) != RUNS_PER_THREAD:
        raise RuntimeError(f"expected {RUNS_PER_THREAD} runs for {threads} threads, got {len(runs)}")
    mask, rule = outlier_mask(runs)
    kept = [x for x, keep in zip(runs, mask) if keep]
    removed = [x for x, keep in zip(runs, mask) if not keep]
    return {
        "threads": str(threads),
        "raw_total": f"{sum(runs):.6f}",
        "raw_mean": f"{statistics.mean(runs):.6f}",
        "raw_median": f"{statistics.median(runs):.6f}",
        "filter_rule": rule,
        "kept_count": str(len(kept)),
        "removed_count": str(len(removed)),
        "filtered_mean": f"{statistics.mean(kept):.6f}",
        "filtered_median": f"{statistics.median(kept):.6f}",
        "filtered_total10_equiv": f"{statistics.mean(kept) * RUNS_PER_THREAD:.6f}",
        "pre_rmse": f"{float(payload['rmse_base']):.6f}",
        "post_rmse": f"{float(payload['rmse']):.6f}",
        "valid": str(bool(payload["valid"])),
        "time_runs": ";".join(f"{x:.6f}" for x in runs),
        "kept_runs": ";".join(f"{x:.6f}" for x in kept),
        "removed_runs": ";".join(f"{x:.6f}" for x in removed),
    }


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    rows = []
    for threads in THREADS:
        print(f"[thread] {threads}: compiling and running {RUNS_PER_THREAD} timed runs")
        with tempfile.TemporaryDirectory(prefix=f"thread_outlier_{threads}_") as tmp:
            work = Path(tmp)
            exe = compile_source(work, threads)
            payload = run_payload(exe, work)
        row = summarize(threads, payload)
        rows.append(row)
        print(
            f"  raw_total={row['raw_total']} filtered_mean={row['filtered_mean']} "
            f"median={row['filtered_median']} removed={row['removed_count']} "
            f"rmse={row['post_rmse']} valid={row['valid']}"
        )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threads", "raw_total", "raw_mean", "raw_median", "filter_rule",
                "kept_count", "removed_count", "filtered_mean", "filtered_median",
                "filtered_total10_equiv", "pre_rmse", "post_rmse", "valid",
                "time_runs", "kept_runs", "removed_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
