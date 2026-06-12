import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "rec-sys" / "task2" / "report"
DATA = ROOT / "rec-sys" / "task2" / "track1" / "secure_data_full_1024" / "judge_data.bin"
RUNNER = ROOT / "rec-sys" / "task2" / "runner" / "cpp" / "main.cpp"
SCANNER = ROOT / "rec-sys" / "task2" / "scripts" / "scan_cpp.py"
OUT = REPORT / "ablation_benchmark_results.csv"

METHODS = [
    ("final", "最终采样残差", "rec-sys/task2/report/ablation_sources/ablation_final.cpp", "main"),
    ("dense_item", "完整物品统计", "solution_backups_20260612/solution-6-24-user-touched-item-full.cpp", "method"),
    ("stride16", "激进采样", "solution_backups_20260612/solution-6-50-stride16-phase13-notls.cpp", "method"),
    ("item_only", "Item-only基线", "solution_backups_20260612/solution-6-3.cpp", "method"),
    ("constant", "无增量常数预测", "solution_backups_20260612/solution-6-33-lower-bound-global.cpp", "lower"),
    ("no_user_prior", "去用户分段先验", "rec-sys/task2/report/ablation_sources/ablation_no_user_prior.cpp", "ablation"),
    ("no_count_terms", "去计数特征", "rec-sys/task2/report/ablation_sources/ablation_no_count_terms.cpp", "ablation"),
    ("no_user_residual", "去用户残差", "rec-sys/task2/report/ablation_sources/ablation_no_user_residual.cpp", "ablation"),
    ("no_item_residual", "去物品残差", "rec-sys/task2/report/ablation_sources/ablation_no_item_residual.cpp", "ablation"),
    ("no_prior_no_count", "去先验和计数", "rec-sys/task2/report/ablation_sources/ablation_no_prior_no_count.cpp", "ablation"),
    ("stride2", "物品stride=2", "rec-sys/task2/report/ablation_sources/ablation_item_stride2.cpp", "sampling"),
    ("stride8", "物品stride=8", "rec-sys/task2/report/ablation_sources/ablation_item_stride8.cpp", "sampling"),
]


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


def compile_method(source: Path, work: Path) -> Path:
    run(["python3", str(SCANNER), str(source)], timeout=60)
    shutil.copy2(source, work / "solution.cpp")
    shutil.copy2(RUNNER, work / "main.cpp")
    exe = work / "bench"
    run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-march=haswell",
            "-fopenmp",
            "main.cpp",
            "-o",
            str(exe),
        ],
        cwd=work,
        timeout=120,
    )
    return exe


def parse_payload(stdout: str):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"runner did not emit JSON:\n{stdout}")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    rows = []
    for method_id, label, rel_source, group in METHODS:
        source = ROOT / rel_source
        print(f"[method] {method_id}: {label}")
        with tempfile.TemporaryDirectory(prefix=f"report_{method_id}_") as tmp:
            work = Path(tmp)
            exe = compile_method(source, work)
            for repeat in range(1, 6):
                proc = run([str(exe), str(DATA), "0.001", "10"], cwd=work, timeout=900)
                payload = parse_payload(proc.stdout)
                row = {
                    "method": method_id,
                    "label": label,
                    "group": group,
                    "repeat": repeat,
                    "total": f"{float(payload['time_sec']):.6f}",
                    "pre_rmse": f"{float(payload['rmse_base']):.6f}",
                    "post_rmse": f"{float(payload['rmse']):.6f}",
                    "valid": str(bool(payload["valid"])),
                    "time_runs": ";".join(f"{float(x):.6f}" for x in payload.get("time_runs", [])),
                }
                rows.append(row)
                print(f"  repeat {repeat}: total={row['total']} rmse={row['post_rmse']} valid={row['valid']}")

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "label", "group", "repeat", "total", "pre_rmse", "post_rmse", "valid", "time_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
