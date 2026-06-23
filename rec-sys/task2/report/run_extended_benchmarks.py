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
PROCESS_OUT = REPORT / "process_benchmark_results.csv"
PROFILE_OUT = REPORT / "stage_profile_results.csv"


PROCESS_METHODS = [
    ("global_mean", "全局均值", SRC_DIR / "constant.cpp", "baseline", "不使用增量信息"),
    ("k2_count", "K2 计数校准", ROOT / "solution-6-19-k2-count-calibrated-repro.cpp", "process", "两项 P/Q 头部特征与计数校准"),
    ("factor128", "128 参因子化", ROOT / "solution-6-19-factorized128-before-speedopt.cpp", "process", "低参数因子化映射"),
    ("thread_local", "线程本地统计", ROOT / "solution-6-21-before-compact-speedopt.cpp", "process", "并行局部统计与快速预测"),
    ("segment_prior", "用户分段先验", ROOT / "solution-6-21-segment-base7-119-candidate.cpp", "process", "119 段用户先验"),
    ("touched_refresh", "触达刷新", ROOT / "solution-6-21-segment-touched-nolazy-before-online.cpp", "process", "只刷新触达用户和物品"),
    ("online_no_table", "在线采样", ROOT / "solution-6-21-online-sampled-stride8-no-lazy-final.cpp", "process", "update 内即时刷新"),
    ("final", "最终 stride=4", ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp", "final", "计数查表与 stride=4 在线采样"),
]

PROFILE_METHODS = [
    ("dense_item", "完整物品统计", SRC_DIR / "dense_item.cpp"),
    ("final", "最终 stride=4", SRC_DIR / "final.cpp"),
    ("stride16", "物品 stride=16", SRC_DIR / "stride16.cpp"),
    ("no_count_terms", "无计数形状项", SRC_DIR / "no_count_terms.cpp"),
]


PROFILE_RUNNER = r'''
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "solution.cpp"

namespace {

template <typename T>
void read_exact(std::ifstream& in, T* data, std::size_t count) {
    const std::size_t bytes = sizeof(T) * count;
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) {
        throw std::runtime_error("judge_data.bin is truncated or unreadable");
    }
}

template <typename T>
T read_one(std::ifstream& in) {
    T value{};
    read_exact(in, &value, 1);
    return value;
}

std::vector<Rating> read_ratings(std::ifstream& in, int rows) {
    std::vector<Rating> ratings(rows);
    for (int row = 0; row < rows; ++row) {
        ratings[row] = Rating{
            read_one<std::int32_t>(in),
            read_one<std::int32_t>(in),
            read_one<float>(in),
        };
    }
    return ratings;
}

float rmse(IncrementalSVD& model, const std::vector<Rating>& test) {
    double sqerr = 0.0;
    const int n = static_cast<int>(test.size());

#pragma omp parallel for reduction(+ : sqerr) schedule(static)
    for (int idx = 0; idx < n; ++idx) {
        const Rating& r = test[idx];
        const float pred = model.predict(r.user, r.item);
        const double err = static_cast<double>(r.rating) - static_cast<double>(pred);
        sqerr += err * err;
    }

    return static_cast<float>(std::sqrt(sqerr / static_cast<double>(test.size())));
}

struct JudgeData {
    int num_users = 0;
    int num_items = 0;
    int latent_dim = 0;
    int incremental_rows = 0;
    int test_rows = 0;
    float global_mean = 0.0f;
    std::vector<float> P;
    std::vector<float> Q;
    std::vector<Rating> incremental;
    std::vector<Rating> test;
};

JudgeData load_judge_data(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open judge_data.bin");
    }

    char magic[8]{};
    read_exact(in, magic, 8);
    if (std::string(magic, 8) != "SVDJUDGE") {
        throw std::runtime_error("invalid judge data magic");
    }

    const int version = read_one<std::int32_t>(in);
    if (version != 1) {
        throw std::runtime_error("unsupported judge data version");
    }

    JudgeData data;
    data.num_users = read_one<std::int32_t>(in);
    data.num_items = read_one<std::int32_t>(in);
    data.latent_dim = read_one<std::int32_t>(in);
    data.incremental_rows = read_one<std::int32_t>(in);
    data.test_rows = read_one<std::int32_t>(in);
    data.global_mean = read_one<float>(in);
    (void)read_one<std::int32_t>(in);
    (void)read_one<std::int32_t>(in);

    data.P.resize(static_cast<std::size_t>(data.num_users) * data.latent_dim);
    data.Q.resize(static_cast<std::size_t>(data.num_items) * data.latent_dim);
    read_exact(in, data.P.data(), data.P.size());
    read_exact(in, data.Q.data(), data.Q.size());
    data.incremental = read_ratings(in, data.incremental_rows);
    data.test = read_ratings(in, data.test_rows);
    return data;
}

void print_array(const std::vector<double>& values) {
    std::cout << "[";
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            std::cout << ",";
        }
        std::cout << values[idx];
    }
    std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            throw std::runtime_error("usage: profile_cpp <judge_data.bin> <epsilon> <runs>");
        }

        const std::string data_path = argv[1];
        const float epsilon = std::stof(argv[2]);
        const int runs = std::max(1, std::stoi(argv[3]));
        JudgeData data = load_judge_data(data_path);

        IncrementalSVD base_model;
        base_model.load_base_model(
            data.P.data(), data.Q.data(), data.num_users, data.num_items,
            data.latent_dim, data.global_mean);
        const float rmse_base = rmse(base_model, data.test);

        std::vector<double> update_runs;
        std::vector<double> predict_runs;
        update_runs.reserve(runs);
        predict_runs.reserve(runs);
        float rmse_new = 0.0f;

        for (int run = 0; run < runs; ++run) {
            std::vector<float> p = data.P;
            std::vector<float> q = data.Q;
            IncrementalSVD model;
            model.load_base_model(
                p.data(), q.data(), data.num_users, data.num_items,
                data.latent_dim, data.global_mean);

            const auto update_start = std::chrono::steady_clock::now();
            const int batch_size = 100000;
            for (int offset = 0; offset < data.incremental_rows; offset += batch_size) {
                const int end = std::min(offset + batch_size, data.incremental_rows);
                std::vector<Rating> batch(
                    data.incremental.begin() + offset,
                    data.incremental.begin() + end);
                model.update(batch);
            }
            const auto update_end = std::chrono::steady_clock::now();
            rmse_new = rmse(model, data.test);
            const auto predict_end = std::chrono::steady_clock::now();
            update_runs.push_back(std::chrono::duration<double>(update_end - update_start).count());
            predict_runs.push_back(std::chrono::duration<double>(predict_end - update_end).count());
        }

        double update_total = 0.0;
        double predict_total = 0.0;
        for (int idx = 0; idx < runs; ++idx) {
            update_total += update_runs[idx];
            predict_total += predict_runs[idx];
        }
        const bool valid = std::isfinite(rmse_new) && (rmse_base - rmse_new >= epsilon);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\"status\":\"success\",";
        std::cout << "\"update_sec\":" << update_total << ",";
        std::cout << "\"predict_sec\":" << predict_total << ",";
        std::cout << "\"time_sec\":" << update_total + predict_total << ",";
        std::cout << "\"update_runs\":";
        print_array(update_runs);
        std::cout << ",\"predict_runs\":";
        print_array(predict_runs);
        std::cout << ",\"rmse_base\":" << rmse_base << ",";
        std::cout << "\"rmse\":" << rmse_new << ",";
        std::cout << "\"valid\":" << (valid ? "true" : "false") << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "{\"status\":\"error\",\"error\":\"" << exc.what() << "\"}" << std::endl;
        return 1;
    }
}
'''


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


def compile_source(source: Path, work: Path, profile=False) -> Path:
    run(["python3", str(SCANNER), str(source)], timeout=60)
    shutil.copy2(source, work / "solution.cpp")
    if profile:
        (work / "main.cpp").write_text(PROFILE_RUNNER, encoding="utf-8", newline="\n")
    else:
        shutil.copy2(RUNNER, work / "main.cpp")
    exe = work / "bench"
    run(
        ["g++", "-O3", "-std=c++17", "-march=native", "-fopenmp", "main.cpp", "-o", str(exe)],
        cwd=work,
        timeout=180,
    )
    return exe


def parse_payload(stdout: str):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            if payload.get("status") != "success":
                raise RuntimeError(payload)
            return payload
    raise RuntimeError(f"runner did not emit JSON:\n{stdout}")


def run_payload(exe: Path, work: Path, runs=5):
    proc = run([str(exe), str(DATA), "0.001", str(runs)], cwd=work, timeout=900)
    return parse_payload(proc.stdout)


def write_process():
    rows = []
    for method_id, label, source, group, note in PROCESS_METHODS:
        print(f"[process] {method_id}: {label}")
        if not source.exists():
            print(f"  missing: {source}")
            continue
        with tempfile.TemporaryDirectory(prefix=f"process_{method_id}_") as tmp:
            work = Path(tmp)
            exe = compile_source(source, work, profile=False)
            payload = run_payload(exe, work, runs=5)
            runs = [float(x) for x in payload.get("time_runs", [])]
            row = {
                "method": method_id,
                "label": label,
                "group": group,
                "note": note,
                "total": f"{float(payload['time_sec']):.6f}",
                "mean_run": f"{sum(runs) / len(runs):.6f}" if runs else "",
                "pre_rmse": f"{float(payload['rmse_base']):.6f}",
                "post_rmse": f"{float(payload['rmse']):.6f}",
                "valid": str(bool(payload["valid"])),
                "time_runs": ";".join(f"{x:.6f}" for x in runs),
            }
            rows.append(row)
            print(f"  total={row['total']} rmse={row['post_rmse']} valid={row['valid']}")
    with PROCESS_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "label", "group", "note", "total", "mean_run",
                "pre_rmse", "post_rmse", "valid", "time_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {PROCESS_OUT}")


def write_profile():
    rows = []
    for method_id, label, source in PROFILE_METHODS:
        print(f"[profile] {method_id}: {label}")
        with tempfile.TemporaryDirectory(prefix=f"profile_{method_id}_") as tmp:
            work = Path(tmp)
            exe = compile_source(source, work, profile=True)
            payload = run_payload(exe, work, runs=5)
            update_runs = [float(x) for x in payload["update_runs"]]
            predict_runs = [float(x) for x in payload["predict_runs"]]
            row = {
                "method": method_id,
                "label": label,
                "update_total": f"{float(payload['update_sec']):.6f}",
                "predict_total": f"{float(payload['predict_sec']):.6f}",
                "total": f"{float(payload['time_sec']):.6f}",
                "update_mean": f"{sum(update_runs) / len(update_runs):.6f}",
                "predict_mean": f"{sum(predict_runs) / len(predict_runs):.6f}",
                "pre_rmse": f"{float(payload['rmse_base']):.6f}",
                "post_rmse": f"{float(payload['rmse']):.6f}",
                "valid": str(bool(payload["valid"])),
                "update_runs": ";".join(f"{x:.6f}" for x in update_runs),
                "predict_runs": ";".join(f"{x:.6f}" for x in predict_runs),
            }
            rows.append(row)
            print(
                f"  update={row['update_total']} predict={row['predict_total']} "
                f"rmse={row['post_rmse']} valid={row['valid']}"
            )
    with PROFILE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "label", "update_total", "predict_total", "total",
                "update_mean", "predict_mean", "pre_rmse", "post_rmse", "valid",
                "update_runs", "predict_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {PROFILE_OUT}")


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    write_process()
    write_profile()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
