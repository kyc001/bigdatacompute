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
    std::vector<float> raw(static_cast<std::size_t>(rows) * 3);
    read_exact(in, raw.data(), raw.size());

    std::vector<Rating> ratings(rows);
    for (int row = 0; row < rows; ++row) {
        const int base = row * 3;
        ratings[row] = Rating{
            static_cast<int>(raw[base]),
            static_cast<int>(raw[base + 1]),
            raw[base + 2],
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

std::string json_bool(bool value) {
    return value ? "true" : "false";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            throw std::runtime_error("usage: benchmark_cpp <judge_data.bin> <epsilon> <runs>");
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

        std::vector<double> elapsed_runs;
        elapsed_runs.reserve(runs);
        float rmse_new = 0.0f;

        for (int run = 0; run < runs; ++run) {
            std::vector<float> p = data.P;
            std::vector<float> q = data.Q;
            IncrementalSVD model;
            model.load_base_model(
                p.data(), q.data(), data.num_users, data.num_items,
                data.latent_dim, data.global_mean);

            const auto start = std::chrono::steady_clock::now();
            const int batch_size = 100000;
            for (int offset = 0; offset < data.incremental_rows; offset += batch_size) {
                const int end = std::min(offset + batch_size, data.incremental_rows);
                std::vector<Rating> batch(
                    data.incremental.begin() + offset,
                    data.incremental.begin() + end);
                model.update(batch);
            }
            rmse_new = rmse(model, data.test);
            const auto end = std::chrono::steady_clock::now();
            elapsed_runs.push_back(std::chrono::duration<double>(end - start).count());
        }

        double total = 0.0;
        for (double elapsed : elapsed_runs) {
            total += elapsed;
        }
        const bool valid = std::isfinite(rmse_new) && (rmse_base - rmse_new >= epsilon);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\"status\":\"success\",";
        std::cout << "\"time_sec\":" << total << ",";
        std::cout << "\"time_runs\":[";
        for (std::size_t idx = 0; idx < elapsed_runs.size(); ++idx) {
            if (idx != 0) {
                std::cout << ",";
            }
            std::cout << elapsed_runs[idx];
        }
        std::cout << "],";
        std::cout << "\"rmse_base\":" << rmse_base << ",";
        std::cout << "\"rmse\":" << rmse_new << ",";
        std::cout << "\"valid\":" << json_bool(valid) << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "{\"status\":\"error\",\"error\":\"" << exc.what() << "\"}" << std::endl;
        return 1;
    }
}
