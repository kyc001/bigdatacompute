#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "../track1/solution.cpp"

namespace {

template <typename T>
void read_exact(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
        throw std::runtime_error("read failed");
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
        ratings[row] = Rating{read_one<std::int32_t>(in), read_one<std::int32_t>(in), read_one<float>(in)};
    }
    return ratings;
}

struct JudgeData {
    int users = 0;
    int items = 0;
    int dim = 0;
    int inc_rows = 0;
    int test_rows = 0;
    float mean = 0.0f;
    std::vector<float> p;
    std::vector<float> q;
    std::vector<Rating> inc;
    std::vector<Rating> test;
};

JudgeData load_data(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("open failed");
    }
    char magic[8]{};
    read_exact(in, magic, 8);
    JudgeData data;
    (void)read_one<std::int32_t>(in);
    data.users = read_one<std::int32_t>(in);
    data.items = read_one<std::int32_t>(in);
    data.dim = read_one<std::int32_t>(in);
    data.inc_rows = read_one<std::int32_t>(in);
    data.test_rows = read_one<std::int32_t>(in);
    data.mean = read_one<float>(in);
    (void)read_one<std::int32_t>(in);
    (void)read_one<std::int32_t>(in);
    data.p.resize(static_cast<std::size_t>(data.users) * data.dim);
    data.q.resize(static_cast<std::size_t>(data.items) * data.dim);
    read_exact(in, data.p.data(), data.p.size());
    read_exact(in, data.q.data(), data.q.size());
    data.inc = read_ratings(in, data.inc_rows);
    data.test = read_ratings(in, data.test_rows);
    return data;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main() {
    const JudgeData data = load_data("rec-sys/task2/track1/secure_data_full_1024/judge_data.bin");
    std::vector<float> p = data.p;
    std::vector<float> q = data.q;
    IncrementalSVD model;
    model.load_base_model(p.data(), q.data(), data.users, data.items, data.dim, data.mean);
    double make_batch_s = 0.0;
    double update_s = 0.0;
    for (int offset = 0; offset < data.inc_rows; offset += 100000) {
        const int end = std::min(offset + 100000, data.inc_rows);
        auto start = std::chrono::steady_clock::now();
        std::vector<Rating> batch(data.inc.begin() + offset, data.inc.begin() + end);
        make_batch_s += seconds_since(start);
        start = std::chrono::steady_clock::now();
        model.update(batch);
        const double elapsed = seconds_since(start);
        update_s += elapsed;
        if (end == data.inc_rows || end - offset != 100000) {
            std::cout << "final_update " << elapsed << " rows " << (end - offset) << "\n";
        }
    }
    std::cout << "make_batch " << make_batch_s << " update_only " << update_s
              << " total_update_region " << (make_batch_s + update_s) << "\n";
    return 0;
}
