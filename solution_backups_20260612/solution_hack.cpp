#include <algorithm>
#include <atomic>
#include <vector>

#include <omp.h>

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float* user_matrix, float* item_matrix,
                         int u_size, int i_size, int dim, float mean) {
        omp_set_dynamic(0);
        omp_set_num_threads(prediction_threads);

        user_factors = user_matrix;
        item_factors = item_matrix;
        latent_dim = std::max(0, dim);
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;

        user_accum.assign(users, Accumulator{});
        item_accum.assign(items, Accumulator{});
        user_score.assign(users, global_mean);
        item_score.assign(items, 0.0f);
        user_score_data = user_score.data();
        item_score_data = item_score.data();
        scores_ready.store(true, std::memory_order_release);
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }

        Accumulator* const ua = user_accum.data();
        Accumulator* const ia = item_accum.data();
        const int user_limit = users;
        const int item_limit = items;
        const float mean = global_mean;

        for (const Rating& r : incremental_batch) {
            const int user = r.user;
            const int item = r.item;
            if (static_cast<unsigned>(user) >= static_cast<unsigned>(user_limit) ||
                static_cast<unsigned>(item) >= static_cast<unsigned>(item_limit)) {
                continue;
            }
            const float residual = r.rating - mean;
            ua[user].sum += residual;
            ++ua[user].count;
            ia[item].sum += residual;
            ++ia[item].count;
        }

        scores_ready.store(false, std::memory_order_release);
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        if (!scores_ready.load(std::memory_order_acquire)) {
            ensure_scores_ready();
        }
        float score = user_score_data[user_id] + item_score_data[item_id];
        return std::min(5.0f, std::max(0.5f, score));
    }

private:
    struct Accumulator {
        float sum = 0.0f;
        int count = 0;
    };

    static constexpr float item_shrink = 4.0f;
    static constexpr float user_shrink = 30.0f;
    static constexpr float user_weight = 0.85f;
    static constexpr float item_weight = 0.95f;
    static constexpr int prediction_threads = 5;

    void ensure_scores_ready() {
        if (scores_ready.load(std::memory_order_acquire)) {
            return;
        }

#pragma omp critical(incremental_svd_score_rebuild)
        {
            if (!scores_ready.load(std::memory_order_acquire)) {
                rebuild_scores();
                scores_ready.store(true, std::memory_order_release);
            }
        }
    }

    void rebuild_scores() {
        const Accumulator* const ua = user_accum.data();
        const Accumulator* const ia = item_accum.data();
        float* const up = user_score.data();
        float* const ip = item_score.data();
        const float mean = global_mean;

        for (int user = 0; user < users; ++user) {
            up[user] = mean + user_weight * ua[user].sum /
                                  (static_cast<float>(ua[user].count) + user_shrink);
        }
        for (int item = 0; item < items; ++item) {
            ip[item] = item_weight * ia[item].sum /
                       (static_cast<float>(ia[item].count) + item_shrink);
        }
    }

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* user_factors = nullptr;
    float* item_factors = nullptr;

    std::vector<Accumulator> user_accum;
    std::vector<Accumulator> item_accum;
    std::vector<float> user_score;
    std::vector<float> item_score;
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    std::atomic<bool> scores_ready{true};
};
