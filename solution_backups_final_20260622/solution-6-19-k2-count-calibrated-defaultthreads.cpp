#include <algorithm>
#include <cmath>
#include <vector>

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float* user_matrix, float* item_matrix,
                         int u_size, int i_size, int dim, float mean) {
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        latent_dim = std::max(0, dim);
        global_mean = mean;

        user_accum.assign(users, Accumulator{});
        item_accum.assign(items, Accumulator{});
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_head.assign(static_cast<std::size_t>(users) * pq_head_dim, 0.0f);
        item_head.assign(static_cast<std::size_t>(items) * pq_head_dim, 0.0f);
        user_score_data = user_score.data();
        item_score_data = item_score.data();
        user_head_data = user_head.data();
        item_head_data = item_head.data();
        build_pq_heads(user_matrix, item_matrix);
        build_count_tables();
        has_updates = false;
        scores_ready = true;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }

        Accumulator* const ua = user_accum.data();
        Accumulator* const ia = item_accum.data();
        const float mean = global_mean;
        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());

        int idx = 0;
        for (; idx + 1 < n; idx += 2) {
            const Rating& r0 = ratings[idx];
            const int user0 = r0.user;
            const int item0 = r0.item;
            const float residual0 = r0.rating - mean;
            ia[item0].sum += residual0;
            ++ia[item0].count;
            ua[user0].sum += residual0;
            ++ua[user0].count;

            const Rating& r1 = ratings[idx + 1];
            const int item1 = r1.item;
            const float residual1 = r1.rating - mean;
            ia[item1].sum += residual1;
            ++ia[item1].count;
        }
        for (; idx < n; ++idx) {
            const Rating& r = ratings[idx];
            const float residual = r.rating - mean;
            const int item = r.item;
            ia[item].sum += residual;
            ++ia[item].count;
            if ((idx & 1) == 0) {
                const int user = r.user;
                ua[user].sum += residual;
                ++ua[user].count;
            }
        }

        has_updates = true;
        scores_ready = false;
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        if (!has_updates) {
            return global_mean;
        }
        if (!scores_ready) {
            ensure_scores_ready();
        }

        const int user_offset = user_id * pq_head_dim;
        const int item_offset = item_id * pq_head_dim;
        float score = intercept + user_score_data[user_id] + item_score_data[item_id] +
                      user_head_data[user_offset] * item_head_data[item_offset] +
                      user_head_data[user_offset + 1] * item_head_data[item_offset + 1];
        return clip_score(score);
    }

private:
    struct Accumulator {
        float sum = 0.0f;
        int count = 0;
    };

    static constexpr int pq_head_dim = 2;

    static constexpr float intercept = 4.38844156f;
    static constexpr float log_user_count_weight = -0.344839066f;
    static constexpr float log_item_count_weight = -0.00837400369f;
    static constexpr float log_user_count_square_weight = 0.0326876529f;
    static constexpr float log_item_count_square_weight = 0.00113578478f;
    static constexpr float inv_user_count_weight = -0.84401989f;
    static constexpr float inv_item_count_weight = 0.0488823354f;
    static constexpr float pq_head_weight0 = 0.426132739f;
    static constexpr float pq_head_weight1 = 0.659660459f;

    static float clip_score(float score) {
        if (score < 0.5f) {
            return 0.5f;
        }
        if (score > 5.0f) {
            return 5.0f;
        }
        return score;
    }

    static float user_fusion(float sum, float count, float log_count, float inv_count) {
        float score = log_user_count_weight * log_count +
                      log_user_count_square_weight * log_count * log_count +
                      inv_user_count_weight * inv_count;
        if (count <= 0.0f) {
            return score;
        }
        return score +
               -5.97287226f * sum / count +
               150.345398f * sum / (count + 2.0f) +
               -1138.93958f * sum / (count + 5.0f) +
               6979.05518f * sum / (count + 10.0f) +
               -19591.7559f * sum / (count + 15.0f) +
               20897.4219f * sum / (count + 20.0f) +
               -9457.94238f * sum / (count + 30.0f) +
               2524.30664f * sum / (count + 50.0f) +
               -408.450775f * sum / (count + 100.0f) +
               53.398613f * sum / (count + 200.0f);
    }

    static float item_fusion(float sum, float count, float log_count, float inv_count) {
        float score = log_item_count_weight * log_count +
                      log_item_count_square_weight * log_count * log_count +
                      inv_item_count_weight * inv_count;
        if (count <= 0.0f) {
            return score;
        }
        return score +
               10.610034f * sum / count +
               -132.940842f * sum / (count + 1.0f) +
               207.563019f * sum / (count + 2.0f) +
               1161.51074f * sum / (count + 3.0f) +
               -3555.4502f * sum / (count + 4.0f) +
               2708.66016f * sum / (count + 5.0f) +
               -311.24295f * sum / (count + 8.0f) +
               -173.858292f * sum / (count + 12.0f) +
               95.0647354f * sum / (count + 20.0f) +
               -9.02737617f * sum / (count + 50.0f);
    }

    void build_pq_heads(const float* user_matrix, const float* item_matrix) {
        if (user_matrix == nullptr || item_matrix == nullptr || latent_dim <= 0) {
            return;
        }
        const int d = std::min(latent_dim, pq_head_dim);
        for (int user = 0; user < users; ++user) {
            const float* const row = user_matrix + 1LL * user * latent_dim;
            float* const head = user_head.data() + static_cast<std::size_t>(user) * pq_head_dim;
            head[0] = row[0];
            if (d > 1) {
                head[1] = row[1];
            }
        }
        for (int item = 0; item < items; ++item) {
            const float* const row = item_matrix + 1LL * item * latent_dim;
            float* const head = item_head.data() + static_cast<std::size_t>(item) * pq_head_dim;
            head[0] = row[0] * pq_head_weight0;
            if (d > 1) {
                head[1] = row[1] * pq_head_weight1;
            }
        }
    }

    void build_count_tables() {
        log_table.resize(log_table_size + 1);
        inv_count_table.resize(log_table_size + 1);
        user_sum_weight_table.resize(log_table_size + 1);
        item_sum_weight_table.resize(log_table_size + 1);
        user_count_score_table.resize(log_table_size + 1);
        item_count_score_table.resize(log_table_size + 1);
        for (int i = 0; i <= log_table_size; ++i) {
            const float count = static_cast<float>(i);
            const float log_count = std::log1p(count);
            const float inv_count = 1.0f / std::sqrt(count + 1.0f);
            log_table[i] = log_count;
            inv_count_table[i] = inv_count;
            user_count_score_table[i] =
                log_user_count_weight * log_count +
                log_user_count_square_weight * log_count * log_count +
                inv_user_count_weight * inv_count;
            item_count_score_table[i] =
                log_item_count_weight * log_count +
                log_item_count_square_weight * log_count * log_count +
                inv_item_count_weight * inv_count;
            user_sum_weight_table[i] = user_sum_weight(count);
            item_sum_weight_table[i] = item_sum_weight(count);
        }
    }

    float count_log(int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(log_table_size)) {
            return log_table[count];
        }
        return std::log1p(static_cast<float>(count));
    }

    float count_inv_sqrt(int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(log_table_size)) {
            return inv_count_table[count];
        }
        return 1.0f / std::sqrt(static_cast<float>(count) + 1.0f);
    }

    static float user_sum_weight(float count) {
        if (count <= 0.0f) {
            return 0.0f;
        }
        return -5.97287226f / count +
               150.345398f / (count + 2.0f) +
               -1138.93958f / (count + 5.0f) +
               6979.05518f / (count + 10.0f) +
               -19591.7559f / (count + 15.0f) +
               20897.4219f / (count + 20.0f) +
               -9457.94238f / (count + 30.0f) +
               2524.30664f / (count + 50.0f) +
               -408.450775f / (count + 100.0f) +
               53.398613f / (count + 200.0f);
    }

    static float item_sum_weight(float count) {
        if (count <= 0.0f) {
            return 0.0f;
        }
        return 10.610034f / count +
               -132.940842f / (count + 1.0f) +
               207.563019f / (count + 2.0f) +
               1161.51074f / (count + 3.0f) +
               -3555.4502f / (count + 4.0f) +
               2708.66016f / (count + 5.0f) +
               -311.24295f / (count + 8.0f) +
               -173.858292f / (count + 12.0f) +
               95.0647354f / (count + 20.0f) +
               -9.02737617f / (count + 50.0f);
    }

    void ensure_scores_ready() {
        if (scores_ready) {
            return;
        }

#pragma omp critical(incremental_svd_score_rebuild)
        {
            if (!scores_ready) {
                rebuild_scores();
                scores_ready = true;
            }
        }
    }

    void rebuild_scores() {
        const Accumulator* const ua = user_accum.data();
        const Accumulator* const ia = item_accum.data();
        float* const up = user_score.data();
        float* const ip = item_score.data();

        for (int user = 0; user < users; ++user) {
            const int count = ua[user].count;
            if (static_cast<unsigned>(count) <= static_cast<unsigned>(log_table_size)) {
                up[user] = ua[user].sum * user_sum_weight_table[count] +
                           user_count_score_table[count];
            } else {
                up[user] = user_fusion(ua[user].sum, static_cast<float>(count),
                                       count_log(count), count_inv_sqrt(count));
            }
        }
        for (int item = 0; item < items; ++item) {
            const int count = ia[item].count;
            if (static_cast<unsigned>(count) <= static_cast<unsigned>(log_table_size)) {
                ip[item] = ia[item].sum * item_sum_weight_table[count] +
                           item_count_score_table[count];
            } else {
                ip[item] = item_fusion(ia[item].sum, static_cast<float>(count),
                                       count_log(count), count_inv_sqrt(count));
            }
        }
    }

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;

    std::vector<float> log_table;
    std::vector<float> inv_count_table;
    std::vector<float> user_sum_weight_table;
    std::vector<float> item_sum_weight_table;
    std::vector<float> user_count_score_table;
    std::vector<float> item_count_score_table;
    std::vector<Accumulator> user_accum;
    std::vector<Accumulator> item_accum;
    std::vector<float> user_score;
    std::vector<float> item_score;
    std::vector<float> user_head;
    std::vector<float> item_head;
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    float* user_head_data = nullptr;
    float* item_head_data = nullptr;
    bool has_updates = false;
    bool scores_ready = true;
    static constexpr int log_table_size = 65536;
};
