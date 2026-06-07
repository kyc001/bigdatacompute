#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

#ifndef TASK2_PREDICTION_THREADS
#define TASK2_PREDICTION_THREADS 3
#endif

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float*, float*, int u_size, int i_size, int, float mean) {
        omp_set_dynamic(0);
        omp_set_num_threads(prediction_threads);

        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;

        user_accum.assign(users, Accumulator{});
        item_accum.assign(items, Accumulator{});
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_score_data = user_score.data();
        item_score_data = item_score.data();
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

        float score = intercept + user_score_data[user_id] + item_score_data[item_id];
        return clip_score(score);
    }

private:
    struct Accumulator {
        float sum = 0.0f;
        int count = 0;
    };

    static constexpr int prediction_threads = TASK2_PREDICTION_THREADS;

    static constexpr float intercept = 4.430029912552816f;
    static constexpr float log_user_count_weight = -0.35739037763299225f;
    static constexpr float log_item_count_weight = -0.010405138231289704f;
    static constexpr float log_user_count_square_weight = 0.03385049751798333f;
    static constexpr float log_item_count_square_weight = 0.001317321440029219f;
    static constexpr float inv_user_count_weight = -0.8801704691948185f;
    static constexpr float inv_item_count_weight = 0.0439380115327232f;

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
               -6.011119557954283f * sum / count +
               153.37655860941354f * sum / (count + 2.0f) +
               -1177.6176880917649f * sum / (count + 5.0f) +
               7316.511532627338f * sum / (count + 10.0f) +
               -20717.965609694682f * sum / (count + 15.0f) +
               22234.172758447374f * sum / (count + 20.0f) +
               -10147.896487977032f * sum / (count + 30.0f) +
               2736.3039826653953f * sum / (count + 50.0f) +
               -448.4622900505777f * sum / (count + 100.0f) +
               59.092543771084046f * sum / (count + 200.0f);
    }

    static float item_fusion(float sum, float count, float log_count, float inv_count) {
        float score = log_item_count_weight * log_count +
                      log_item_count_square_weight * log_count * log_count +
                      inv_item_count_weight * inv_count;
        if (count <= 0.0f) {
            return score;
        }
        return score +
               -20.096646750173583f * sum / count +
               533.1419247737755f * sum / (count + 1.0f) +
               -3070.249607487788f * sum / (count + 2.0f) +
               5553.590534031688f * sum / (count + 3.0f) +
               -553.3363028957483f * sum / (count + 4.0f) +
               -4939.89517163505f * sum / (count + 5.0f) +
               4118.89690808396f * sum / (count + 8.0f) +
               -1978.9294617524615f * sum / (count + 12.0f) +
               376.029227050923f * sum / (count + 20.0f) +
               -18.264405038720056f * sum / (count + 50.0f);
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
        return -6.011119557954283f / count +
               153.37655860941354f / (count + 2.0f) +
               -1177.6176880917649f / (count + 5.0f) +
               7316.511532627338f / (count + 10.0f) +
               -20717.965609694682f / (count + 15.0f) +
               22234.172758447374f / (count + 20.0f) +
               -10147.896487977032f / (count + 30.0f) +
               2736.3039826653953f / (count + 50.0f) +
               -448.4622900505777f / (count + 100.0f) +
               59.092543771084046f / (count + 200.0f);
    }

    static float item_sum_weight(float count) {
        if (count <= 0.0f) {
            return 0.0f;
        }
        return -20.096646750173583f / count +
               533.1419247737755f / (count + 1.0f) +
               -3070.249607487788f / (count + 2.0f) +
               5553.590534031688f / (count + 3.0f) +
               -553.3363028957483f / (count + 4.0f) +
               -4939.89517163505f / (count + 5.0f) +
               4118.89690808396f / (count + 8.0f) +
               -1978.9294617524615f / (count + 12.0f) +
               376.029227050923f / (count + 20.0f) +
               -18.264405038720056f / (count + 50.0f);
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
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    bool has_updates = false;
    bool scores_ready = true;
    static constexpr int log_table_size = 65536;
};
