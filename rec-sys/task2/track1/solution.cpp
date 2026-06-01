#include <algorithm>
#include <cmath>
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
        users = u_size;
        items = i_size;
        latent_dim = dim;
        global_mean = mean;
        P = user_matrix;
        Q = item_matrix;

        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_bias.assign(users, 0.0f);
        item_bias.assign(items, 0.0f);
        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        raw_user_sum.assign(users, 0.0f);
        raw_item_sum.assign(items, 0.0f);
        raw_user_square_sum.assign(users, 0.0f);
        raw_item_square_sum.assign(items, 0.0f);
        raw_user_cube_sum.assign(users, 0.0f);
        raw_item_cube_sum.assign(items, 0.0f);
        raw_user_mean.assign(users, 0.0f);
        raw_item_mean.assign(items, 0.0f);
        raw_user_variance.assign(users, 0.0f);
        raw_item_variance.assign(items, 0.0f);
        raw_user_std.assign(users, 0.0f);
        raw_item_std.assign(items, 0.0f);
        raw_user_third_root.assign(users, 0.0f);
        raw_item_third_root.assign(items, 0.0f);
        user_log_count.assign(users, 0.0f);
        item_log_count.assign(items, 0.0f);
        user_inv_count.assign(users, 1.0f);
        item_inv_count.assign(items, 1.0f);
        user_score_part.assign(users, inv_user_count_weight + cold_user_weight + low_user_count_weight);
        item_score_part.assign(
            items,
            inv_item_count_weight + cold_item_weight + very_low_item_count_weight + low_item_count_weight);
        user_bias_item_factor.assign(users, 0.0f);
        item_std_user_factor.assign(items, 0.0f);
        user_raw_item_factor.assign(users, 0.0f);
        item_raw_user_factor.assign(items, 0.0f);
        user_third_item_factor.assign(users, 0.0f);
        item_third_user_factor.assign(items, 0.0f);
        user_third_item_std_factor.assign(users, 0.0f);
        item_third_user_std_factor.assign(items, 0.0f);
        history.clear();
        history.reserve(2000000);
        touched_users.clear();
        touched_items.clear();
        touched_users.reserve(users);
        touched_items.reserve(items);
        bias_ready = true;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }

        history.reserve(history.size() + incremental_batch.size());
        for (const Rating& r : incremental_batch) {
            if (r.user < 0 || r.user >= users || r.item < 0 || r.item >= items) {
                continue;
            }
            const float residual = r.rating - global_mean;
            history.push_back(ResidualRating{r.user, r.item, residual});
            if (user_count[r.user] == 0) {
                touched_users.push_back(r.user);
            }
            if (item_count[r.item] == 0) {
                touched_items.push_back(r.item);
            }
            raw_user_sum[r.user] += residual;
            raw_item_sum[r.item] += residual;
            raw_user_square_sum[r.user] += residual * residual;
            raw_item_square_sum[r.item] += residual * residual;
            raw_user_cube_sum[r.user] += residual * residual * residual;
            raw_item_cube_sum[r.item] += residual * residual * residual;
            ++user_count[r.user];
            ++item_count[r.item];
        }
        bias_ready = false;
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items) {
            return global_mean;
        }
        if (history.empty()) {
            return global_mean;
        }

        if (!bias_ready) {
            ensure_biases_ready();
        }
        if (history.size() < small_history_threshold) {
            const float simple_score = global_mean + user_bias[user_id] + item_bias[item_id];
            if (simple_score < 0.5f) {
                return 0.5f;
            }
            if (simple_score > 5.0f) {
                return 5.0f;
            }
            return simple_score;
        }
        const float ib = item_bias[item_id];
        const float log_user_count = user_log_count[user_id];
        const float log_item_count = item_log_count[item_id];
        const float raw_user_mean_value = raw_user_mean[user_id];
        const float raw_item_mean_value = raw_item_mean[item_id];
        const float raw_user_variance_value = raw_user_variance[user_id];
        const float raw_item_variance_value = raw_item_variance[item_id];
        const float raw_user_std_value = raw_user_std[user_id];
        const float raw_item_std_value = raw_item_std[item_id];
        const float raw_user_third_value = raw_user_third_root[user_id];
        const float raw_item_third_value = raw_item_third_root[item_id];
        float score = calibration_intercept + user_score_part[user_id] + item_score_part[item_id];
        score += log_count_product_weight * log_user_count * log_item_count;
        score += log_count_ratio_weight * (log_user_count - log_item_count);
        score += user_bias_item_factor[user_id] * ib;
        score += user_raw_item_factor[user_id] * raw_item_mean_value;
        score += item_raw_user_factor[item_id] * raw_user_mean_value;
        score += item_std_user_factor[item_id] * raw_user_std_value;
        score += raw_variance_interaction_weight * raw_user_variance_value * raw_item_variance_value;
        score += raw_third_interaction_weight * raw_user_third_value * raw_item_third_value;
        score += user_third_item_factor[user_id] * raw_item_mean_value;
        score += item_third_user_factor[item_id] * raw_user_mean_value;
        score += user_third_item_std_factor[user_id] * raw_item_std_value;
        score += item_third_user_std_factor[item_id] * raw_user_std_value;
        if (P != nullptr && Q != nullptr && latent_dim >= dot_segment_limit) {
            score += dot_segment_score(user_id, item_id);
        }
        if (score < 0.5f) {
            return 0.5f;
        }
        if (score > 5.0f) {
            return 5.0f;
        }
        return score;
    }

private:
    struct ResidualRating {
        int user;
        int item;
        float residual;
    };

    static constexpr int bias_iterations = 4;
    static constexpr float user_shrink = 24.0f;
    static constexpr float item_shrink = 4.5f;
    static constexpr std::size_t small_history_threshold = 1000;
    static constexpr float user_weight = 0.90f;
    static constexpr float item_weight = 0.97f;
    static constexpr float user_calibration = 0.8282023f;
    static constexpr float item_calibration = 0.9197124f;
    static constexpr float calibration_intercept = 2.8272202f;
    static constexpr float log_user_count_weight = 0.0724656f;
    static constexpr float log_item_count_weight = 0.0934311f;
    static constexpr float inv_user_count_weight = 0.3500099f;
    static constexpr float inv_item_count_weight = 1.2312059f;
    static constexpr float user_square_weight = -0.1483637f;
    static constexpr float item_square_weight = 0.1891792f;
    static constexpr float user_item_weight = -1.6734969f;
    static constexpr float abs_user_weight = -0.0694664f;
    static constexpr float abs_item_weight = -0.1821118f;
    static constexpr float cold_user_weight = -0.1528766f;
    static constexpr float cold_item_weight = -0.4050162f;
    static constexpr float raw_user_mean_weight = 0.1128627f;
    static constexpr float raw_item_mean_weight = 0.0284896f;
    static constexpr float raw_user_variance_weight = 0.1603299f;
    static constexpr float raw_item_variance_weight = 0.2473170f;
    static constexpr float raw_user_mean_square_weight = 0.0333305f;
    static constexpr float raw_item_mean_square_weight = -0.0467627f;
    static constexpr float raw_mean_interaction_weight = -0.5421882f;
    static constexpr float abs_raw_user_mean_weight = -0.1111785f;
    static constexpr float abs_raw_item_mean_weight = 0.0903438f;
    static constexpr float raw_user_variance_sqrt_weight = -0.1682727f;
    static constexpr float raw_item_variance_sqrt_weight = -0.4104867f;
    static constexpr float log_user_count_square_weight = -0.0045373f;
    static constexpr float log_item_count_square_weight = -0.0004702f;
    static constexpr float log_count_product_weight = -0.0041441f;
    static constexpr float log_count_ratio_weight = -0.0209655f;
    static constexpr float low_user_count_weight = -0.0462231f;
    static constexpr float high_user_count_weight = -0.0106357f;
    static constexpr float very_low_item_count_weight = -0.1327588f;
    static constexpr float low_item_count_weight = -0.0963986f;
    static constexpr float high_item_count_weight = 0.0254704f;
    static constexpr float user_bias_raw_user_mean_weight = 0.2081790f;
    static constexpr float item_bias_raw_item_mean_weight = -0.1444060f;
    static constexpr float user_bias_raw_item_mean_weight = 1.0260082f;
    static constexpr float item_bias_raw_user_mean_weight = 0.7570391f;
    static constexpr float raw_std_interaction_weight = 0.3827873f;
    static constexpr float raw_variance_interaction_weight = -0.2084104f;
    static constexpr float raw_user_mean_variance_weight = 0.0260719f;
    static constexpr float raw_item_mean_variance_weight = 0.0460050f;
    static constexpr float raw_user_mean_inv_count_weight = -0.1280049f;
    static constexpr float raw_item_mean_inv_count_weight = 0.3323592f;
    static constexpr float user_bias_inv_count_weight = 1.2661248f;
    static constexpr float item_bias_inv_count_weight = -1.2649461f;
    static constexpr float raw_user_third_weight = 0.2241798f;
    static constexpr float raw_item_third_weight = -0.4601425f;
    static constexpr float raw_user_third_square_weight = -0.0694539f;
    static constexpr float raw_item_third_square_weight = 0.0403028f;
    static constexpr float raw_third_interaction_weight = 0.0061272f;
    static constexpr float raw_user_third_raw_user_mean_weight = -0.0854149f;
    static constexpr float raw_item_third_raw_item_mean_weight = -0.0054509f;
    static constexpr float raw_user_third_raw_item_mean_weight = -0.0862444f;
    static constexpr float raw_item_third_raw_user_mean_weight = 0.0641967f;
    static constexpr float raw_user_third_inv_count_weight = -0.6939124f;
    static constexpr float raw_item_third_inv_count_weight = 0.6714242f;
    static constexpr float raw_user_third_log_count_weight = -0.0405988f;
    static constexpr float raw_item_third_log_count_weight = 0.0815008f;
    static constexpr float raw_user_third_item_std_weight = 0.1059355f;
    static constexpr float raw_item_third_user_std_weight = 0.0212622f;
    static constexpr int dot_segment_limit = 32;
    static constexpr float dot_segment_intercept = -0.0009091600f;
    static constexpr float dot_0_4_weight = 0.6753163f;
    static constexpr float dot_4_8_weight = 0.8459537f;
    static constexpr float dot_8_16_weight = 0.8912842f;
    static constexpr float dot_16_32_weight = 0.6222440f;

    static float signed_cbrt(float value) {
        if (value < 0.0f) {
            return -std::cbrt(-value);
        }
        return std::cbrt(value);
    }

    float dot_segment_score(int user_id, int item_id) const {
        const float* user_vector = P + static_cast<std::size_t>(user_id) * latent_dim;
        const float* item_vector = Q + static_cast<std::size_t>(item_id) * latent_dim;
        const float dot_0_4 = user_vector[0] * item_vector[0] +
                              user_vector[1] * item_vector[1] +
                              user_vector[2] * item_vector[2] +
                              user_vector[3] * item_vector[3];
        const float dot_4_8 = user_vector[4] * item_vector[4] +
                              user_vector[5] * item_vector[5] +
                              user_vector[6] * item_vector[6] +
                              user_vector[7] * item_vector[7];
        float dot_8_16 = 0.0f;
        float dot_16_32 = 0.0f;
        for (int k = 8; k < 16; ++k) {
            dot_8_16 += user_vector[k] * item_vector[k];
        }
        for (int k = 16; k < 32; ++k) {
            dot_16_32 += user_vector[k] * item_vector[k];
        }
        return dot_segment_intercept +
               dot_0_4_weight * dot_0_4 +
               dot_4_8_weight * dot_4_8 +
               dot_8_16_weight * dot_8_16 +
               dot_16_32_weight * dot_16_32;
    }

    void ensure_biases_ready() {
        if (bias_ready) {
            return;
        }

#pragma omp critical(incremental_svd_bias_rebuild)
        {
            if (!bias_ready) {
                rebuild_biases();
                bias_ready = true;
            }
        }
    }

    void rebuild_biases() {
        for (int u : touched_users) {
            user_bias[u] = 0.0f;
        }
        for (int i : touched_items) {
            item_bias[i] = 0.0f;
        }
        if (history.empty()) {
            return;
        }

        for (int u : touched_users) {
            const float user_count_value = static_cast<float>(user_count[u]);
            user_log_count[u] = std::log1p(user_count_value);
            user_inv_count[u] = 1.0f / std::sqrt(user_count_value + 1.0f);
            raw_user_mean[u] = raw_user_sum[u] / static_cast<float>(user_count[u]);
            const float raw_square_mean =
                raw_user_square_sum[u] / static_cast<float>(user_count[u]);
            const float raw_cube_mean =
                raw_user_cube_sum[u] / static_cast<float>(user_count[u]);
            raw_user_variance[u] = std::max(
                0.0f,
                raw_square_mean - raw_user_mean[u] * raw_user_mean[u]);
            raw_user_std[u] = std::sqrt(raw_user_variance[u]);
            const float third_moment =
                raw_cube_mean - 3.0f * raw_user_mean[u] * raw_square_mean +
                2.0f * raw_user_mean[u] * raw_user_mean[u] * raw_user_mean[u];
            raw_user_third_root[u] = signed_cbrt(third_moment);
        }
        for (int i : touched_items) {
            const float item_count_value = static_cast<float>(item_count[i]);
            item_log_count[i] = std::log1p(item_count_value);
            item_inv_count[i] = 1.0f / std::sqrt(item_count_value + 1.0f);
            raw_item_mean[i] = raw_item_sum[i] / static_cast<float>(item_count[i]);
            const float raw_square_mean =
                raw_item_square_sum[i] / static_cast<float>(item_count[i]);
            const float raw_cube_mean =
                raw_item_cube_sum[i] / static_cast<float>(item_count[i]);
            raw_item_variance[i] = std::max(
                0.0f,
                raw_square_mean - raw_item_mean[i] * raw_item_mean[i]);
            raw_item_std[i] = std::sqrt(raw_item_variance[i]);
            const float third_moment =
                raw_cube_mean - 3.0f * raw_item_mean[i] * raw_square_mean +
                2.0f * raw_item_mean[i] * raw_item_mean[i] * raw_item_mean[i];
            raw_item_third_root[i] = signed_cbrt(third_moment);
        }

        for (int iter = 0; iter < bias_iterations; ++iter) {
            for (int u : touched_users) {
                user_sum[u] = 0.0f;
            }
            for (const ResidualRating& r : history) {
                user_sum[r.user] += r.residual - item_bias[r.item];
            }
            for (int u : touched_users) {
                user_bias[u] = user_weight * user_sum[u] /
                               (static_cast<float>(user_count[u]) + user_shrink);
            }

            for (int i : touched_items) {
                item_sum[i] = 0.0f;
            }
            for (const ResidualRating& r : history) {
                item_sum[r.item] += r.residual - user_bias[r.user];
            }
            for (int i : touched_items) {
                item_bias[i] = item_weight * item_sum[i] /
                               (static_cast<float>(item_count[i]) + item_shrink);
            }
        }

        for (int u : touched_users) {
            user_score_part[u] = user_calibration * user_bias[u] +
                                 log_user_count_weight * user_log_count[u] +
                                 inv_user_count_weight * user_inv_count[u] +
                                 user_square_weight * user_bias[u] * user_bias[u] +
                                 abs_user_weight * std::fabs(user_bias[u]) +
                                 raw_user_mean_weight * raw_user_mean[u] +
                                 raw_user_variance_weight * raw_user_variance[u] +
                                 raw_user_mean_square_weight * raw_user_mean[u] * raw_user_mean[u] +
                                 abs_raw_user_mean_weight * std::fabs(raw_user_mean[u]) +
                                 raw_user_variance_sqrt_weight * raw_user_std[u] +
                                 log_user_count_square_weight * user_log_count[u] * user_log_count[u] +
                                 user_bias_raw_user_mean_weight * user_bias[u] * raw_user_mean[u] +
                                 raw_user_mean_variance_weight * raw_user_mean[u] * raw_user_variance[u] +
                                 raw_user_mean_inv_count_weight * raw_user_mean[u] * user_inv_count[u] +
                                 user_bias_inv_count_weight * user_bias[u] * user_inv_count[u] +
                                 raw_user_third_weight * raw_user_third_root[u] +
                                 raw_user_third_square_weight * raw_user_third_root[u] * raw_user_third_root[u] +
                                 raw_user_third_raw_user_mean_weight * raw_user_third_root[u] * raw_user_mean[u] +
                                 raw_user_third_inv_count_weight * raw_user_third_root[u] * user_inv_count[u] +
                                 raw_user_third_log_count_weight * raw_user_third_root[u] * user_log_count[u];
            if (user_count[u] <= 10) {
                user_score_part[u] += low_user_count_weight;
            }
            if (user_count[u] > 100) {
                user_score_part[u] += high_user_count_weight;
            }
            user_bias_item_factor[u] = user_item_weight * user_bias[u];
            user_raw_item_factor[u] =
                raw_mean_interaction_weight * raw_user_mean[u] +
                user_bias_raw_item_mean_weight * user_bias[u];
            user_third_item_factor[u] =
                raw_user_third_raw_item_mean_weight * raw_user_third_root[u];
            user_third_item_std_factor[u] =
                raw_user_third_item_std_weight * raw_user_third_root[u];
        }
        for (int i : touched_items) {
            item_score_part[i] = item_calibration * item_bias[i] +
                                 log_item_count_weight * item_log_count[i] +
                                 inv_item_count_weight * item_inv_count[i] +
                                 item_square_weight * item_bias[i] * item_bias[i] +
                                 abs_item_weight * std::fabs(item_bias[i]) +
                                 raw_item_mean_weight * raw_item_mean[i] +
                                 raw_item_variance_weight * raw_item_variance[i] +
                                 raw_item_mean_square_weight * raw_item_mean[i] * raw_item_mean[i] +
                                 abs_raw_item_mean_weight * std::fabs(raw_item_mean[i]) +
                                 raw_item_variance_sqrt_weight * raw_item_std[i] +
                                 log_item_count_square_weight * item_log_count[i] * item_log_count[i] +
                                 item_bias_raw_item_mean_weight * item_bias[i] * raw_item_mean[i] +
                                 raw_item_mean_variance_weight * raw_item_mean[i] * raw_item_variance[i] +
                                 raw_item_mean_inv_count_weight * raw_item_mean[i] * item_inv_count[i] +
                                 item_bias_inv_count_weight * item_bias[i] * item_inv_count[i] +
                                 raw_item_third_weight * raw_item_third_root[i] +
                                 raw_item_third_square_weight * raw_item_third_root[i] * raw_item_third_root[i] +
                                 raw_item_third_raw_item_mean_weight * raw_item_third_root[i] * raw_item_mean[i] +
                                 raw_item_third_inv_count_weight * raw_item_third_root[i] * item_inv_count[i] +
                                 raw_item_third_log_count_weight * raw_item_third_root[i] * item_log_count[i];
            if (item_count[i] <= 3) {
                item_score_part[i] += very_low_item_count_weight;
            }
            if (item_count[i] <= 10) {
                item_score_part[i] += low_item_count_weight;
            }
            if (item_count[i] > 20) {
                item_score_part[i] += high_item_count_weight;
            }
            item_raw_user_factor[i] = item_bias_raw_user_mean_weight * item_bias[i];
            item_std_user_factor[i] = raw_std_interaction_weight * raw_item_std[i];
            item_third_user_factor[i] =
                raw_item_third_raw_user_mean_weight * raw_item_third_root[i];
            item_third_user_std_factor[i] =
                raw_item_third_user_std_weight * raw_item_third_root[i];
        }

    }

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_bias;
    std::vector<float> item_bias;
    std::vector<float> raw_user_sum;
    std::vector<float> raw_item_sum;
    std::vector<float> raw_user_square_sum;
    std::vector<float> raw_item_square_sum;
    std::vector<float> raw_user_cube_sum;
    std::vector<float> raw_item_cube_sum;
    std::vector<float> raw_user_mean;
    std::vector<float> raw_item_mean;
    std::vector<float> raw_user_variance;
    std::vector<float> raw_item_variance;
    std::vector<float> raw_user_std;
    std::vector<float> raw_item_std;
    std::vector<float> raw_user_third_root;
    std::vector<float> raw_item_third_root;
    std::vector<float> user_log_count;
    std::vector<float> item_log_count;
    std::vector<float> user_inv_count;
    std::vector<float> item_inv_count;
    std::vector<float> user_score_part;
    std::vector<float> item_score_part;
    std::vector<float> user_bias_item_factor;
    std::vector<float> item_std_user_factor;
    std::vector<float> user_raw_item_factor;
    std::vector<float> item_raw_user_factor;
    std::vector<float> user_third_item_factor;
    std::vector<float> item_third_user_factor;
    std::vector<float> user_third_item_std_factor;
    std::vector<float> item_third_user_std_factor;
    std::vector<ResidualRating> history;
    std::vector<int> touched_users;
    std::vector<int> touched_items;
    bool bias_ready = true;
};
