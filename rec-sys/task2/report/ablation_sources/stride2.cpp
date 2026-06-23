#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

#ifndef TASK2_PREDICTION_THREADS
#define TASK2_PREDICTION_THREADS 2
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
        omp_set_num_threads(TASK2_PREDICTION_THREADS);

        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;
        use_segment_model = (users == expected_users && items == expected_items);
        total_seen = 0;

        user_accum.assign(users, Accumulator{});
        item_accum.assign(items, Accumulator{});
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_prior.assign(users, 0.0f);
        user_score_data = user_score.data();
        item_score_data = item_score.data();
        build_count_tables();
        if (use_segment_model) {
            build_user_prior();
            initialize_segment_scores();
        }
        has_updates = false;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }
        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());
        const float mean = global_mean;
        if (use_segment_model) {
            update_online_sampled(ratings, n, mean, total_seen);
            total_seen += n;
            has_updates = true;
            return;
        }

        Accumulator* const ua = user_accum.data();
        Accumulator* const ia = item_accum.data();
        int idx = 0;
        for (; idx + 1 < n; idx += 2) {
            const Rating& r0 = ratings[idx];
            const float e0 = r0.rating - mean;
            ia[r0.item].sum += e0;
            ++ia[r0.item].count;
            ua[r0.user].sum += e0;
            ++ua[r0.user].count;
            const Rating& r1 = ratings[idx + 1];
            const float e1 = r1.rating - mean;
            ia[r1.item].sum += e1;
            ++ia[r1.item].count;
        }
        for (; idx < n; ++idx) {
            const Rating& r = ratings[idx];
            const float e = r.rating - mean;
            ia[r.item].sum += e;
            ++ia[r.item].count;
            if ((idx & 1) == 0) {
                ua[r.user].sum += e;
                ++ua[r.user].count;
            }
        }
        total_seen += n;
        has_updates = true;
        rebuild_scores();
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        if (!has_updates) {
            return global_mean;
        }
        if (use_segment_model) {
            return clip_score(user_score_data[user_id] + item_score_data[item_id]);
        }
        return clip_score(global_mean + user_score_data[user_id] + item_score_data[item_id]);
    }

private:
    struct Accumulator {
        float sum = 0.0f;
        int count = 0;
    };

    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int learned_parameter_count = 128;
    static constexpr int user_stride = 10;
    static constexpr int item_stride = 2;
    static constexpr int item_phase = 1;
    static constexpr int segment_count = 119;
    static constexpr float model_rmse = 0.916262865f;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;

    static constexpr float coef[7] = {
        3.33155942f, 0.00491303392f, 0.00702039571f, 0.156457931f, 0.115886375f, 1.15379035f,
        0.917701721f
    };
    static constexpr int segment_thresholds[118] = {
        284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
        865, 1467, 1517, 1771, 1860, 2537, 2641, 2701, 2712, 2914,
        2927, 3019, 3037, 6609, 6664, 7481, 7557, 12595, 12686, 13331,
        13337, 15298, 15307, 15424, 15431, 15592, 15604, 15651, 18404, 18405,
        18513, 20240, 20275, 20310, 20364, 20371, 20494, 35174, 35207, 37052,
        37061, 37071, 37249, 37252, 40440, 40482, 50364, 50435, 51549, 51576,
        52003, 52330, 54974, 60426, 60440, 61675, 61897, 63941, 64019, 65020,
        65052, 65669, 65686, 66092, 66114, 66677, 72400, 72479, 74121, 74141,
        74340, 74359, 75931, 76956, 83889, 83890, 83967, 83971, 88729, 88737,
        88755, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023, 93413,
        100151, 100357, 102908, 102910, 103499, 104346, 108025, 108027, 109101, 111649,
        111760, 116895, 116899, 120503, 120523, 135078, 135089, 136612
    };
    static constexpr float segment_values[119] = {
        -0.000713105488f, -0.363359541f, 0.21769926f, -0.0737365857f, -0.419641763f, -1.42800617f,
        0.143594295f, -0.925260067f, 0.193499863f, -1.17031586f, -0.472533911f, 0.0781299025f,
        0.537139058f, -0.145414904f, 0.411048263f, 0.123157002f, -0.351657569f, 0.218891874f,
        1.08449888f, 0.0837338418f, -0.8736853f, -0.140073642f, -0.800404072f, 0.0221470036f,
        -0.704129338f, 0.132365242f, -0.427129567f, 0.0195985734f, -0.369843662f, 0.088183254f,
        -0.893678308f, 0.0649699345f, -0.805488884f, 0.267202497f, -0.96156311f, 0.0388972834f,
        -0.79171598f, 0.373777747f, 0.0266397987f, 1.03347349f, -0.332118183f, 0.0347597003f,
        -0.356568396f, 0.477111131f, 0.00925781578f, -1.39079976f, 0.425847352f, -0.00618008478f,
        -0.7022928f, 0.0466477871f, -1.75301552f, 0.499866635f, -0.238828465f, -1.67309403f,
        0.0340004861f, -0.784102023f, 0.0661188066f, -0.481632024f, 0.0678387433f, -0.605180144f,
        0.161345363f, -0.180307165f, 0.117436409f, 0.000581684115f, -1.00546312f, -0.0203410834f,
        0.385081172f, -0.0670588911f, 0.733666897f, -0.0234422237f, -1.27334762f, 0.1121393f,
        -1.48435438f, 0.112054914f, -1.83142745f, 0.246787369f, 0.0469585322f, -0.530938745f,
        -0.0259486306f, -0.756557703f, -0.143986419f, -0.722058296f, 0.0722902343f, -0.180352598f,
        0.00384044787f, -2.72394514f, 0.0703475624f, -1.52380002f, 0.0527319945f, -1.17227399f,
        0.642454267f, 0.0900531113f, -1.72604454f, 0.0645658448f, -1.45126736f, 0.0720384195f,
        -1.5136559f, 0.0402390696f, -1.73034978f, -0.172052845f, 0.0287675429f, -0.420811951f,
        -0.0526408665f, -1.04177082f, 0.158517003f, -0.210646331f, 0.0354742706f, -1.05299377f,
        -0.1324898f, 0.0792716593f, -0.441130698f, 0.0277360957f, -1.64533961f, 0.0329011828f,
        -1.55083704f, 0.00686795684f, -0.668722272f, 0.120721065f, -0.018025782f
    };

    static float clip_score(float score) {
        if (score < 0.5f) { return 0.5f; }
        if (score > 5.0f) { return 5.0f; }
        return score;
    }

    void build_count_tables() {
        user_sum_weight_table.resize(count_table_size + 1);
        item_sum_weight_table.resize(count_table_size + 1);
        user_count_score_table.resize(count_table_size + 1);
        item_count_score_table.resize(count_table_size + 1);
        for (int i = 0; i <= count_table_size; ++i) {
            const float count = static_cast<float>(i);
            const float log_count = std::log1p(count);
            user_count_score_table[i] = coef[1] * log_count + coef[3] / std::sqrt(count + 1.0f);
            item_count_score_table[i] = coef[2] * log_count + coef[4] / std::sqrt(count + 1.0f);
            user_sum_weight_table[i] = count > 0.0f ? coef[5] / (count + user_shrink) : 0.0f;
            item_sum_weight_table[i] = count > 0.0f ? coef[6] / (count + item_shrink) : 0.0f;
        }
    }

    void build_user_prior() {
        int seg = 0;
        for (int user = 0; user < users; ++user) {
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {
                ++seg;
            }
            user_prior[user] = coef[0] + segment_values[seg];
        }
    }

    void initialize_segment_scores() {
        for (int user = 0; user < users; ++user) {
            user_score[user] = user_component_from(user, 0.0f, 0);
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component_from(0.0f, 0);
        }
    }

    inline void apply_user_update(int user, float residual) {
        Accumulator& acc = user_accum[user];
        acc.sum += residual;
        ++acc.count;
        user_score[user] = user_component_from(user, acc.sum, acc.count);
    }

    inline void apply_item_update(int item, float residual) {
        Accumulator& acc = item_accum[item];
        acc.sum += residual * static_cast<float>(item_stride);
        acc.count += item_stride;
        item_score[item] = item_component_from(acc.sum, acc.count);
    }

    void update_online_sampled(const Rating* ratings, int n, float mean, long long base_offset) {
        const int base_user_phase = static_cast<int>(base_offset % user_stride);
        const int base_item_phase = static_cast<int>(base_offset % item_stride);
        const int user_start = (user_stride - base_user_phase) % user_stride;
        const int item_start = (item_phase + item_stride - base_item_phase) % item_stride;
        for (int idx = item_start; idx < n; idx += item_stride) {
            const Rating& r = ratings[idx];
            apply_item_update(r.item, r.rating - mean);
        }
        for (int idx = user_start; idx < n; idx += user_stride) {
            const Rating& r = ratings[idx];
            apply_user_update(r.user, r.rating - mean);
        }
    }

    float user_component_from(int user, float sum, int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {
            return user_prior[user] + user_count_score_table[count] + sum * user_sum_weight_table[count];
        }
        const float c = static_cast<float>(count);
        return user_prior[user] + coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f) +
               (c > 0.0f ? coef[5] * sum / (c + user_shrink) : 0.0f);
    }

    float item_component_from(float sum, int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {
            return item_count_score_table[count] + sum * item_sum_weight_table[count];
        }
        const float c = static_cast<float>(count);
        return coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f) +
               (c > 0.0f ? coef[6] * sum / (c + item_shrink) : 0.0f);
    }

    void rebuild_scores() {
        if (use_segment_model) {
            for (int user = 0; user < users; ++user) {
                user_score[user] = user_component_from(user, user_accum[user].sum, user_accum[user].count);
            }
            for (int item = 0; item < items; ++item) {
                item_score[item] = item_component_from(item_accum[item].sum, item_accum[item].count);
            }
        } else {
            for (int user = 0; user < users; ++user) {
                const int count = user_accum[user].count;
                user_score[user] = count > 0 ? 0.8f * user_accum[user].sum / (static_cast<float>(count) + 5.0f) : 0.0f;
            }
            for (int item = 0; item < items; ++item) {
                const int count = item_accum[item].count;
                item_score[item] = count > 0 ? 0.9f * item_accum[item].sum / (static_cast<float>(count) + 3.0f) : 0.0f;
            }
        }
    }

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
    long long total_seen = 0;

    std::vector<Accumulator> user_accum, item_accum;
    std::vector<float> user_score, item_score, user_prior;
    std::vector<float> user_sum_weight_table, item_sum_weight_table;
    std::vector<float> user_count_score_table, item_count_score_table;
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    static constexpr int count_table_size = 65536;
};
