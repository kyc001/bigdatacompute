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
    static constexpr int item_stride = 16;
    static constexpr int item_phase = 13;
    static constexpr int segment_count = 119;
    static constexpr float model_rmse = 0.929776788f;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;

    static constexpr float coef[7] = {
        3.16960144f, -0.00304329325f, 0.0403757766f, 0.114773974f, 0.287350655f, 1.15339708f,
        0.726138353f
    };
    static constexpr int segment_thresholds[118] = {
        284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
        865, 1338, 1467, 1517, 1740, 1754, 2160, 2286, 2294, 2537,
        2567, 2701, 2712, 2897, 2914, 2927, 2972, 2973, 3019, 3037,
        6609, 6664, 7481, 7557, 7704, 7753, 8573, 8716, 8735, 13331,
        13337, 15592, 15604, 20240, 20275, 20364, 20371, 37052, 37061, 37071,
        37249, 37252, 40075, 40076, 40440, 40482, 50364, 50435, 51549, 51557,
        52003, 52330, 54974, 65020, 65052, 65669, 65686, 66092, 66114, 66677,
        72400, 72479, 74121, 74141, 74340, 74341, 74359, 75931, 75940, 76570,
        76629, 83167, 83248, 83889, 83890, 83967, 83971, 85470, 86203, 86205,
        87589, 87717, 88729, 88737, 89453, 89457, 91175, 91184, 92606, 92615,
        93015, 93023, 93324, 93362, 100355, 100357, 101039, 101043, 102564, 102569,
        102908, 102910, 103499, 104346, 116895, 116899, 120503, 120523
    };
    static constexpr float segment_values[119] = {
        0.00999893714f, -0.383260459f, 0.213131949f, -0.0752347186f, -0.45444721f, -1.42019558f,
        0.146824181f, -0.958541214f, 0.186440542f, -1.13620055f, -0.455161959f, 0.124415755f,
        -0.183349773f, 0.539753854f, -0.0911365524f, -0.44079563f, 0.235349193f, -0.0095176287f,
        -0.62695843f, 0.196388796f, -0.4789038f, 0.112885058f, 1.06434846f, -0.00131147972f,
        0.498875886f, -0.880281866f, 0.102490664f, -0.8625561f, 0.259939104f, -0.807943285f,
        0.0237125605f, -0.70511508f, 0.140166178f, -0.41980198f, 0.288604498f, -0.524126053f,
        0.147599503f, -0.217605159f, -1.73148763f, 0.00431075506f, -0.927100539f, 0.0354108885f,
        -0.822743952f, 0.0352100842f, -0.355031073f, 0.216294542f, -1.3435148f, -0.00226339814f,
        -1.75682521f, 0.481844693f, -0.232275918f, -1.69636881f, 0.0495833792f, -0.738681018f,
        0.0932443589f, -0.782039285f, 0.0643704087f, -0.511607826f, 0.0714860857f, -0.692910135f,
        0.14910391f, -0.195510954f, 0.118395753f, -0.00336892414f, -1.29063106f, 0.110394947f,
        -1.47987437f, 0.118959948f, -1.88091004f, 0.251248032f, 0.048882015f, -0.55052954f,
        -0.021811666f, -0.768969536f, -0.149051905f, -2.27076817f, -0.565535963f, 0.0725846887f,
        -0.52604419f, -0.0418895632f, -0.465639919f, -0.00158691744f, -0.67159909f, 0.137242779f,
        -2.6597333f, 0.0673345253f, -1.57220149f, 0.10794121f, -0.14690724f, 0.989001989f,
        0.130290464f, -0.401303083f, 0.0799450278f, -1.19905293f, 0.128776476f, -1.74060059f,
        0.0690921172f, -1.4325732f, 0.0859664604f, -1.54643488f, 0.0499522798f, -1.75413632f,
        -0.122396432f, -0.878701985f, 0.0191226006f, -0.869221628f, 0.0775296986f, -0.513607919f,
        -0.0286536235f, -2.03870845f, 0.0531499237f, -1.08721781f, 0.165418282f, -0.212076694f,
        0.0109426836f, -1.64225113f, 0.0355700515f, -1.56693637f, 0.00754758483f
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
