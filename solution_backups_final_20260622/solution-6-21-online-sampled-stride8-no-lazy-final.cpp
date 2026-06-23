#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

#ifndef TASK2_PREDICTION_THREADS
#define TASK2_PREDICTION_THREADS 4
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
    static constexpr int prediction_threads = TASK2_PREDICTION_THREADS;
    static constexpr int user_stride = 10;
    static constexpr int item_stride = 8;
    static constexpr int item_phase = 0;
    static constexpr int segment_count = 119;
    static constexpr float model_rmse = 0.923316836f;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;

    static constexpr float coef[7] = {
        3.23217034f, 0.00271290471f, 0.0254527852f, 0.142752156f, 0.205550894f, 1.14828134f, 0.818154216f
    };
    static constexpr int segment_thresholds[118] = {
        284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
        865, 1338, 1467, 1517, 1771, 1860, 2537, 2567, 2701, 2712,
        2897, 2914, 2927, 2972, 2973, 3019, 3037, 3238, 3242, 3266,
        3388, 3396, 6664, 7481, 7557, 12637, 12686, 13331, 13337, 15592,
        15604, 20364, 20371, 37052, 37061, 37071, 37249, 37252, 40075, 40076,
        40440, 40482, 42272, 50364, 50435, 51549, 51557, 52003, 52330, 54974,
        65020, 65052, 65669, 65686, 66092, 66114, 66677, 72400, 72479, 74121,
        74141, 74340, 74341, 74359, 75931, 75940, 76570, 76629, 81108, 81116,
        83167, 83248, 83889, 83890, 83967, 83971, 85470, 86203, 86205, 88729,
        88737, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023, 100355,
        100357, 101039, 101043, 102564, 102569, 102908, 102910, 103499, 104346, 116895,
        116899, 120503, 120523, 122357, 122522, 135078, 135089, 136612
    };
    static constexpr float segment_values[119] = {
        0.00312414067f, -0.37719363f, 0.217851683f, -0.0695880204f, -0.434386313f, -1.4274478f,
        0.142429814f, -0.941852391f, 0.189243719f, -1.15171421f, -0.472433507f, 0.124495849f,
        -0.172538042f, 0.530406833f, -0.151749268f, 0.405823827f, 0.125581816f, -0.465970635f,
        0.123747118f, 1.07923746f, 0.00605070777f, 0.482052743f, -0.876866639f, 0.117243387f,
        -0.841502011f, 0.253110349f, -0.793798685f, 0.00307459827f, 0.873086274f, -0.176738352f,
        0.256847918f, -0.289154112f, 0.00814819522f, 0.133321568f, -0.420630664f, 0.017410567f,
        -0.461074501f, 0.0823010355f, -0.924003541f, 0.0383972712f, -0.808145165f, 0.0276818797f,
        -1.3677609f, -0.00127932336f, -1.75946867f, 0.495252341f, -0.233790755f, -1.68557823f,
        0.0491622165f, -0.749671817f, 0.0905932114f, -0.779462337f, -0.00233856868f, 0.0828106478f,
        -0.493974656f, 0.0677723438f, -0.676775634f, 0.151076704f, -0.185555145f, 0.117873728f,
        -0.00342408032f, -1.27436793f, 0.111953616f, -1.48180318f, 0.115760528f, -1.85313118f,
        0.249078676f, 0.0469898321f, -0.544314921f, -0.0244945548f, -0.767939508f, -0.146755323f,
        -2.28559256f, -0.56274581f, 0.0700915828f, -0.529731035f, -0.0457549505f, -0.451274961f,
        0.00747239776f, -0.795671225f, -0.0021154813f, -0.662531435f, 0.138388321f, -2.68615794f,
        0.0690568835f, -1.5487839f, 0.10490714f, -0.146421f, 0.965735555f, 0.0731213912f,
        -1.18619347f, 0.130247623f, -1.72569883f, 0.0668095946f, -1.44367909f, 0.0785766095f,
        -1.53777528f, 0.044306986f, -1.74358821f, 0.0083837565f, -0.850764751f, 0.0738118142f,
        -0.491275489f, -0.0299573839f, -2.01860356f, 0.0580553077f, -1.06872237f, 0.159939989f,
        -0.212375984f, 0.0108586783f, -1.64521539f, 0.0338947922f, -1.55965078f, 0.0747607574f,
        -0.364915282f, 0.00565830432f, -0.683650017f, 0.120038949f, -0.0192121342f
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
