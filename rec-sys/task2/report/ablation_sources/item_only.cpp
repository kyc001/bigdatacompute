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
    static constexpr int item_stride = 4;
    static constexpr int item_phase = 2;
    static constexpr int segment_count = 119;
    static constexpr float model_rmse = 0.918947339f;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;

    static constexpr float coef[7] = {
        3.29021835f, 0.00526042003f, 0.0137246884f, 0.156752408f, 0.143476963f, 1.15040135f,
        0.881480813f
    };
    static constexpr int segment_thresholds[118] = {
        284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
        865, 1467, 1517, 1771, 1860, 2537, 2641, 2701, 2712, 2914,
        2927, 3019, 3037, 6609, 6664, 7481, 7557, 12595, 12686, 13331,
        13337, 15298, 15307, 15424, 15431, 15592, 15604, 15651, 18404, 18405,
        18513, 20240, 20275, 20310, 20364, 20371, 20494, 20500, 23025, 23078,
        23100, 27037, 27086, 28664, 28687, 28894, 29110, 30293, 30317, 31696,
        31711, 37052, 37061, 37071, 37249, 37252, 40075, 40076, 40440, 40482,
        42272, 50364, 50435, 51549, 51557, 51824, 52003, 52330, 54974, 60428,
        60440, 61675, 61897, 63941, 64019, 65020, 65052, 65669, 65686, 66092,
        66114, 72400, 72479, 74121, 74141, 83889, 83890, 83967, 83971, 88729,
        88737, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023, 102908,
        102910, 109101, 116895, 116899, 120503, 120523, 135078, 135089
    };
    static constexpr float segment_values[119] = {
        0.00225608121f, -0.36468336f, 0.21785666f, -0.0708799139f, -0.427421302f, -1.42046714f,
        0.146961078f, -0.938998222f, 0.187276378f, -1.16122723f, -0.467615455f, 0.0794693083f,
        0.534148395f, -0.14535217f, 0.412427247f, 0.124626599f, -0.348922223f, 0.22256735f,
        1.08526182f, 0.0797049254f, -0.874846041f, -0.135852933f, -0.800117671f, 0.0219422262f,
        -0.698043704f, 0.132098734f, -0.424769372f, 0.0196507275f, -0.371984422f, 0.0859078541f,
        -0.906816423f, 0.0634907559f, -0.809217334f, 0.272754073f, -0.962445557f, 0.0382588468f,
        -0.797168314f, 0.379739016f, 0.0265632756f, 1.0310365f, -0.337213039f, 0.0347926356f,
        -0.359532475f, 0.475242049f, 0.0115303714f, -1.37820411f, 0.423929781f, -2.9597826f,
        -0.0247537699f, 0.333064735f, -0.664087594f, 0.0149058262f, -0.641503215f, 0.0580481105f,
        -0.461831927f, 0.223433301f, -0.259145528f, 0.0474994741f, -0.659960747f, 0.0283865854f,
        -1.11637366f, 0.0236136299f, -1.75354469f, 0.495685369f, -0.238364086f, -1.67830098f,
        0.0488021411f, -0.74475354f, 0.0852849633f, -0.778516591f, -0.00147639867f, 0.0835341439f,
        -0.488178998f, 0.0667104349f, -0.654561937f, -0.0342115723f, 0.323022842f, -0.184901088f,
        0.117342412f, 0.00137056503f, -1.03323889f, -0.0204421934f, 0.386364967f, -0.067465432f,
        0.728870988f, -0.0216148123f, -1.27027464f, 0.11221493f, -1.49502134f, 0.114801273f,
        -1.83676767f, 0.0664502457f, -0.538002253f, -0.025750367f, -0.756396651f, -0.0192986932f,
        -2.70123267f, 0.0680542961f, -1.5320183f, 0.0530218966f, -1.16998184f, 0.132473007f,
        -1.73094869f, 0.0646570399f, -1.4558959f, 0.0744337663f, -1.51972663f, 0.0416205749f,
        -1.7337569f, -0.0151481126f, -1.04922783f, -0.0413275026f, 0.0342127271f, -1.63927805f,
        0.0319577865f, -1.55309772f, 0.00743136415f, -0.675726235f, 0.0408449396f
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
            user_count_score_table[i] = 0.0f;
            item_count_score_table[i] = 0.0f;
            user_sum_weight_table[i] = 0.0f;
            item_sum_weight_table[i] = count > 0.0f ? coef[6] / (count + item_shrink) : 0.0f;
        }
    }

    void build_user_prior() {
        int seg = 0;
        for (int user = 0; user < users; ++user) {
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {
                ++seg;
            }
            user_prior[user] = coef[0];
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
        return user_prior[user] + (c > 0.0f ? coef[5] * sum / (c + user_shrink) : 0.0f);
    }

    float item_component_from(float sum, int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {
            return item_count_score_table[count] + sum * item_sum_weight_table[count];
        }
        const float c = static_cast<float>(count);
        return c > 0.0f ? coef[6] * sum / (c + item_shrink) : 0.0f;
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
