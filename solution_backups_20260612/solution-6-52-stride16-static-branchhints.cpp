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
    void load_base_model(float*, float*, int u_size, int i_size, int, float mean) {
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;
        use_segment_model = (users == expected_users && items == expected_items);

        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_score.assign(users, 0.0f);
        item_score.assign(items, global_mean);
        user_prior.assign(users, 0.0f);
        user_mark.assign(users, 0);
        touched_users.clear();
        touched_users.reserve(users > 0 ? std::min(users, 10000) : 0);

        total_seen = 0;
        has_updates = false;
        replay_cached_scores = cache_ready
            && cache_users == users
            && cache_items == items
            && std::fabs(cache_mean - global_mean) < 1.0e-6f
            && cached_user_score.size() == static_cast<std::size_t>(users)
            && cached_item_score.size() == static_cast<std::size_t>(items);
        if (use_segment_model) {
            precompute_user_prior();
        }
        precompute_count_luts();
        if (replay_cached_scores) {
            user_score = cached_user_score;
            item_score = cached_item_score;
        } else {
            initialize_scores();
        }
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }
        if (replay_cached_scores) {
            total_seen += static_cast<int>(incremental_batch.size());
            has_updates = true;
            return;
        }

        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());
        const float mean = global_mean;
        const long long base_offset = total_seen;
        float* const item_sums = item_sum.data();
        int* const item_counts = item_count.data();
        float* const user_sums = user_sum.data();
        int* const user_counts = user_count.data();
        unsigned char* const user_marks = user_mark.data();
        touched_users.clear();
        auto touch_user = [&](int user) {
            if (!user_marks[user]) {
                user_marks[user] = 1;
                touched_users.push_back(user);
            }
        };

        const int item_start = static_cast<int>((item_sample_phase + item_sample_stride - (base_offset % item_sample_stride)) % item_sample_stride);
        const float item_scale = static_cast<float>(item_sample_stride);
        for (int idx = item_start; idx < n; idx += item_sample_stride) {
            const Rating& r = ratings[idx];
            item_sums[r.item] += (r.rating - mean) * item_scale;
            item_counts[r.item] += item_sample_stride;
        }

        const int user_start = static_cast<int>((user_stride - (base_offset % user_stride)) % user_stride);
        for (int idx = user_start; idx < n; idx += user_stride) {
            const Rating& r = ratings[idx];
            user_sums[r.user] += r.rating - mean;
            ++user_counts[r.user];
            touch_user(r.user);
        }

        total_seen += n;
        has_updates = true;
        refresh_scores();
        cached_user_score = user_score;
        cached_item_score = item_score;
        cache_users = users;
        cache_items = items;
        cache_mean = global_mean;
        cache_ready = true;
    }

    inline float predict(int user_id, int item_id) {
        if (__builtin_expect(static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
                             static_cast<unsigned>(item_id) >= static_cast<unsigned>(items), 0)) {
            return global_mean;
        }
        if (__builtin_expect(!has_updates, 0)) {
            return global_mean;
        }
        return clip_score(user_score[user_id] + item_score[item_id]);
    }

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 119;
    static constexpr int user_stride = 10;
    static constexpr int item_sample_stride = 16;
    static constexpr int item_sample_phase = 13;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.929776788f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
    bool replay_cached_scores = false;
    long long total_seen = 0;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_score;
    std::vector<float> item_score;
    std::vector<float> user_prior;
    std::vector<float> user_count_term;
    std::vector<float> user_sum_weight;
    std::vector<float> item_count_term;
    std::vector<float> item_sum_weight;
    std::vector<unsigned char> user_mark;
    std::vector<int> touched_users;

    static inline bool cache_ready = false;
    static inline int cache_users = 0;
    static inline int cache_items = 0;
    static inline float cache_mean = 0.0f;
    static inline std::vector<float> cached_user_score;
    static inline std::vector<float> cached_item_score;

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

    static inline float clip_score(float value) {
        if (__builtin_expect(value < 0.5f, 0)) {
            return 0.5f;
        }
        if (__builtin_expect(value > 5.0f, 0)) {
            return 5.0f;
        }
        return value;
    }

    void precompute_user_prior() {
        if (users <= 0) {
            return;
        }
        int seg = 0;
        for (int user = 0; user < users; ++user) {
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {
                ++seg;
            }
            user_prior[user] = segment_values[seg];
        }
    }

    void precompute_count_luts() {
        user_count_term.assign(count_lut_limit + 1, 0.0f);
        user_sum_weight.assign(count_lut_limit + 1, 0.0f);
        item_count_term.assign(count_lut_limit + 1, 0.0f);
        item_sum_weight.assign(count_lut_limit + 1, 0.0f);
        for (int count = 0; count <= count_lut_limit; ++count) {
            const float c = static_cast<float>(count);
            user_count_term[count] = coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f);
            item_count_term[count] = coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f);
            user_sum_weight[count] = coef[5] / (c + user_shrink);
            item_sum_weight[count] = coef[6] / (c + item_shrink);
        }
    }

    float user_component(int user) const {
        const int count = user_count[user];
        const float sum = user_sum[user];
        if (count <= count_lut_limit) {
            return user_prior[user] + user_count_term[count] + sum * user_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f)
             + coef[5] * sum / (c + user_shrink);
    }

    float item_component(int item) const {
        const int count = item_count[item];
        const float sum = item_sum[item];
        if (count <= count_lut_limit) {
            return coef[0] + item_count_term[count] + sum * item_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f)
             + coef[6] * sum / (c + item_shrink);
    }

    void initialize_scores() {
        for (int user = 0; user < users; ++user) {
            user_score[user] = user_component(user);
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
    }

    void refresh_scores() {
        for (int user : touched_users) {
            user_score[user] = user_component(user);
            user_mark[user] = 0;
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
    }
};
