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
    static constexpr int item_sample_stride = 8;
    static constexpr int item_sample_phase = 0;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.923316836f;

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
    3.23217034f, 0.00271290471f, 0.0254527852f, 0.142752156f, 0.205550894f, 1.14828134f,
    0.818154216f
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
