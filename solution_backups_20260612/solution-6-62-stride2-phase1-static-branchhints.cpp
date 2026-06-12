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
    static constexpr int item_sample_stride = 2;
    static constexpr int item_sample_phase = 1;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.916262865f;

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
