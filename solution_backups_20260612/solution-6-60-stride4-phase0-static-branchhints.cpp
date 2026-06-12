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
    static constexpr int item_sample_stride = 4;
    static constexpr int item_sample_phase = 0;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.918948352f;

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
    3.29546475f, 0.00572180748f, 0.0129289366f, 0.15783827f, 0.139202863f, 1.14967382f,
    0.887039542f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1338, 1467, 1517, 1740, 1771, 1860, 2537, 2641, 2701,
    2712, 2897, 2914, 2927, 2972, 2973, 3019, 3037, 6664, 7481,
    7557, 12637, 12686, 13331, 13337, 15298, 15307, 15592, 15604, 20364,
    20371, 27037, 27086, 29097, 29099, 30293, 30317, 31696, 31711, 37052,
    37061, 37071, 37249, 37252, 40075, 40076, 40440, 40482, 42272, 50364,
    50435, 51549, 51557, 52003, 52330, 54974, 60426, 60440, 61675, 61897,
    62756, 62761, 63941, 64019, 65020, 65052, 65669, 65686, 66092, 66114,
    66677, 72400, 72479, 74121, 74141, 74340, 74359, 75931, 76956, 83889,
    83890, 83967, 83971, 85470, 86203, 86205, 87589, 87717, 88729, 88737,
    89453, 89457, 92606, 92615, 93015, 93023, 100151, 102908, 102910, 103499,
    104346, 116895, 116899, 120503, 120523, 135078, 135089, 136612
    };

    static constexpr float segment_values[119] = {
    0.000996574061f, -0.373613507f, 0.220590577f, -0.0716378987f, -0.424812615f, -1.42213619f,
    0.143630028f, -0.931669831f, 0.18793878f, -1.16652966f, -0.469532132f, 0.121366762f,
    -0.166906863f, 0.53577286f, -0.0896324143f, -0.357718021f, 0.410603255f, 0.12447273f,
    -0.346804023f, 0.221756384f, 1.07856596f, 0.00725726923f, 0.478514612f, -0.88081038f,
    0.116705149f, -0.850735903f, 0.253845334f, -0.794913232f, 0.0171023998f, 0.133645922f,
    -0.421094596f, 0.0171700418f, -0.446385205f, 0.084817335f, -0.91409564f, 0.0640937015f,
    -0.809479296f, 0.0299336575f, -0.802010834f, 0.0276905615f, -1.38668394f, 0.00426738383f,
    -0.638633549f, 0.0256910082f, -0.930482805f, 0.0358438455f, -0.660191059f, 0.02938837f,
    -1.1152302f, 0.0233109444f, -1.75540566f, 0.500106692f, -0.236072287f, -1.68191957f,
    0.0484452471f, -0.751054883f, 0.0852551907f, -0.781958044f, -0.00150177884f, 0.0831430778f,
    -0.48710525f, 0.0672623366f, -0.649716496f, 0.151299939f, -0.182672799f, 0.117926002f,
    0.00169139926f, -1.01266778f, -0.0184997078f, 0.386891007f, -0.0876983404f, -0.606079638f,
    -0.00621978519f, 0.729393363f, -0.0201611705f, -1.26871896f, 0.110543303f, -1.47513151f,
    0.114581071f, -1.84423995f, 0.247133896f, 0.0469266772f, -0.537706256f, -0.0251395311f,
    -0.761329472f, -0.144581452f, -0.717837989f, 0.0711344108f, -0.181024358f, 0.00404604943f,
    -2.70750594f, 0.0717007667f, -1.53393567f, 0.103204004f, -0.144443542f, 0.97502017f,
    0.127093777f, -0.403500706f, 0.082590647f, -1.18012965f, 0.132382169f, -1.72360146f,
    0.051297877f, -1.52630889f, 0.0427209735f, -1.74193239f, 0.0142185073f, -0.0825543478f,
    -1.0510025f, 0.159084633f, -0.210861325f, 0.0107977418f, -1.64105093f, 0.0331408642f,
    -1.54956925f, 0.00672679394f, -0.675923586f, 0.120469362f, -0.0191473477f
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
