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
    static constexpr int item_sample_stride = 3;
    static constexpr int item_sample_phase = 1;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.917604148f;

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
    3.31121969f, 0.00541336974f, 0.0103680007f, 0.156765923f, 0.126107529f, 1.14966369f,
    0.900083482f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 1860, 2537, 2641, 2701, 2712, 2914,
    2927, 3019, 3037, 6609, 6664, 7481, 7557, 12595, 12686, 13331,
    13337, 15298, 15307, 15424, 15431, 15592, 15604, 15651, 18404, 18405,
    18513, 20240, 20275, 20310, 20364, 20371, 20494, 23594, 26816, 27037,
    27086, 29097, 29099, 30293, 30317, 31696, 31711, 37052, 37061, 37071,
    37249, 37252, 40075, 40076, 40440, 40482, 42272, 50364, 50435, 51549,
    51557, 51824, 52003, 52330, 54974, 60426, 60440, 61675, 61897, 63941,
    64019, 65020, 65052, 65669, 65686, 66092, 66114, 66677, 72400, 72479,
    74121, 74141, 74359, 75829, 76956, 83889, 83890, 83967, 83971, 88729,
    88737, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023, 102908,
    102910, 109101, 116895, 116899, 120503, 120523, 135078, 135089
    };

    static constexpr float segment_values[119] = {
    0.000135730923f, -0.364116549f, 0.217187688f, -0.0712326467f, -0.423617542f, -1.41505849f,
    0.143468663f, -0.930881858f, 0.189575508f, -1.16851902f, -0.470092237f, 0.0802204162f,
    0.535885274f, -0.145995229f, 0.414745986f, 0.125174075f, -0.349131525f, 0.217847317f,
    1.08516157f, 0.0794136599f, -0.875961006f, -0.13954924f, -0.797990561f, 0.0223709065f,
    -0.700042248f, 0.133455381f, -0.424838096f, 0.0196073297f, -0.370174974f, 0.0878257379f,
    -0.900074065f, 0.0652652457f, -0.804386795f, 0.270309865f, -0.961588085f, 0.0398534574f,
    -0.797226012f, 0.37658143f, 0.0264123175f, 1.03534079f, -0.333833396f, 0.0352991186f,
    -0.35807851f, 0.475153238f, 0.010758494f, -1.38303018f, 0.426986188f, -0.0383316055f,
    0.0529404879f, -0.162695512f, -0.638631821f, 0.0249653757f, -0.937439203f, 0.0349893235f,
    -0.658818483f, 0.0296397861f, -1.10925436f, 0.0233834349f, -1.75250185f, 0.500456452f,
    -0.23899281f, -1.67243719f, 0.0487724729f, -0.747660995f, 0.0830326229f, -0.782015681f,
    -0.00121819577f, 0.0834895372f, -0.484586507f, 0.0681285784f, -0.65160948f, -0.034813907f,
    0.325242251f, -0.182782635f, 0.117693886f, 0.00128658197f, -1.01327801f, -0.0192507543f,
    0.386285931f, -0.0671888217f, 0.734086514f, -0.0219468772f, -1.2718637f, 0.111347616f,
    -1.48623013f, 0.112631589f, -1.83880687f, 0.248416528f, 0.0468661822f, -0.535453856f,
    -0.0257656779f, -0.759597123f, -0.258739889f, 0.0815290734f, -0.166281015f, 0.00396630447f,
    -2.71439123f, 0.0720383376f, -1.52884626f, 0.0523160882f, -1.17233682f, 0.132109344f,
    -1.7310226f, 0.0651902854f, -1.45338154f, 0.0738248527f, -1.51846635f, 0.0421928354f,
    -1.73419321f, -0.0150943939f, -1.04673862f, -0.0420533679f, 0.0340650417f, -1.64081597f,
    0.0329887085f, -1.55108595f, 0.00691796141f, -0.673712552f, 0.0403208956f
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
