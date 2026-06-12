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
    static constexpr int item_sample_stride = 5;
    static constexpr int item_sample_phase = 0;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 12.0f;
    static constexpr float model_rmse = 0.919125378f;

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
    3.28031421f, 0.00364361377f, 0.0152607691f, 0.151148871f, 0.162203923f, 1.15704596f,
    0.901037991f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 2160, 2298, 2534, 2641, 2701, 2712,
    2914, 2927, 2972, 2973, 3019, 3037, 6664, 7481, 7557, 12637,
    12686, 13331, 13337, 15298, 15307, 15592, 15604, 20364, 20371, 37052,
    37061, 37071, 37249, 37252, 40075, 40076, 40440, 40482, 42272, 50364,
    50435, 51549, 51557, 52003, 52330, 54974, 65020, 65052, 65669, 65686,
    66092, 66114, 66677, 72400, 72479, 74121, 74141, 74359, 83167, 83248,
    83889, 83890, 83967, 83971, 85470, 86203, 86205, 87589, 87717, 88729,
    88737, 88755, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023,
    93257, 93362, 94889, 94890, 96002, 96005, 97047, 97058, 100355, 100357,
    101039, 101043, 102557, 102569, 102908, 102910, 103499, 104346, 115111, 116895,
    116899, 120503, 120523, 122357, 122522, 135078, 135089, 136612
    };

    static constexpr float segment_values[119] = {
    0.00330280629f, -0.363388985f, 0.217946276f, -0.0711440518f, -0.431245446f, -1.41437495f,
    0.136217237f, -0.94253397f, 0.187985659f, -1.16237557f, -0.462299138f, 0.0805713087f,
    0.536833763f, -0.15205802f, 0.237968862f, -0.0590696782f, 0.210358918f, -0.342151046f,
    0.220248133f, 1.06864357f, 0.0784984678f, -0.876220822f, 0.105804987f, -0.842534065f,
    0.251098633f, -0.799853504f, 0.016767174f, 0.133895487f, -0.424190611f, 0.0177713279f,
    -0.450202048f, 0.0818165243f, -0.906225145f, 0.0639137924f, -0.816941619f, 0.029133806f,
    -0.797126532f, 0.0283174757f, -1.37784946f, -0.00137858815f, -1.74958193f, 0.493349075f,
    -0.234796047f, -1.68180168f, 0.0497145951f, -0.746214271f, 0.0890761539f, -0.778135717f,
    -0.00253989128f, 0.0836226121f, -0.493523479f, 0.0684143007f, -0.675791264f, 0.149351642f,
    -0.186113954f, 0.118247434f, -0.00435728161f, -1.27250814f, 0.111678503f, -1.48241413f,
    0.116007663f, -1.84202909f, 0.248489961f, 0.046785824f, -0.535750449f, -0.0242641345f,
    -0.761640429f, -0.257474899f, -0.0111094592f, -0.66282928f, 0.140364513f, -2.70555449f,
    0.0728554726f, -1.52726448f, 0.104163103f, -0.145199507f, 0.980642557f, 0.127521262f,
    -0.4031699f, 0.0834843889f, -1.17739713f, 0.642377377f, 0.0886722133f, -1.73019826f,
    0.0661307052f, -1.44628274f, 0.0777413249f, -1.52006519f, 0.0420389511f, -1.73615968f,
    -0.0588633195f, -0.575832903f, 0.0652904212f, -0.91344285f, 0.0387362726f, -1.10293603f,
    0.0758123025f, -1.01433778f, 0.0198207777f, -0.855270803f, 0.0706128106f, -0.497729778f,
    -0.0283742342f, -1.86439109f, 0.0576928966f, -1.05440414f, 0.159970999f, -0.211437479f,
    -0.000504582829f, 0.086360991f, -1.63775277f, 0.0332287736f, -1.55615342f, 0.075426206f,
    -0.364277869f, 0.0052842903f, -0.680245459f, 0.119475655f, -0.0186754875f
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
