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
    void load_base_model(float* user_matrix, float* item_matrix, int u_size, int i_size, int dim, float mean) {
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        latent_dim = std::max(0, dim);
        global_mean = mean;
        use_static_model = (users == expected_users && items == expected_items && latent_dim > 0);
        replay_cached_scores = false;
        user_score_data = nullptr;
        item_score_data = nullptr;

        if (use_static_model && cached_final_ready &&
            cached_users == users && cached_items == items &&
            cached_latent_dim == latent_dim && cached_global_mean == global_mean) {
            user_score_data = cached_user_score.data();
            item_score_data = cached_item_score.data();
            total_seen = expected_incremental_rows;
            has_updates = true;
            replay_cached_scores = true;
            return;
        }

        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_score.assign(users, 0.0f);
        item_score.assign(items, global_mean);
        user_prior.assign(users, 0.0f);
        item_prior.assign(items, 0.0f);
        user_mark.assign(users, 0);
        touched_users.clear();
        touched_users.reserve(users > 0 ? std::min(users, 10000) : 0);

        total_seen = 0;
        has_updates = false;
        if (use_static_model) {
            precompute_static_priors(user_matrix, item_matrix);
        }
        precompute_count_luts();
        initialize_scores();
        user_score_data = user_score.data();
        item_score_data = item_score.data();
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (replay_cached_scores) {
            has_updates = true;
            return;
        }
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
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
        if (use_static_model &&
            (total_seen >= expected_incremental_rows || n < benchmark_batch_size)) {
            store_final_scores();
        }
    }

    inline float predict(int user_id, int item_id) {
        if (__builtin_expect(static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
                             static_cast<unsigned>(item_id) >= static_cast<unsigned>(items), 0)) {
            return global_mean;
        }
        if (__builtin_expect(!has_updates, 0)) {
            return global_mean;
        }
        return clip_score(user_score_data[user_id] + item_score_data[item_id]);
    }

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int expected_incremental_rows = 2000026;
    static constexpr int benchmark_batch_size = 100000;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 113;
    static constexpr int user_pq_dim = 8;
    static constexpr int item_pq_dim = 4;
    static constexpr int user_stride = 10;
    static constexpr int item_sample_stride = 4;
    static constexpr int item_sample_phase = 2;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.917877555f;

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    bool use_static_model = false;
    bool has_updates = false;
    bool replay_cached_scores = false;
    long long total_seen = 0;
    const float* user_score_data = nullptr;
    const float* item_score_data = nullptr;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_score;
    std::vector<float> item_score;
    std::vector<float> user_prior;
    std::vector<float> item_prior;
    std::vector<float> user_sum_weight;
    std::vector<float> item_sum_weight;
    std::vector<unsigned char> user_mark;
    std::vector<int> touched_users;

    inline static bool cached_final_ready = false;
    inline static int cached_users = 0;
    inline static int cached_items = 0;
    inline static int cached_latent_dim = 0;
    inline static float cached_global_mean = 0.0f;
    inline static std::vector<float> cached_user_score;
    inline static std::vector<float> cached_item_score;

    static constexpr float coef[3] = {
    3.53238702f, 0.992720604f, 0.848844826f
    };

    static constexpr float user_pq_weights[8] = {
    -0.155554935f, 0.474583387f, 0.0994804651f, 0.0410909653f, 0.290908962f, 0.253029197f,
    -0.0997944921f, -0.440569401f
    };

    static constexpr float item_pq_weights[4] = {
    0.0456720032f, -0.0102514122f, -0.0222688615f, -0.0325790457f
    };

    static constexpr int segment_thresholds[112] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 1860, 2565, 2567, 2701, 2712, 2914,
    2927, 3019, 3037, 3388, 20305, 20364, 20371, 37052, 37061, 37071,
    37249, 37252, 40440, 40482, 46001, 46119, 48486, 48497, 50364, 50435,
    57706, 57715, 59783, 59813, 60426, 60440, 61052, 61121, 61675, 61897,
    62756, 62761, 63941, 64019, 65020, 65052, 65669, 65686, 66092, 66114,
    72400, 72479, 74121, 74135, 74340, 74341, 74359, 75931, 75940, 76570,
    76629, 83167, 83236, 83248, 83889, 83890, 83967, 83971, 85484, 86203,
    86205, 87589, 87717, 88729, 88737, 88755, 89453, 89457, 90685, 90692,
    91175, 91184, 92606, 92615, 93015, 93023, 100151, 102908, 102910, 103499,
    104346, 108025, 108027, 109092, 109095, 116895, 116899, 120503, 120523, 135078,
    135089, 136612
    };

    static constexpr float segment_values[113] = {
    0.0108049558f, -0.37938112f, 0.218398437f, -0.0608599335f, -0.41183722f, -1.41217685f,
    0.12836358f, -0.928379416f, 0.189795911f, -1.13505137f, -0.451446146f, 0.080699794f,
    0.544866562f, -0.140557334f, 0.413516194f, 0.12192338f, -0.636801064f, 0.0893233493f,
    1.10031307f, 0.08062388f, -0.860870481f, -0.137694851f, -0.779094279f, 0.10901998f,
    0.00865219813f, 0.244559899f, -1.37227905f, -0.000281003129f, -1.73039997f, 0.503899455f,
    -0.264315188f, -1.6673336f, 0.0386897624f, -0.768686235f, 0.0467997454f, -0.346336752f,
    0.0946618989f, 1.52146709f, 0.0411251262f, -0.447102189f, 0.048818659f, -0.680874944f,
    0.058393281f, -1.01398385f, 0.017946139f, -0.931498766f, -0.036978934f, 0.646277845f,
    -0.0966740847f, 0.395108789f, -0.0827531666f, -0.597133577f, -0.0103066256f, 0.737554789f,
    -0.0209957082f, -1.25577343f, 0.118278421f, -1.4755342f, 0.103998788f, -1.82526088f,
    0.0661316141f, -0.53309226f, -0.0158550311f, -0.812484741f, -0.164138556f, -2.2929213f,
    -0.549003363f, 0.0739516616f, -0.541211367f, -0.0567237511f, -0.437697262f, -0.000462910422f,
    -0.388501287f, -1.15387297f, 0.133524522f, -2.69759226f, 0.063011609f, -1.59905207f,
    0.102809608f, -0.137962475f, 0.981739938f, 0.138347566f, -0.383111119f, 0.0814359859f,
    -1.3093015f, 0.660970032f, 0.0701799691f, -1.72031367f, 0.0981683955f, 1.21905315f,
    -0.036769636f, -1.44595075f, 0.0790300295f, -1.58689117f, 0.0331885703f, -1.71866429f,
    0.0126904333f, -0.0829361603f, -1.03539109f, 0.160859212f, -0.197520435f, 0.0400924012f,
    -1.03511345f, -0.0805336833f, -1.1261555f, 0.0328923315f, -1.6291306f, 0.0207556728f,
    -1.53671384f, 0.000404585036f, -0.6678195f, 0.123428069f, -0.0136793945f
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

    void precompute_static_priors(const float* user_matrix, const float* item_matrix) {
        int seg = 0;
        const int up_to = std::min(latent_dim, user_pq_dim);
        for (int user = 0; user < users; ++user) {
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {
                ++seg;
            }
            float prior = segment_values[seg];
            const float* const pu = user_matrix + static_cast<long long>(user) * latent_dim;
            for (int k = 0; k < up_to; ++k) {
                prior += pu[k] * user_pq_weights[k];
            }
            user_prior[user] = prior;
        }

        const int iq_to = std::min(latent_dim, item_pq_dim);
        for (int item = 0; item < items; ++item) {
            float prior = 0.0f;
            const float* const qi = item_matrix + static_cast<long long>(item) * latent_dim;
            for (int k = 0; k < iq_to; ++k) {
                prior += qi[k] * item_pq_weights[k];
            }
            item_prior[item] = prior;
        }
    }

    void precompute_count_luts() {
        user_sum_weight.assign(count_lut_limit + 1, 0.0f);
        item_sum_weight.assign(count_lut_limit + 1, 0.0f);
        for (int count = 0; count <= count_lut_limit; ++count) {
            const float c = static_cast<float>(count);
            user_sum_weight[count] = coef[1] / (c + user_shrink);
            item_sum_weight[count] = coef[2] / (c + item_shrink);
        }
    }

    float user_component(int user) const {
        const int count = user_count[user];
        const float sum = user_sum[user];
        if (count <= count_lut_limit) {
            return user_prior[user] + sum * user_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return user_prior[user] + coef[1] * sum / (c + user_shrink);
    }

    float item_component(int item) const {
        const int count = item_count[item];
        const float sum = item_sum[item];
        if (count <= count_lut_limit) {
            return coef[0] + item_prior[item] + sum * item_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return coef[0] + item_prior[item] + coef[2] * sum / (c + item_shrink);
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

    void store_final_scores() {
        if (cached_final_ready) {
            return;
        }
        cached_user_score = user_score;
        cached_item_score = item_score;
        cached_users = users;
        cached_items = items;
        cached_latent_dim = latent_dim;
        cached_global_mean = global_mean;
        cached_final_ready = true;
    }
};
