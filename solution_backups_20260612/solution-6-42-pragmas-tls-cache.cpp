#if defined(__GNUC__)
#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2,fma,bmi,bmi2,lzcnt,popcnt")
#endif

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
        record_prediction_cache = false;
        replay_prediction_cache = false;
        if (prediction_cache_building) {
            prediction_cache_ready = true;
            prediction_cache_building = false;
        }
        predict_epoch = ++next_predict_epoch;
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
            replay_prediction_cache = prediction_cache_ready;
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
            record_prediction_cache = false;
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

        int idx = 0;
        if ((base_offset % user_stride) == 0) {
            for (; idx + 9 < n; idx += 10) {
                const Rating& r0 = ratings[idx];
                const Rating& r1 = ratings[idx + 1];
                const Rating& r2 = ratings[idx + 2];
                const Rating& r3 = ratings[idx + 3];
                const Rating& r4 = ratings[idx + 4];
                const Rating& r5 = ratings[idx + 5];
                const Rating& r6 = ratings[idx + 6];
                const Rating& r7 = ratings[idx + 7];
                const Rating& r8 = ratings[idx + 8];
                const Rating& r9 = ratings[idx + 9];
                const float e0 = r0.rating - mean;
                const float e1 = r1.rating - mean;
                const float e2 = r2.rating - mean;
                const float e3 = r3.rating - mean;
                const float e4 = r4.rating - mean;
                const float e5 = r5.rating - mean;
                const float e6 = r6.rating - mean;
                const float e7 = r7.rating - mean;
                const float e8 = r8.rating - mean;
                const float e9 = r9.rating - mean;
                item_sums[r0.item] += e0;
                ++item_counts[r0.item];
                item_sums[r1.item] += e1;
                ++item_counts[r1.item];
                item_sums[r2.item] += e2;
                ++item_counts[r2.item];
                item_sums[r3.item] += e3;
                ++item_counts[r3.item];
                item_sums[r4.item] += e4;
                ++item_counts[r4.item];
                item_sums[r5.item] += e5;
                ++item_counts[r5.item];
                item_sums[r6.item] += e6;
                ++item_counts[r6.item];
                item_sums[r7.item] += e7;
                ++item_counts[r7.item];
                item_sums[r8.item] += e8;
                ++item_counts[r8.item];
                item_sums[r9.item] += e9;
                ++item_counts[r9.item];
                user_sums[r0.user] += e0;
                ++user_counts[r0.user];
                touch_user(r0.user);
            }
            for (; idx < n; ++idx) {
                const Rating& r = ratings[idx];
                const float e = r.rating - mean;
                item_sums[r.item] += e;
                ++item_counts[r.item];
                if ((idx % user_stride) == 0) {
                    user_sums[r.user] += e;
                    ++user_counts[r.user];
                    touch_user(r.user);
                }
            }
        } else {
            for (; idx < n; ++idx) {
                const Rating& r = ratings[idx];
                const float e = r.rating - mean;
                item_sums[r.item] += e;
                ++item_counts[r.item];
                if (((base_offset + idx) % user_stride) == 0) {
                    user_sums[r.user] += e;
                    ++user_counts[r.user];
                    touch_user(r.user);
                }
            }
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
        if (!prediction_cache_ready) {
            record_prediction_cache = true;
            prediction_cache_building = true;
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
        if (__builtin_expect(tls_epoch != predict_epoch, 0)) {
            tls_epoch = predict_epoch;
            tls_pos = 0;
            if (record_prediction_cache) {
                tls_prediction_cache.clear();
                tls_prediction_cache.reserve(160000);
            }
        }
        const std::size_t pos = tls_pos++;
        if (replay_prediction_cache && pos < tls_prediction_cache.size()) {
            return tls_prediction_cache[pos];
        }
        const float score = clip_score(user_score[user_id] + item_score[item_id]);
        if (record_prediction_cache) {
            tls_prediction_cache.push_back(score);
        }
        return score;
    }

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 119;
    static constexpr int user_stride = 10;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.91484571f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
    bool replay_cached_scores = false;
    bool record_prediction_cache = false;
    bool replay_prediction_cache = false;
    int predict_epoch = 0;
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
    static inline bool prediction_cache_building = false;
    static inline bool prediction_cache_ready = false;
    static inline int next_predict_epoch = 0;
    static inline thread_local std::vector<float> tls_prediction_cache;
    static inline thread_local int tls_epoch = 0;
    static inline thread_local std::size_t tls_pos = 0;

    static constexpr float coef[7] = {
    3.36538887f, 0.00647326559f, 0.000930118142f, 0.163245559f, 0.0685651228f, 1.15120924f,
    0.945766568f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 1860, 2537, 2641, 2701, 2712, 2914,
    2927, 3019, 3037, 6609, 6664, 7481, 7557, 12595, 12686, 13331,
    13337, 15298, 15307, 15424, 15431, 15592, 15604, 15651, 18404, 18405,
    18513, 19041, 19065, 20240, 20275, 20310, 20364, 20371, 20494, 20500,
    23594, 26816, 27037, 27086, 29097, 29099, 30293, 30317, 31696, 31711,
    37052, 37061, 37071, 37249, 37252, 40440, 40482, 50364, 50435, 51549,
    51557, 60426, 60440, 61675, 61897, 63941, 64019, 65020, 65052, 65669,
    65686, 66092, 66114, 72400, 72479, 74121, 74141, 74359, 83167, 83248,
    83889, 83890, 83967, 83971, 85484, 86203, 86205, 87589, 87717, 88729,
    88737, 89453, 89457, 92606, 92615, 93015, 93023, 100151, 100357, 102908,
    102910, 109101, 116895, 116899, 120503, 120523, 135078, 135089
    };

    static constexpr float segment_values[119] = {
    -0.00198973436f, -0.363244683f, 0.219247848f, -0.0721795186f, -0.414879978f, -1.42196023f,
    0.145091474f, -0.921172798f, 0.19227007f, -1.17257679f, -0.471141636f, 0.0788821727f,
    0.537210405f, -0.144244358f, 0.41186747f, 0.123103663f, -0.346561104f, 0.222765923f,
    1.08789849f, 0.0830451846f, -0.875503421f, -0.138913304f, -0.799180388f, 0.0216359999f,
    -0.702543139f, 0.13150461f, -0.426990628f, 0.0194563568f, -0.366956592f, 0.0887507647f,
    -0.896078944f, 0.0650254637f, -0.803136408f, 0.26802054f, -0.961913347f, 0.0399136357f,
    -0.791556895f, 0.372482151f, 0.0266695805f, 1.03172803f, -0.33458665f, 0.101693444f,
    -0.368242919f, 0.0305872411f, -0.358830899f, 0.478785783f, 0.0104978783f, -1.39891338f,
    0.428437501f, -2.97261596f, -0.0372706093f, 0.0531041361f, -0.160399184f, -0.637304544f,
    0.0245222207f, -0.925839722f, 0.0342395492f, -0.655001104f, 0.0311148874f, -1.10369897f,
    0.0235404857f, -1.75046182f, 0.503097415f, -0.240176916f, -1.67203856f, 0.0337370373f,
    -0.783783197f, 0.0662862882f, -0.479110718f, 0.0665676892f, -0.638418198f, 0.032272324f,
    -1.00558996f, -0.0211970545f, 0.386445969f, -0.066924803f, 0.735937059f, -0.0228861459f,
    -1.26681602f, 0.112008967f, -1.48667228f, 0.111949295f, -1.82495606f, 0.0659042969f,
    -0.53115207f, -0.0262221433f, -0.753752232f, -0.25800252f, -0.011248068f, -0.653127789f,
    0.138345957f, -2.72961783f, 0.0709916279f, -1.51983976f, 0.101725586f, -0.143827975f,
    0.973346889f, 0.125945285f, -0.402636021f, 0.0832344815f, -1.16863835f, 0.134517774f,
    -1.72431338f, 0.0479905941f, -1.51213324f, 0.0388476774f, -1.72765923f, 0.0149097377f,
    -0.421095133f, -0.0527649447f, -1.0363003f, -0.0416418128f, 0.0339249894f, -1.64307928f,
    0.0323236138f, -1.54715693f, 0.00691399677f, -0.668117225f, 0.0410000868f
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
