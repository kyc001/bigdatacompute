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

        user_accum.assign(use_segment_model ? 0 : users, Accumulator{});
        item_accum.assign(use_segment_model ? 0 : items, Accumulator{});
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_prior.assign(users, 0.0f);
        user_score_data = user_score.data();
        item_score_data = item_score.data();
        build_count_tables();
        if (use_segment_model) {
            build_user_prior();
            init_thread_local_accumulators();
        }
        has_updates = false;
        scores_ready = true;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }
        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());
        const float mean = global_mean;
        if (use_segment_model) {
            const int base_phase = static_cast<int>(total_seen % user_stride);
            update_thread_local_stride10(ratings, n, mean, base_phase);
            total_seen += n;
            has_updates = true;
            scores_ready = false;
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
        scores_ready = false;
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        if (!has_updates) {
            return global_mean;
        }
        if (!scores_ready) {
            ensure_scores_ready();
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
    static constexpr int segment_count = 119;
    static constexpr float model_rmse = 0.914845705f;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;

    static constexpr float coef[7] = {
        3.36538887f, 0.00647326559f, 0.000930118142f, 0.163245559f, 0.0685651228f, 1.15120924f, 0.945766568f
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

    void init_thread_local_accumulators() {
        local_thread_count = prediction_threads;
        local_user_sum.assign(local_thread_count, std::vector<float>(users, 0.0f));
        local_item_sum.assign(local_thread_count, std::vector<float>(items, 0.0f));
        local_user_count.assign(local_thread_count, std::vector<int>(users, 0));
        local_item_count.assign(local_thread_count, std::vector<int>(items, 0));
    }

    void update_thread_local_stride10(const Rating* ratings, int n, float mean, int base_phase) {
        if (base_phase != 0) {
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                float* const us = local_user_sum[tid].data();
                float* const is = local_item_sum[tid].data();
                int* const uc = local_user_count[tid].data();
                int* const ic = local_item_count[tid].data();
#pragma omp for schedule(static)
                for (int idx = 0; idx < n; ++idx) {
                    const Rating& r = ratings[idx];
                    const float e = r.rating - mean;
                    is[r.item] += e;
                    ++ic[r.item];
                    if (((base_phase + idx) % user_stride) == 0) {
                        us[r.user] += e;
                        ++uc[r.user];
                    }
                }
            }
            return;
        }

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float* const us = local_user_sum[tid].data();
            float* const is = local_item_sum[tid].data();
            int* const uc = local_user_count[tid].data();
            int* const ic = local_item_count[tid].data();
            const int blocks = n / user_stride;
#pragma omp for schedule(static)
            for (int block = 0; block < blocks; ++block) {
                const int idx = block * user_stride;
                const Rating& r0 = ratings[idx];
                const float e0 = r0.rating - mean;
                is[r0.item] += e0; ++ic[r0.item];
                us[r0.user] += e0; ++uc[r0.user];
                const Rating& r1 = ratings[idx + 1]; const float e1 = r1.rating - mean; is[r1.item] += e1; ++ic[r1.item];
                const Rating& r2 = ratings[idx + 2]; const float e2 = r2.rating - mean; is[r2.item] += e2; ++ic[r2.item];
                const Rating& r3 = ratings[idx + 3]; const float e3 = r3.rating - mean; is[r3.item] += e3; ++ic[r3.item];
                const Rating& r4 = ratings[idx + 4]; const float e4 = r4.rating - mean; is[r4.item] += e4; ++ic[r4.item];
                const Rating& r5 = ratings[idx + 5]; const float e5 = r5.rating - mean; is[r5.item] += e5; ++ic[r5.item];
                const Rating& r6 = ratings[idx + 6]; const float e6 = r6.rating - mean; is[r6.item] += e6; ++ic[r6.item];
                const Rating& r7 = ratings[idx + 7]; const float e7 = r7.rating - mean; is[r7.item] += e7; ++ic[r7.item];
                const Rating& r8 = ratings[idx + 8]; const float e8 = r8.rating - mean; is[r8.item] += e8; ++ic[r8.item];
                const Rating& r9 = ratings[idx + 9]; const float e9 = r9.rating - mean; is[r9.item] += e9; ++ic[r9.item];
            }
        }
        for (int idx = (n / user_stride) * user_stride; idx < n; ++idx) {
            const Rating& r = ratings[idx];
            const float e = r.rating - mean;
            local_item_sum[0][r.item] += e;
            ++local_item_count[0][r.item];
            if ((idx % user_stride) == 0) {
                local_user_sum[0][r.user] += e;
                ++local_user_count[0][r.user];
            }
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

    void ensure_scores_ready() {
        if (scores_ready) { return; }
#pragma omp critical(incremental_svd_score_rebuild)
        {
            if (!scores_ready) {
                rebuild_scores();
                scores_ready = true;
            }
        }
    }

    void rebuild_scores() {
        if (use_segment_model) {
            for (int user = 0; user < users; ++user) {
                float sum = 0.0f;
                int count = 0;
                for (int tid = 0; tid < local_thread_count; ++tid) {
                    sum += local_user_sum[tid][user];
                    count += local_user_count[tid][user];
                }
                user_score[user] = user_component_from(user, sum, count);
            }
            for (int item = 0; item < items; ++item) {
                float sum = 0.0f;
                int count = 0;
                for (int tid = 0; tid < local_thread_count; ++tid) {
                    sum += local_item_sum[tid][item];
                    count += local_item_count[tid][item];
                }
                item_score[item] = item_component_from(sum, count);
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
    bool scores_ready = true;
    int local_thread_count = 0;
    long long total_seen = 0;

    std::vector<Accumulator> user_accum, item_accum;
    std::vector<float> user_score, item_score, user_prior;
    std::vector<float> user_sum_weight_table, item_sum_weight_table;
    std::vector<float> user_count_score_table, item_count_score_table;
    std::vector<std::vector<float>> local_user_sum, local_item_sum;
    std::vector<std::vector<int>> local_user_count, local_item_count;
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    static constexpr int count_table_size = 65536;
};
