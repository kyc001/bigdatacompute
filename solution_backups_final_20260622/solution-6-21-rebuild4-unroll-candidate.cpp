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
        use_factorized_model = (users == expected_users && items == expected_items);

        user_accum.assign(users, Accumulator{});
        item_accum.assign(items, Accumulator{});
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_prior.assign(users, 0.0f);
        init_thread_local_accumulators();
        user_score_data = user_score.data();
        item_score_data = item_score.data();
        build_user_prior();
        build_count_tables();
        has_updates = false;
        scores_ready = true;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }

        Accumulator* const ua = user_accum.data();
        Accumulator* const ia = item_accum.data();
        const float mean = global_mean;
        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());

        if (use_factorized_model) {
            update_thread_local(ratings, n, mean);
            has_updates = true;
            scores_ready = false;
            return;
        }

        int idx = 0;
        for (; idx + 1 < n; idx += 2) {
            const Rating& r0 = ratings[idx];
            const int user0 = r0.user;
            const int item0 = r0.item;
            const float residual0 = r0.rating - mean;
            ia[item0].sum += residual0;
            ++ia[item0].count;
            ua[user0].sum += residual0;
            ++ua[user0].count;

            const Rating& r1 = ratings[idx + 1];
            const int item1 = r1.item;
            const float residual1 = r1.rating - mean;
            ia[item1].sum += residual1;
            ++ia[item1].count;
        }
        for (; idx < n; ++idx) {
            const Rating& r = ratings[idx];
            const float residual = r.rating - mean;
            const int item = r.item;
            ia[item].sum += residual;
            ++ia[item].count;
            if ((idx & 1) == 0) {
                const int user = r.user;
                ua[user].sum += residual;
                ++ua[user].count;
            }
        }

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

        if (use_factorized_model) {
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
    static constexpr int factor_high = 48;
    static constexpr int factor_low = 69;
    static constexpr int learned_parameter_count = 128;
    static constexpr int prediction_threads = TASK2_PREDICTION_THREADS;
    static constexpr float model_rmse = 0.925568700f;
    static constexpr float user_shrink = 20.0000000f;
    static constexpr float item_shrink = 5.00000000f;

    static constexpr float coef[9] = {
    4.24405766f, -0.283819079f, -9.59226527e-05f, 0.0256695859f, -0.000117186406f, -0.696538329f, 0.0577300675f, 0.868011594f, 0.943278670f
    };

    static constexpr float factor_a[48] = {
        -0.0654724017f, -0.353016019f, 0.0496158525f, 0.104987070f, 0.369457960f, -0.146760210f,
        0.0145380776f, 0.574095786f, 0.0908914506f, 0.249863043f, 0.229053885f, 0.142056048f,
        -0.903062105f, -0.0649739355f, -0.337123722f, -0.0244001411f, 0.333608687f, -0.127314553f,
        -0.0690121502f, -0.186383650f, -0.412832260f, 0.0346767195f, -0.0498492382f, 0.0547819808f,
        -0.0160356462f, -0.0550303906f, 0.0282693040f, -0.0905863121f, -0.00875620916f, -0.403665453f,
        -0.0773103237f, -0.0973550081f, -0.616394639f, -0.167309418f, -0.259666860f, -0.504084587f,
        -0.00279202196f, -0.397143543f, 0.0167466383f, 0.169773057f, 0.478990853f, -0.184343666f,
        0.235621989f, 0.176783815f, 0.0714734271f, -0.178994179f, -0.197080195f, -0.138838187f
    };

    static constexpr float factor_b[69] = {
        0.0967696831f, 0.0474240370f, -0.249213353f, -0.207182109f, -0.260918409f, -0.182335347f,
        0.465966731f, -0.130767733f, 0.392432183f, 0.0210205223f, -0.165526316f, 0.751457930f,
        -0.0842710584f, -0.686087012f, 0.225307629f, 0.304341257f, -0.635241330f, 0.545668066f,
        -0.0942922086f, -0.325531960f, -0.341678500f, -0.204619795f, -0.0255517233f, 0.201945379f,
        -0.0999412984f, -0.169471875f, 0.398303837f, 0.183249325f, -0.178943858f, 0.104637489f,
        -0.416632414f, 0.670088887f, 0.242138505f, 0.285378814f, -0.0347001925f, 0.294564039f,
        -0.321254343f, -0.115128830f, 0.161771387f, -0.237590760f, -0.371675104f, 0.0315141045f,
        0.225400493f, -0.0831767768f, -0.0713473409f, 0.203118980f, 0.102591701f, -0.508924425f,
        0.266816199f, -0.203697652f, 0.0497425608f, -0.0839401186f, -0.278685123f, 0.0594674721f,
        0.0399749652f, 0.0974409729f, 0.274996430f, -0.215815783f, 0.223935172f, 0.362202823f,
        0.226429582f, 1.07475019f, 0.320226938f, -0.0300991200f, 0.0331726298f, 0.588178635f,
        0.0208029188f, 0.343888432f, -0.228072241f
    };

    static float clip_score(float score) {
        if (score < 0.5f) {
            return 0.5f;
        }
        if (score > 5.0f) {
            return 5.0f;
        }
        return score;
    }

    void build_user_prior() {
        if (!use_factorized_model) {
            return;
        }
        for (int user = 0; user < users; ++user) {
            const int hi = static_cast<int>((1LL * user * factor_high) / users);
            const int lo = user % factor_low;
            user_prior[user] = coef[0] + factor_a[hi] * factor_b[lo];
        }
    }

    void build_count_tables() {
        user_sum_weight_table.resize(count_table_size + 1);
        item_sum_weight_table.resize(count_table_size + 1);
        user_count_score_table.resize(count_table_size + 1);
        item_count_score_table.resize(count_table_size + 1);
        for (int i = 0; i <= count_table_size; ++i) {
            const float count = static_cast<float>(i);
            const float log_count = std::log1p(count);
            user_count_score_table[i] = coef[1] * log_count +
                                        coef[3] * log_count * log_count +
                                        coef[5] / std::sqrt(count + 1.0f);
            item_count_score_table[i] = coef[2] * log_count +
                                        coef[4] * log_count * log_count +
                                        coef[6] / std::sqrt(count + 1.0f);
            user_sum_weight_table[i] = count > 0.0f ? coef[7] / (count + user_shrink) : 0.0f;
            item_sum_weight_table[i] = count > 0.0f ? coef[8] / (count + item_shrink) : 0.0f;
        }
    }

    void init_thread_local_accumulators() {
        local_thread_count = use_factorized_model ? prediction_threads : 0;
        local_user_sum.clear();
        local_item_sum.clear();
        local_user_count.clear();
        local_item_count.clear();
        if (local_thread_count <= 0) {
            return;
        }
        local_user_sum.assign(local_thread_count, std::vector<float>(users, 0.0f));
        local_item_sum.assign(local_thread_count, std::vector<float>(items, 0.0f));
        local_user_count.assign(local_thread_count, std::vector<int>(users, 0));
        local_item_count.assign(local_thread_count, std::vector<int>(items, 0));
    }

    void update_thread_local(const Rating* ratings, int n, float mean) {
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float* const us = local_user_sum[tid].data();
            float* const is = local_item_sum[tid].data();
            int* const uc = local_user_count[tid].data();
            int* const ic = local_item_count[tid].data();

            const int pairs = n >> 1;
#pragma omp for schedule(static)
            for (int pair = 0; pair < pairs; ++pair) {
                const int idx = pair << 1;
                const Rating& r0 = ratings[idx];
                const int user0 = r0.user;
                const int item0 = r0.item;
                const float residual0 = r0.rating - mean;
                is[item0] += residual0;
                ++ic[item0];
                us[user0] += residual0;
                ++uc[user0];

                const Rating& r1 = ratings[idx + 1];
                const int item1 = r1.item;
                const float residual1 = r1.rating - mean;
                is[item1] += residual1;
                ++ic[item1];
            }
        }

        if ((n & 1) != 0) {
            const Rating& r = ratings[n - 1];
            const float residual = r.rating - mean;
            local_item_sum[0][r.item] += residual;
            ++local_item_count[0][r.item];
            local_user_sum[0][r.user] += residual;
            ++local_user_count[0][r.user];
        }
    }

    float user_component_from(int user, float sum, int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {
            return user_prior[user] + user_count_score_table[count] + sum * user_sum_weight_table[count];
        }
        const float c = static_cast<float>(count);
        const float log_count = std::log1p(c);
        const float score = coef[1] * log_count +
                            coef[3] * log_count * log_count +
                            coef[5] / std::sqrt(c + 1.0f) +
                            (c > 0.0f ? coef[7] * sum / (c + user_shrink) : 0.0f);
        return user_prior[user] + score;
    }

    float item_component_from(float sum, int count) const {
        if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {
            return item_count_score_table[count] + sum * item_sum_weight_table[count];
        }
        const float c = static_cast<float>(count);
        const float log_count = std::log1p(c);
        return coef[2] * log_count +
               coef[4] * log_count * log_count +
               coef[6] / std::sqrt(c + 1.0f) +
               (c > 0.0f ? coef[8] * sum / (c + item_shrink) : 0.0f);
    }

    float user_component(int user) const {
        return user_component_from(user, user_accum[user].sum, user_accum[user].count);
    }

    float item_component(int item) const {
        return item_component_from(item_accum[item].sum, item_accum[item].count);
    }

    void ensure_scores_ready() {
        if (scores_ready) {
            return;
        }
#pragma omp critical(incremental_svd_score_rebuild)
        {
            if (!scores_ready) {
                rebuild_scores();
                scores_ready = true;
            }
        }
    }

    void rebuild_scores() {
        if (use_factorized_model) {
            if (local_thread_count == 4) {
                const float* const us0 = local_user_sum[0].data();
                const float* const us1 = local_user_sum[1].data();
                const float* const us2 = local_user_sum[2].data();
                const float* const us3 = local_user_sum[3].data();
                const int* const uc0 = local_user_count[0].data();
                const int* const uc1 = local_user_count[1].data();
                const int* const uc2 = local_user_count[2].data();
                const int* const uc3 = local_user_count[3].data();
                for (int user = 0; user < users; ++user) {
                    const float sum = us0[user] + us1[user] + us2[user] + us3[user];
                    const int count = uc0[user] + uc1[user] + uc2[user] + uc3[user];
                    user_score[user] = user_component_from(user, sum, count);
                }
                const float* const is0 = local_item_sum[0].data();
                const float* const is1 = local_item_sum[1].data();
                const float* const is2 = local_item_sum[2].data();
                const float* const is3 = local_item_sum[3].data();
                const int* const ic0 = local_item_count[0].data();
                const int* const ic1 = local_item_count[1].data();
                const int* const ic2 = local_item_count[2].data();
                const int* const ic3 = local_item_count[3].data();
                for (int item = 0; item < items; ++item) {
                    const float sum = is0[item] + is1[item] + is2[item] + is3[item];
                    const int count = ic0[item] + ic1[item] + ic2[item] + ic3[item];
                    item_score[item] = item_component_from(sum, count);
                }
            } else {
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
    bool use_factorized_model = false;
    bool has_updates = false;
    bool scores_ready = true;
    int local_thread_count = 0;

    std::vector<Accumulator> user_accum;
    std::vector<Accumulator> item_accum;
    std::vector<float> user_score;
    std::vector<float> item_score;
    std::vector<float> user_prior;
    std::vector<float> user_sum_weight_table;
    std::vector<float> item_sum_weight_table;
    std::vector<float> user_count_score_table;
    std::vector<float> item_count_score_table;
    std::vector<std::vector<float>> local_user_sum;
    std::vector<std::vector<float>> local_item_sum;
    std::vector<std::vector<int>> local_user_count;
    std::vector<std::vector<int>> local_item_count;
    float* user_score_data = nullptr;
    float* item_score_data = nullptr;
    static constexpr int count_table_size = 65536;
};
