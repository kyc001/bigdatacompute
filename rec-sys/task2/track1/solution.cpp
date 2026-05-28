#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float* user_matrix, float* item_matrix,
                         int u_size, int i_size, int dim, float mean) {
        users = u_size;
        items = i_size;
        latent_dim = dim;
        global_mean = mean;
        P = user_matrix;
        Q = item_matrix;

        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_bias.assign(users, 0.0f);
        item_bias.assign(items, 0.0f);
        prepared_threads = 0;
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || P == nullptr || Q == nullptr) {
            return;
        }

        const int n = static_cast<int>(incremental_batch.size());
        int thread_count = 1;
#ifdef _OPENMP
        thread_count = std::min(16, std::max(1, omp_get_max_threads()));
#endif

        prepare_thread_buffers(thread_count);

#pragma omp parallel num_threads(thread_count)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::vector<float>& us = local_user_sum[tid];
            std::vector<float>& is = local_item_sum[tid];
            std::vector<int>& uc = local_user_count[tid];
            std::vector<int>& ic = local_item_count[tid];
            std::vector<int>& tu = local_touched_users[tid];
            std::vector<int>& ti = local_touched_items[tid];

#pragma omp for schedule(static)
            for (int idx = 0; idx < n; ++idx) {
                const Rating& r = incremental_batch[idx];
                if (r.user < 0 || r.user >= users || r.item < 0 || r.item >= items) {
                    continue;
                }

                const float residual = r.rating - global_mean;

                if (uc[r.user] == 0) {
                    tu.push_back(r.user);
                }
                if (ic[r.item] == 0) {
                    ti.push_back(r.item);
                }
                us[r.user] += residual;
                is[r.item] += residual;
                ++uc[r.user];
                ++ic[r.item];
            }
        }

        for (int t = 0; t < thread_count; ++t) {
            for (int u : local_touched_users[t]) {
                const int c = local_user_count[t][u];
                user_sum[u] += local_user_sum[t][u];
                user_count[u] += c;
                user_bias[u] = user_weight * user_sum[u] /
                               (static_cast<float>(user_count[u]) + user_shrink);
                local_user_sum[t][u] = 0.0f;
                local_user_count[t][u] = 0;
            }
            local_touched_users[t].clear();

            for (int i : local_touched_items[t]) {
                const int c = local_item_count[t][i];
                item_sum[i] += local_item_sum[t][i];
                item_count[i] += c;
                item_bias[i] = item_weight * item_sum[i] /
                               (static_cast<float>(item_count[i]) + item_shrink);
                local_item_sum[t][i] = 0.0f;
                local_item_count[t][i] = 0;
            }
            local_touched_items[t].clear();
        }
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items ||
            P == nullptr || Q == nullptr) {
            return global_mean;
        }

        float score = global_mean + user_bias[user_id] + item_bias[item_id];
        return std::min(5.0f, std::max(0.5f, score));
    }

private:
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float user_weight = 0.75f;
    static constexpr float item_weight = 1.0f;

    float dot(const float* a, const float* b) const {
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        int k = 0;
        const int limit = latent_dim - (latent_dim % 4);
        for (; k < limit; k += 4) {
            s0 += a[k] * b[k];
            s1 += a[k + 1] * b[k + 1];
            s2 += a[k + 2] * b[k + 2];
            s3 += a[k + 3] * b[k + 3];
        }
        float total = (s0 + s1) + (s2 + s3);
        for (; k < latent_dim; ++k) {
            total += a[k] * b[k];
        }
        return total;
    }

    void prepare_thread_buffers(int thread_count) {
        if (prepared_threads != thread_count) {
            local_user_sum.assign(thread_count, std::vector<float>(users, 0.0f));
            local_item_sum.assign(thread_count, std::vector<float>(items, 0.0f));
            local_user_count.assign(thread_count, std::vector<int>(users, 0));
            local_item_count.assign(thread_count, std::vector<int>(items, 0));
            local_touched_users.assign(thread_count, std::vector<int>());
            local_touched_items.assign(thread_count, std::vector<int>());
            for (int t = 0; t < thread_count; ++t) {
                local_touched_users[t].reserve(std::min(users, 8192));
                local_touched_items[t].reserve(std::min(items, 8192));
            }
            prepared_threads = thread_count;
            return;
        }

        for (int t = 0; t < prepared_threads; ++t) {
            local_touched_users[t].clear();
            local_touched_items[t].clear();
        }
    }

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_bias;
    std::vector<float> item_bias;
    int prepared_threads = 0;
    std::vector<std::vector<float>> local_user_sum;
    std::vector<std::vector<float>> local_item_sum;
    std::vector<std::vector<int>> local_user_count;
    std::vector<std::vector<int>> local_item_count;
    std::vector<std::vector<int>> local_touched_users;
    std::vector<std::vector<int>> local_touched_items;
};
