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
        users = u_size;
        items = i_size;
        latent_dim = dim;
        global_mean = mean;
        P = user_matrix;
        Q = item_matrix;
        seen.clear();
    }

    void update(const std::vector<Rating>& incremental_batch) {
        seen.insert(seen.end(), incremental_batch.begin(), incremental_batch.end());
        if (seen.empty()) {
            return;
        }

        // Sparse full-batch reference: seen is the non-zero COO rating list.
        // It performs deterministic SGD over all observed incremental ratings.
        const float lr = 0.008f;
        const float reg = 0.04f;
        const int epochs = 2;
        std::vector<float> pu(latent_dim);
        std::vector<float> qi(latent_dim);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const Rating& r : seen) {
                if (r.user < 0 || r.user >= users || r.item < 0 || r.item >= items) {
                    continue;
                }
                float* p_row = P + static_cast<long long>(r.user) * latent_dim;
                float* q_row = Q + static_cast<long long>(r.item) * latent_dim;
                float pred = global_mean;
                for (int k = 0; k < latent_dim; ++k) {
                    pu[k] = p_row[k];
                    qi[k] = q_row[k];
                    pred += pu[k] * qi[k];
                }
                const float err = r.rating - pred;
                for (int k = 0; k < latent_dim; ++k) {
                    p_row[k] += lr * (err * qi[k] - reg * pu[k]);
                    q_row[k] += lr * (err * pu[k] - reg * qi[k]);
                }
            }
        }
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items) {
            return global_mean;
        }
        float score = global_mean;
        const float* p_row = P + static_cast<long long>(user_id) * latent_dim;
        const float* q_row = Q + static_cast<long long>(item_id) * latent_dim;
        for (int k = 0; k < latent_dim; ++k) {
            score += p_row[k] * q_row[k];
        }
        return std::min(5.0f, std::max(0.5f, score));
    }

private:
    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;
    std::vector<Rating> seen;
};
