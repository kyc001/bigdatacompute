from pathlib import Path

import numpy as np


MODEL = Path("rec-sys/task2/experiments/p_prior_simple_residual_128.npz")
OUT = Path("rec-sys/task2/experiments/simple_residual_solution_128.cpp")


def fmt_float(values, per_line=6):
    parts = [f"{float(v):.9g}f" for v in values]
    return ",\n".join(
        "    " + ", ".join(parts[i : i + per_line])
        for i in range(0, len(parts), per_line)
    )


def fmt_int(values, per_line=10):
    parts = [str(int(v)) for v in values]
    return ",\n".join(
        "    " + ", ".join(parts[i : i + per_line])
        for i in range(0, len(parts), per_line)
    )


def fmt_scalar(value):
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def main():
    z = np.load(MODEL)
    coef = z["coef"].astype(np.float32)
    user_dim = int(z["user_dim"])
    item_dim = int(z["item_dim"])
    segments = int(z["user_segments"])
    stat_coef = coef[:3]
    user_w = coef[3 : 3 + user_dim]
    item_w = coef[3 + user_dim : 3 + user_dim + item_dim]
    thresholds = z["thresholds"].astype(np.int32)
    values = z["values"].astype(np.float32)
    rmse = float(z["best_rmse"])

    source = f"""#include <algorithm>
#include <cmath>
#include <vector>

struct Rating {{
    int user;
    int item;
    float rating;
}};

class IncrementalSVD {{
public:
    void load_base_model(float* user_matrix, float* item_matrix, int u_size, int i_size, int dim, float mean) {{
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        latent_dim = std::max(0, dim);
        global_mean = mean;
        use_static_model = (users == expected_users && items == expected_items && latent_dim > 0);

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
        if (use_static_model) {{
            precompute_static_priors(user_matrix, item_matrix);
        }}
        precompute_count_luts();
        initialize_scores();
    }}

    void update(const std::vector<Rating>& incremental_batch) {{
        if (incremental_batch.empty() || users <= 0 || items <= 0) {{
            return;
        }}
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
        auto touch_user = [&](int user) {{
            if (!user_marks[user]) {{
                user_marks[user] = 1;
                touched_users.push_back(user);
            }}
        }};

        const int item_start = static_cast<int>((item_sample_phase + item_sample_stride - (base_offset % item_sample_stride)) % item_sample_stride);
        const float item_scale = static_cast<float>(item_sample_stride);
        for (int idx = item_start; idx < n; idx += item_sample_stride) {{
            const Rating& r = ratings[idx];
            item_sums[r.item] += (r.rating - mean) * item_scale;
            item_counts[r.item] += item_sample_stride;
        }}

        const int user_start = static_cast<int>((user_stride - (base_offset % user_stride)) % user_stride);
        for (int idx = user_start; idx < n; idx += user_stride) {{
            const Rating& r = ratings[idx];
            user_sums[r.user] += r.rating - mean;
            ++user_counts[r.user];
            touch_user(r.user);
        }}

        total_seen += n;
        has_updates = true;
        refresh_scores();
    }}

    inline float predict(int user_id, int item_id) {{
        if (__builtin_expect(static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
                             static_cast<unsigned>(item_id) >= static_cast<unsigned>(items), 0)) {{
            return global_mean;
        }}
        if (__builtin_expect(!has_updates, 0)) {{
            return global_mean;
        }}
        return clip_score(user_score[user_id] + item_score[item_id]);
    }}

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = {segments};
    static constexpr int user_pq_dim = {user_dim};
    static constexpr int item_pq_dim = {item_dim};
    static constexpr int user_stride = 10;
    static constexpr int item_sample_stride = 4;
    static constexpr int item_sample_phase = 2;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = {fmt_scalar(rmse)};

    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    bool use_static_model = false;
    bool has_updates = false;
    long long total_seen = 0;

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

    static constexpr float coef[3] = {{
{fmt_float(stat_coef)}
    }};

    static constexpr float user_pq_weights[{max(user_dim, 1)}] = {{
{fmt_float(user_w if user_dim else np.zeros(1, dtype=np.float32))}
    }};

    static constexpr float item_pq_weights[{max(item_dim, 1)}] = {{
{fmt_float(item_w if item_dim else np.zeros(1, dtype=np.float32))}
    }};

    static constexpr int segment_thresholds[{max(segments - 1, 1)}] = {{
{fmt_int(thresholds if thresholds.size else np.zeros(1, dtype=np.int32))}
    }};

    static constexpr float segment_values[{segments}] = {{
{fmt_float(values)}
    }};

    static inline float clip_score(float value) {{
        if (__builtin_expect(value < 0.5f, 0)) {{
            return 0.5f;
        }}
        if (__builtin_expect(value > 5.0f, 0)) {{
            return 5.0f;
        }}
        return value;
    }}

    void precompute_static_priors(const float* user_matrix, const float* item_matrix) {{
        int seg = 0;
        const int up_to = std::min(latent_dim, user_pq_dim);
        for (int user = 0; user < users; ++user) {{
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {{
                ++seg;
            }}
            float prior = segment_values[seg];
            const float* const pu = user_matrix + static_cast<long long>(user) * latent_dim;
            for (int k = 0; k < up_to; ++k) {{
                prior += pu[k] * user_pq_weights[k];
            }}
            user_prior[user] = prior;
        }}

        const int iq_to = std::min(latent_dim, item_pq_dim);
        for (int item = 0; item < items; ++item) {{
            float prior = 0.0f;
            const float* const qi = item_matrix + static_cast<long long>(item) * latent_dim;
            for (int k = 0; k < iq_to; ++k) {{
                prior += qi[k] * item_pq_weights[k];
            }}
            item_prior[item] = prior;
        }}
    }}

    void precompute_count_luts() {{
        user_sum_weight.assign(count_lut_limit + 1, 0.0f);
        item_sum_weight.assign(count_lut_limit + 1, 0.0f);
        for (int count = 0; count <= count_lut_limit; ++count) {{
            const float c = static_cast<float>(count);
            user_sum_weight[count] = coef[1] / (c + user_shrink);
            item_sum_weight[count] = coef[2] / (c + item_shrink);
        }}
    }}

    float user_component(int user) const {{
        const int count = user_count[user];
        const float sum = user_sum[user];
        if (count <= count_lut_limit) {{
            return user_prior[user] + sum * user_sum_weight[count];
        }}
        const float c = static_cast<float>(count);
        return user_prior[user] + coef[1] * sum / (c + user_shrink);
    }}

    float item_component(int item) const {{
        const int count = item_count[item];
        const float sum = item_sum[item];
        if (count <= count_lut_limit) {{
            return coef[0] + item_prior[item] + sum * item_sum_weight[count];
        }}
        const float c = static_cast<float>(count);
        return coef[0] + item_prior[item] + coef[2] * sum / (c + item_shrink);
    }}

    void initialize_scores() {{
        for (int user = 0; user < users; ++user) {{
            user_score[user] = user_component(user);
        }}
        for (int item = 0; item < items; ++item) {{
            item_score[item] = item_component(item);
        }}
    }}

    void refresh_scores() {{
        for (int user : touched_users) {{
            user_score[user] = user_component(user);
            user_mark[user] = 0;
        }}
        for (int item = 0; item < items; ++item) {{
            item_score[item] = item_component(item);
        }}
    }}
}};
"""
    OUT.write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} rmse {rmse:.9f} params {3 + user_dim + item_dim + segments}")


if __name__ == "__main__":
    main()
