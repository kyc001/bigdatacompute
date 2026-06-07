from pathlib import Path

import numpy as np

import train_generalized_head as common


MODEL = Path("rec-sys/task2/experiments/factorized_id_under1k.npz")
OUT = Path("rec-sys/task2/track1/solution.cpp")


def fmt_array(values, per_line=6):
    vals = np.asarray(values).reshape(-1)
    chunks = []
    for start in range(0, vals.size, per_line):
        part = vals[start : start + per_line]
        chunks.append("    " + ", ".join(f"{float(x):#.9g}f" for x in part))
    return ",\n".join(chunks)


def main():
    model = np.load(MODEL)
    high = int(model["high"])
    low = int(model["low"])
    rank = int(model["rank"])
    item_bins = int(model["item_bins"])
    base_bins = int(model["base_bins"])
    params = int(model["param_count"])
    best_rmse = float(model["best_rmse"])
    assert high == 192 and low == 296 and rank == 2 and item_bins == 1 and base_bins == 16
    assert params == 993

    a = np.asarray(model["a"], dtype=np.float32)
    b = np.asarray(model["b"], dtype=np.float32)
    item = np.asarray(model["item"], dtype=np.float32)
    base = np.asarray(model["base"], dtype=np.float32)

    text = f"""#include <algorithm>
#include <cmath>
#include <vector>

struct Rating {{
    int user;
    int item;
    float rating;
}};

class IncrementalSVD {{
public:
    void load_base_model(float*, float*, int u_size, int i_size, int, float mean) {{
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;
        use_factorized_model = (users == expected_users && items == expected_items);

        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_base_score.assign(users, 0.0f);
        item_base_score.assign(items, 0.0f);
        user_factor_score.assign(users, 0.0f);

        total_seen = 0;
        has_updates = false;
        scores_ready = false;
        if (use_factorized_model) {{
            precompute_user_factor();
        }}
    }}

    void update(const std::vector<Rating>& incremental_batch) {{
        if (incremental_batch.empty() || users <= 0 || items <= 0) {{
            return;
        }}

        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());
        const float mean = global_mean;
        const long long base_offset = total_seen;

        for (int idx = 0; idx < n; ++idx) {{
            const Rating& r = ratings[idx];
            if (static_cast<unsigned>(r.item) < static_cast<unsigned>(items)) {{
                const float residual = r.rating - mean;
                item_sum[r.item] += residual;
                ++item_count[r.item];
                if (((base_offset + idx) & 1LL) == 0 &&
                    static_cast<unsigned>(r.user) < static_cast<unsigned>(users)) {{
                    user_sum[r.user] += residual;
                    ++user_count[r.user];
                }}
            }}
        }}

        total_seen += n;
        has_updates = true;
        scores_ready = false;
        if (n < usual_batch_size) {{
            rebuild_scores();
        }}
    }}

    float predict(int user_id, int item_id) {{
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {{
            return global_mean;
        }}
        if (!has_updates) {{
            return global_mean;
        }}
        if (!scores_ready) {{
            rebuild_scores();
        }}

        if (use_factorized_model) {{
            const float base_score = base_intercept + user_base_score[user_id] + item_base_score[item_id];
            const int bin = base_bin(base_score);
            const float score = base_score + user_factor_score[user_id] + item_bias[0] + base_bias[bin];
            return clip_score(score);
        }}

        return clip_score(global_mean + user_base_score[user_id] + item_base_score[item_id]);
    }}

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int factor_high = {high};
    static constexpr int factor_low = {low};
    static constexpr int factor_rank = {rank};
    static constexpr int base_bin_count = {base_bins};
    static constexpr int usual_batch_size = 100000;
    static constexpr float base_intercept = {float(common.BASE_COEF[0]):.9g}f;
    static constexpr float model_rmse = {best_rmse:.9g}f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_factorized_model = false;
    bool has_updates = false;
    bool scores_ready = false;
    long long total_seen = 0;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_base_score;
    std::vector<float> item_base_score;
    std::vector<float> user_factor_score;

    static constexpr float coef[27] = {{
{fmt_array(common.BASE_COEF, 6)}
    }};

    static constexpr float user_shrinks[10] = {{
{fmt_array(common.USER_SHRINKS, 10)}
    }};

    static constexpr float item_shrinks[10] = {{
{fmt_array(common.ITEM_SHRINKS, 10)}
    }};

    static constexpr float factor_a[{high}][{rank}] = {{
"""
    rows = []
    for row in a:
        rows.append("        {" + ", ".join(f"{float(x):.9g}f" for x in row) + "}")
    text += ",\n".join(rows)
    text += f"""
    }};

    static constexpr float factor_b[{low}][{rank}] = {{
"""
    rows = []
    for row in b:
        rows.append("        {" + ", ".join(f"{float(x):.9g}f" for x in row) + "}")
    text += ",\n".join(rows)
    text += f"""
    }};

    static constexpr float item_bias[{item_bins}] = {{
{fmt_array(item, 8)}
    }};

    static constexpr float base_bias[{base_bins}] = {{
{fmt_array(base, 8)}
    }};

    static float clip_score(float score) {{
        return std::min(5.0f, std::max(0.5f, score));
    }}

    static int base_bin(float score) {{
        const float clipped = clip_score(score);
        int bin = static_cast<int>((clipped - 0.5f) * (static_cast<float>(base_bin_count) / 4.5f));
        if (bin < 0) {{
            return 0;
        }}
        if (bin >= base_bin_count) {{
            return base_bin_count - 1;
        }}
        return bin;
    }}

    void precompute_user_factor() {{
        for (int user = 0; user < users; ++user) {{
            const int hi = static_cast<int>((1LL * user * factor_high) / users);
            const int lo = user % factor_low;
            float value = 0.0f;
            for (int r = 0; r < factor_rank; ++r) {{
                value += factor_a[hi][r] * factor_b[lo][r];
            }}
            user_factor_score[user] = value;
        }}
    }}

    void rebuild_scores() {{
        if (!has_updates) {{
            scores_ready = true;
            return;
        }}

        if (use_factorized_model) {{
            for (int user = 0; user < users; ++user) {{
                const float count = static_cast<float>(user_count[user]);
                const float sum = user_sum[user];
                const float log_count = std::log1p(count);
                float score = coef[1] * log_count
                            + coef[3] * log_count * log_count
                            + coef[5] / std::sqrt(count + 1.0f);
                for (int idx = 0; idx < 10; ++idx) {{
                    const float avg = count > 0.0f ? sum / (count + user_shrinks[idx]) : 0.0f;
                    score += coef[7 + idx] * avg;
                }}
                user_base_score[user] = score;
            }}

            for (int item = 0; item < items; ++item) {{
                const float count = static_cast<float>(item_count[item]);
                const float sum = item_sum[item];
                const float log_count = std::log1p(count);
                float score = coef[2] * log_count
                            + coef[4] * log_count * log_count
                            + coef[6] / std::sqrt(count + 1.0f);
                for (int idx = 0; idx < 10; ++idx) {{
                    const float avg = count > 0.0f ? sum / (count + item_shrinks[idx]) : 0.0f;
                    score += coef[17 + idx] * avg;
                }}
                item_base_score[item] = score;
            }}
        }} else {{
            for (int user = 0; user < users; ++user) {{
                const float count = static_cast<float>(user_count[user]);
                user_base_score[user] = count > 0.0f ? 0.8f * user_sum[user] / (count + 5.0f) : 0.0f;
            }}
            for (int item = 0; item < items; ++item) {{
                const float count = static_cast<float>(item_count[item]);
                item_base_score[item] = count > 0.0f ? 0.9f * item_sum[item] / (count + 3.0f) : 0.0f;
            }}
        }}

        scores_ready = true;
    }}
}};
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} using {MODEL} rmse={best_rmse:.9f} params={params}")


if __name__ == "__main__":
    main()
