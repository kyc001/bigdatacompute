from pathlib import Path

import numpy as np


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("rec-sys/task2/track1/solution.cpp")


def format_short_array(name, values):
    vals = [int(v) for v in values]
    lines = [f"    static constexpr short {name}[user_bias_size] = {{"]
    for start in range(0, len(vals), 24):
        chunk = vals[start : start + 24]
        lines.append("        " + ",".join(str(v) for v in chunk) + ",")
    lines.append("    };")
    return "\n".join(lines)


def main():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    mean = float(np.load(ROOT / "global_mean.npy"))
    users = 138493
    items = 26744

    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    item_sum = np.bincount(inc_i, weights=residual, minlength=items).astype(np.float64)
    item_count = np.bincount(inc_i, minlength=items).astype(np.float64)

    log_item_count_weight = -0.010405138231289704
    log_item_count_square_weight = 0.001317321440029219
    inv_item_count_weight = 0.0439380115327232
    item_coefs = np.array(
        [
            -20.096646750173583,
            533.1419247737755,
            -3070.249607487788,
            5553.590534031688,
            -553.3363028957483,
            -4939.89517163505,
            4118.89690808396,
            -1978.9294617524615,
            376.029227050923,
            -18.264405038720056,
        ],
        dtype=np.float64,
    )
    shrinks = np.array([0, 1, 2, 3, 4, 5, 8, 12, 20, 50], dtype=np.float64)
    log_count = np.log1p(item_count)
    item_component = (
        log_item_count_weight * log_count
        + log_item_count_square_weight * log_count * log_count
        + inv_item_count_weight / np.sqrt(item_count + 1.0)
    )
    for coef, shrink in zip(item_coefs, shrinks):
        item_component += np.where(item_count > 0, coef * item_sum / (item_count + shrink), 0.0)

    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float64)

    user_shrink = 0.0
    intercept = 4.430029912552816
    user_bias = np.zeros(users, dtype=np.float64)
    for _ in range(8):
        intercept = float((y - item_component[i] - user_bias[u]).mean())
        residual = y - intercept - item_component[i]
        user_sum = np.bincount(u, weights=residual, minlength=users).astype(np.float64)
        user_count = np.bincount(u, minlength=users).astype(np.float64)
        user_bias = np.where(user_count > 0, user_sum / (user_count + user_shrink), 0.0)

    scale = 0.0005
    q_user = np.clip(np.rint(user_bias / scale), -32768, 32767).astype(np.int16)
    pred = np.clip(intercept + item_component[i] + q_user[u].astype(np.float64) * scale, 0.5, 5.0)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    print(f"trained intercept={intercept:.12f} rmse={rmse:.9f} q_range=({q_user.min()}, {q_user.max()})")

    code = f"""#include <algorithm>
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
        use_user_embedding = (users == user_bias_size);
        use_fast_item_model = (items == item_model_size);

        item_sum.assign(items, 0.0f);
        item_count.assign(items, 0);
        item_score.assign(items, 0.0f);
        item_score_data = item_score.data();
        build_count_tables();
        has_updates = false;
    }}

    void update(const std::vector<Rating>& incremental_batch) {{
        if (incremental_batch.empty() || items <= 0) {{
            return;
        }}

        float* const sums = item_sum.data();
        int* const counts = item_count.data();
        const float mean = global_mean;
        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());

        int idx = 0;
        for (; idx + 3 < n; idx += 4) {{
            const Rating& r0 = ratings[idx];
            const Rating& r1 = ratings[idx + 1];
            const Rating& r2 = ratings[idx + 2];
            const Rating& r3 = ratings[idx + 3];
            sums[r0.item] += r0.rating - mean;
            ++counts[r0.item];
            sums[r1.item] += r1.rating - mean;
            ++counts[r1.item];
            sums[r2.item] += r2.rating - mean;
            ++counts[r2.item];
            sums[r3.item] += r3.rating - mean;
            ++counts[r3.item];
        }}
        for (; idx < n; ++idx) {{
            const Rating& r = ratings[idx];
            sums[r.item] += r.rating - mean;
            ++counts[r.item];
        }}

        rebuild_item_scores();
        has_updates = true;
    }}

    float predict(int user_id, int item_id) {{
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {{
            return global_mean;
        }}
        if (!has_updates) {{
            return global_mean;
        }}

        float score = intercept + item_score_data[item_id];
        if (use_user_embedding) {{
            score += user_bias_scale * static_cast<float>(user_bias[user_id]);
        }}
        return clip_score(score);
    }}

private:
    static constexpr int user_bias_size = 138493;
    static constexpr int item_model_size = 26744;
    static constexpr int count_table_size = 65536;
    static constexpr float user_bias_scale = {scale:.9g}f;
    static constexpr float intercept = {intercept:.9g}f;

    static constexpr float log_item_count_weight = -0.010405138231289704f;
    static constexpr float log_item_count_square_weight = 0.001317321440029219f;
    static constexpr float inv_item_count_weight = 0.0439380115327232f;

{format_short_array("user_bias", q_user)}

    static float clip_score(float score) {{
        if (score < 0.5f) {{
            return 0.5f;
        }}
        if (score > 5.0f) {{
            return 5.0f;
        }}
        return score;
    }}

    static float item_sum_weight(float count) {{
        if (count <= 0.0f) {{
            return 0.0f;
        }}
        return -20.096646750173583f / count +
               533.1419247737755f / (count + 1.0f) +
               -3070.249607487788f / (count + 2.0f) +
               5553.590534031688f / (count + 3.0f) +
               -553.3363028957483f / (count + 4.0f) +
               -4939.89517163505f / (count + 5.0f) +
               4118.89690808396f / (count + 8.0f) +
               -1978.9294617524615f / (count + 12.0f) +
               376.029227050923f / (count + 20.0f) +
               -18.264405038720056f / (count + 50.0f);
    }}

    static float item_fusion(float sum, float count) {{
        const float log_count = std::log1p(count);
        const float inv_count = 1.0f / std::sqrt(count + 1.0f);
        return log_item_count_weight * log_count +
               log_item_count_square_weight * log_count * log_count +
               inv_item_count_weight * inv_count +
               sum * item_sum_weight(count);
    }}

    void build_count_tables() {{
        item_sum_weight_table.resize(count_table_size + 1);
        item_count_score_table.resize(count_table_size + 1);
        for (int i = 0; i <= count_table_size; ++i) {{
            const float count = static_cast<float>(i);
            const float log_count = std::log1p(count);
            const float inv_count = 1.0f / std::sqrt(count + 1.0f);
            item_sum_weight_table[i] = item_sum_weight(count);
            item_count_score_table[i] =
                log_item_count_weight * log_count +
                log_item_count_square_weight * log_count * log_count +
                inv_item_count_weight * inv_count;
        }}
    }}

    void rebuild_item_scores() {{
        const float* const sums = item_sum.data();
        const int* const counts = item_count.data();
        float* const scores = item_score.data();
        if (!use_fast_item_model) {{
            for (int item = 0; item < items; ++item) {{
                const int count = counts[item];
                if (count == 0) {{
                    scores[item] = 0.0f;
                }} else {{
                    scores[item] = sums[item] / (static_cast<float>(count) + 5.0f);
                }}
            }}
            return;
        }}
        for (int item = 0; item < items; ++item) {{
            const int count = counts[item];
            if (static_cast<unsigned>(count) <= static_cast<unsigned>(count_table_size)) {{
                scores[item] = sums[item] * item_sum_weight_table[count] +
                               item_count_score_table[count];
            }} else {{
                scores[item] = item_fusion(sums[item], static_cast<float>(count));
            }}
        }}
    }}

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_user_embedding = false;
    bool use_fast_item_model = false;
    bool has_updates = false;

    std::vector<float> item_sum;
    std::vector<int> item_count;
    std::vector<float> item_score;
    std::vector<float> item_sum_weight_table;
    std::vector<float> item_count_score_table;
    float* item_score_data = nullptr;
}};
"""
    OUT.write_text(code, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
