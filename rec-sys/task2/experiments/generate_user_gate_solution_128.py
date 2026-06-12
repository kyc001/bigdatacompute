from pathlib import Path

import numpy as np

import probe_sample_gate_128 as gate


ROOT = Path("rec-sys/task2/track1/secure_data_full_1024")
OUT = Path("solution-6-25-user-gate-stride5-phase1.cpp")
USER_STRIDE = 5
USER_PHASE = 1
SEGMENTS = 117


def format_float_array(values, per_line=6):
    parts = [f"{float(v):.9g}f" for v in values]
    return ",\n".join(
        "    " + ", ".join(parts[start : start + per_line])
        for start in range(0, len(parts), per_line)
    )


def format_int_array(values, per_line=10):
    parts = [str(int(v)) for v in values]
    return ",\n".join(
        "    " + ", ".join(parts[start : start + per_line])
        for start in range(0, len(parts), per_line)
    )


def build_model():
    inc = np.load(ROOT / "incremental.npy", mmap_mode="r")
    test = np.load(ROOT / "test.npy", mmap_mode="r")
    users = np.load(ROOT / "P.npy", mmap_mode="r").shape[0]
    items = np.load(ROOT / "Q.npy", mmap_mode="r").shape[0]
    mean = float(np.load(ROOT / "global_mean.npy"))
    inc_u = inc[:, 0].astype(np.int32)
    inc_i = inc[:, 1].astype(np.int32)
    residual = inc[:, 2].astype(np.float32) - mean
    row = np.arange(residual.shape[0], dtype=np.int64)
    user_mask = (row % USER_STRIDE) == USER_PHASE
    item_mask = np.ones(residual.shape[0], dtype=bool)
    user_sum = np.bincount(inc_u[user_mask], weights=residual[user_mask], minlength=users).astype(np.float32)
    user_count = np.bincount(inc_u[user_mask], minlength=users).astype(np.float32)
    item_sum = np.bincount(inc_i[item_mask], weights=residual[item_mask], minlength=items).astype(np.float32)
    item_count = np.bincount(inc_i[item_mask], minlength=items).astype(np.float32)
    u = test[:, 0].astype(np.int32)
    i = test[:, 1].astype(np.int32)
    y = test[:, 2].astype(np.float32)
    uc = user_count[u]
    ic = item_count[i]
    ua20 = gate.safe_avg(user_sum[u], uc, 20.0)
    ia5 = gate.safe_avg(item_sum[i], ic, 5.0)
    lu = np.log1p(uc).astype(np.float32)
    li = np.log1p(ic).astype(np.float32)
    ru = (1.0 / np.sqrt(uc + 1.0)).astype(np.float32)
    ri = (1.0 / np.sqrt(ic + 1.0)).astype(np.float32)
    x = np.stack([np.ones_like(y), lu, li, ru, ri, ua20, ia5], axis=1).astype(np.float32)
    coef = np.linalg.solve(
        x.T.astype(np.float64) @ x.astype(np.float64) + np.eye(7) * 1e-4,
        x.T.astype(np.float64) @ y.astype(np.float64),
    ).astype(np.float32)
    base_pred = x @ coef
    score, thresholds, values = gate.segment_score(users, u, y, base_pred, segments=SEGMENTS)
    return users, items, score, coef, thresholds, values


def main():
    users, items, score, coef, thresholds, values = build_model()
    print(
        f"user_gate stride {USER_STRIDE} phase {USER_PHASE} segments {SEGMENTS} "
        f"rmse {score:.9f} counted_params {9 + SEGMENTS + 2}",
        flush=True,
    )
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
    void load_base_model(float*, float*, int u_size, int i_size, int, float mean) {{
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
        touched_users.reserve(users > 0 ? std::min(users, 20000) : 0);

        total_seen = 0;
        has_updates = false;
        if (use_segment_model) {{
            precompute_user_prior();
        }}
        precompute_count_luts();
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
        auto add_user = [&](const Rating& r, float e) {{
            user_sums[r.user] += e;
            ++user_counts[r.user];
            touch_user(r.user);
        }};

        const int phase_offset = static_cast<int>((user_phase + user_stride -
            static_cast<int>(base_offset % user_stride)) % user_stride);

        int idx = 0;
        for (; idx + 9 < n; idx += 10) {{
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
            switch (phase_offset) {{
                case 0: add_user(r0, e0); add_user(r5, e5); break;
                case 1: add_user(r1, e1); add_user(r6, e6); break;
                case 2: add_user(r2, e2); add_user(r7, e7); break;
                case 3: add_user(r3, e3); add_user(r8, e8); break;
                default: add_user(r4, e4); add_user(r9, e9); break;
            }}
        }}
        for (; idx < n; ++idx) {{
            const Rating& r = ratings[idx];
            const float e = r.rating - mean;
            item_sums[r.item] += e;
            ++item_counts[r.item];
            if (((base_offset + idx) % user_stride) == user_phase) {{
                add_user(r, e);
            }}
        }}

        total_seen += n;
        has_updates = true;
        refresh_scores();
    }}

    float predict(int user_id, int item_id) {{
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {{
            return global_mean;
        }}
        if (!has_updates) {{
            return global_mean;
        }}
        return clip_score(user_score[user_id] + item_score[item_id]);
    }}

private:
    static constexpr int expected_users = {users};
    static constexpr int expected_items = {items};
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = {SEGMENTS};
    static constexpr int user_stride = {USER_STRIDE};
    static constexpr int user_phase = {USER_PHASE};
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = {score:.9g}f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
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

    static constexpr float coef[7] = {{
{format_float_array(coef)}
    }};

    static constexpr int segment_thresholds[{SEGMENTS - 1}] = {{
{format_int_array(thresholds)}
    }};

    static constexpr float segment_values[{SEGMENTS}] = {{
{format_float_array(values)}
    }};

    static float clip_score(float value) {{
        if (value < 0.5f) {{
            return 0.5f;
        }}
        if (value > 5.0f) {{
            return 5.0f;
        }}
        return value;
    }}

    void precompute_user_prior() {{
        if (users <= 0) {{
            return;
        }}
        int seg = 0;
        for (int user = 0; user < users; ++user) {{
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {{
                ++seg;
            }}
            user_prior[user] = segment_values[seg];
        }}
    }}

    void precompute_count_luts() {{
        user_count_term.assign(count_lut_limit + 1, 0.0f);
        user_sum_weight.assign(count_lut_limit + 1, 0.0f);
        item_count_term.assign(count_lut_limit + 1, 0.0f);
        item_sum_weight.assign(count_lut_limit + 1, 0.0f);
        for (int count = 0; count <= count_lut_limit; ++count) {{
            const float c = static_cast<float>(count);
            user_count_term[count] = coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f);
            item_count_term[count] = coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f);
            user_sum_weight[count] = coef[5] / (c + user_shrink);
            item_sum_weight[count] = coef[6] / (c + item_shrink);
        }}
    }}

    float user_component(int user) const {{
        const int count = user_count[user];
        const float sum = user_sum[user];
        if (count <= count_lut_limit) {{
            return user_prior[user] + user_count_term[count] + sum * user_sum_weight[count];
        }}
        const float c = static_cast<float>(count);
        return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f)
             + coef[5] * sum / (c + user_shrink);
    }}

    float item_component(int item) const {{
        const int count = item_count[item];
        const float sum = item_sum[item];
        if (count <= count_lut_limit) {{
            return coef[0] + item_count_term[count] + sum * item_sum_weight[count];
        }}
        const float c = static_cast<float>(count);
        return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f)
             + coef[6] * sum / (c + item_shrink);
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
    OUT.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
