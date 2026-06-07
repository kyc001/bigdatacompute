import re
from pathlib import Path

import numpy as np


WEIGHTS = Path("rec-sys/task2/experiments/generalized_mlp_all_k256_h32.npz")
OUT = Path("rec-sys/task2/track1/solution.cpp")


def format_array(name, values, suffix):
    vals = [float(v) for v in np.ravel(values)]
    lines = [f"    static constexpr float {name}{suffix} = {{"]
    for start in range(0, len(vals), 8):
        chunk = vals[start : start + 8]
        lines.append("        " + ", ".join(f"{v:.9g}f" for v in chunk) + ",")
    lines.append("    };")
    return "\n".join(lines)


def transform_linear(w, mean, std):
    inv = 1.0 / std
    coef = (w * inv).astype(np.float32)
    bias = -float(np.dot(mean, coef))
    return coef, bias


def transform_layer(weight, bias, mean, std):
    inv = 1.0 / std
    coef = (weight * inv.reshape(1, -1)).astype(np.float32)
    adjusted = bias.astype(np.float32) - mean.astype(np.float32).dot(coef.T)
    return coef, adjusted.astype(np.float32)


def main():
    text = OUT.read_text(encoding="utf-8")
    w = np.load(WEIGHTS)
    k = int(w["K"])
    hidden = int(w["hidden"])

    p_mu = w["p_mu"].astype(np.float32)
    p_std = w["p_std"].astype(np.float32)
    q_mu = w["q_mu"].astype(np.float32)
    q_std = w["q_std"].astype(np.float32)

    user_linear, user_linear_bias = transform_linear(w["user_w"].astype(np.float32), p_mu, p_std)
    item_linear, item_linear_bias = transform_linear(w["item_w"].astype(np.float32), q_mu, q_std)
    user_hidden, user_hidden_bias = transform_layer(
        w["user_fc1.weight"].astype(np.float32), w["user_fc1.bias"].astype(np.float32), p_mu, p_std
    )
    item_hidden, item_hidden_bias = transform_layer(
        w["item_fc1.weight"].astype(np.float32), w["item_fc1.bias"].astype(np.float32), q_mu, q_std
    )
    user_out = w["user_fc2.weight"].reshape(-1).astype(np.float32)
    item_out = w["item_fc2.weight"].reshape(-1).astype(np.float32)
    user_output_bias = user_linear_bias + float(w["user_fc2.bias"].reshape(-1)[0])
    item_output_bias = item_linear_bias + float(w["item_fc2.bias"].reshape(-1)[0])
    residual_bias = float(w["bias"])

    arrays = f"""    static constexpr int prediction_threads = TASK2_PREDICTION_THREADS;
    static constexpr int embedding_dim = {k};
    static constexpr int hidden_dim = {hidden};

    static constexpr float intercept = {4.430029912552816 + residual_bias:.9g}f;
    static constexpr float log_user_count_weight = -0.35739037763299225f;
    static constexpr float log_item_count_weight = -0.010405138231289704f;
    static constexpr float log_user_count_square_weight = 0.03385049751798333f;
    static constexpr float log_item_count_square_weight = 0.001317321440029219f;
    static constexpr float inv_user_count_weight = -0.8801704691948185f;
    static constexpr float inv_item_count_weight = 0.0439380115327232f;

    static constexpr float user_output_bias = {user_output_bias:.9g}f;
    static constexpr float item_output_bias = {item_output_bias:.9g}f;

{format_array("user_embedding_weight", user_linear, "[embedding_dim]")}

{format_array("item_embedding_weight", item_linear, "[embedding_dim]")}

{format_array("user_hidden_bias", user_hidden_bias, "[hidden_dim]")}

{format_array("item_hidden_bias", item_hidden_bias, "[hidden_dim]")}

{format_array("user_hidden_out", user_out, "[hidden_dim]")}

{format_array("item_hidden_out", item_out, "[hidden_dim]")}

{format_array("user_hidden_weight", user_hidden, "[hidden_dim][embedding_dim]")}

{format_array("item_hidden_weight", item_hidden, "[hidden_dim][embedding_dim]")}

"""

    text = re.sub(
        r"    static constexpr int prediction_threads = TASK2_PREDICTION_THREADS;\n"
        r"    static constexpr int embedding_dim = 1024;\n\n"
        r"    static constexpr float intercept = .*?\n\n"
        r"(?=    static float clip_score)",
        arrays,
        text,
        flags=re.S,
    )

    mlp_build = """    void build_embedding_offsets(const float* user_matrix, const float* item_matrix) {
        const int d = std::min(latent_dim, embedding_dim);
        for (int user = 0; user < users; ++user) {
            const float* const row = user_matrix + 1LL * user * latent_dim;
            float score = user_output_bias;
            for (int k = 0; k < d; ++k) {
                score += row[k] * user_embedding_weight[k];
            }
            for (int h = 0; h < hidden_dim; ++h) {
                const float* const weights = user_hidden_weight[h];
                float activation = user_hidden_bias[h];
                for (int k = 0; k < d; ++k) {
                    activation += row[k] * weights[k];
                }
                if (activation > 0.0f) {
                    score += activation * user_hidden_out[h];
                }
            }
            user_static[user] = score;
        }
        for (int item = 0; item < items; ++item) {
            const float* const row = item_matrix + 1LL * item * latent_dim;
            float score = item_output_bias;
            for (int k = 0; k < d; ++k) {
                score += row[k] * item_embedding_weight[k];
            }
            for (int h = 0; h < hidden_dim; ++h) {
                const float* const weights = item_hidden_weight[h];
                float activation = item_hidden_bias[h];
                for (int k = 0; k < d; ++k) {
                    activation += row[k] * weights[k];
                }
                if (activation > 0.0f) {
                    score += activation * item_hidden_out[h];
                }
            }
            item_static[item] = score;
        }
    }
"""

    text = re.sub(
        r"    void build_embedding_offsets\(const float\* user_matrix, const float\* item_matrix\) \{.*?\n    \}\n\n"
        r"    void build_count_tables",
        mlp_build + "\n    void build_count_tables",
        text,
        flags=re.S,
    )

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} with additive MLP head K={k} hidden={hidden}")


if __name__ == "__main__":
    main()
