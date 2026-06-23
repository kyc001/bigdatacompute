import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "rec-sys" / "task2" / "report"
EXP = ROOT / "rec-sys" / "task2" / "experiments"
OUT = REPORT / "ablation_sources"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp").read_text(encoding="utf-8")


def format_float_array(values, per_line=6):
    parts = [f"{float(v):.9g}f" for v in values]
    return ",\n".join("        " + ", ".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line))


def format_int_array(values, per_line=10):
    parts = [str(int(v)) for v in values]
    return ",\n".join("        " + ", ".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line))


def replace_regex(text, pattern, repl):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"pattern not replaced: {pattern[:80]}")
    return new


def replace_literal(text, old, new):
    if old not in text:
        raise RuntimeError(f"literal not found: {old[:80]}")
    return text.replace(old, new, 1)


def with_npz(text, npz_name, stride=None, phase=None, table_size=8192):
    z = np.load(EXP / npz_name)
    if stride is None:
        stride = int(z["stride"]) if "stride" in z.files else 1
    if phase is None:
        phase = int(z["phase"]) if "phase" in z.files else 0
    rmse = float(z["best_rmse"])
    text = replace_regex(text, r"static constexpr int item_stride = \d+;", f"static constexpr int item_stride = {stride};")
    text = replace_regex(text, r"static constexpr int item_phase = \d+;", f"static constexpr int item_phase = {phase};")
    text = replace_regex(text, r"static constexpr float model_rmse = [^;]+;", f"static constexpr float model_rmse = {rmse:.9g}f;")
    text = replace_regex(
        text,
        r"static constexpr float coef\[7\] = \{.*?\n    \};",
        "static constexpr float coef[7] = {\n" + format_float_array(z["coef"]) + "\n    };",
    )
    text = replace_regex(
        text,
        r"static constexpr int segment_thresholds\[118\] = \{.*?\n    \};",
        "static constexpr int segment_thresholds[118] = {\n" + format_int_array(z["thresholds"]) + "\n    };",
    )
    text = replace_regex(
        text,
        r"static constexpr float segment_values\[119\] = \{.*?\n    \};",
        "static constexpr float segment_values[119] = {\n" + format_float_array(z["values"]) + "\n    };",
    )
    text = replace_regex(text, r"static constexpr int count_table_size = \d+;", f"static constexpr int count_table_size = {table_size};")
    return text


def no_segment_prior(text):
    return replace_literal(
        text,
        "user_prior[user] = coef[0] + segment_values[seg];",
        "user_prior[user] = coef[0];",
    )


def no_count_terms(text):
    text = text.replace(
        "user_count_score_table[i] = coef[1] * log_count + coef[3] / std::sqrt(count + 1.0f);",
        "user_count_score_table[i] = 0.0f;",
    )
    text = text.replace(
        "item_count_score_table[i] = coef[2] * log_count + coef[4] / std::sqrt(count + 1.0f);",
        "item_count_score_table[i] = 0.0f;",
    )
    text = text.replace(
        "return user_prior[user] + coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f) +\n"
        "               (c > 0.0f ? coef[5] * sum / (c + user_shrink) : 0.0f);",
        "return user_prior[user] + (c > 0.0f ? coef[5] * sum / (c + user_shrink) : 0.0f);",
    )
    text = text.replace(
        "return coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f) +\n"
        "               (c > 0.0f ? coef[6] * sum / (c + item_shrink) : 0.0f);",
        "return c > 0.0f ? coef[6] * sum / (c + item_shrink) : 0.0f;",
    )
    return text


def no_user_residual(text):
    text = text.replace(
        "user_sum_weight_table[i] = count > 0.0f ? coef[5] / (count + user_shrink) : 0.0f;",
        "user_sum_weight_table[i] = 0.0f;",
    )
    text = text.replace(
        "return user_prior[user] + coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f) +\n"
        "               (c > 0.0f ? coef[5] * sum / (c + user_shrink) : 0.0f);",
        "return user_prior[user] + coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f);",
    )
    return text


def no_item_residual(text):
    text = text.replace(
        "item_sum_weight_table[i] = count > 0.0f ? coef[6] / (count + item_shrink) : 0.0f;",
        "item_sum_weight_table[i] = 0.0f;",
    )
    text = text.replace(
        "return coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f) +\n"
        "               (c > 0.0f ? coef[6] * sum / (c + item_shrink) : 0.0f);",
        "return coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f);",
    )
    return text


def item_only(text):
    return no_user_residual(no_count_terms(no_segment_prior(text)))


def constant_predictor(text):
    text = replace_literal(
        text,
        "if (!has_updates) {\n            return global_mean;\n        }\n        if (use_segment_model) {\n            return clip_score(user_score_data[user_id] + item_score_data[item_id]);\n        }\n        return clip_score(global_mean + user_score_data[user_id] + item_score_data[item_id]);",
        "return global_mean;",
    )
    text = replace_literal(
        text,
        "update_online_sampled(ratings, n, mean, total_seen);",
        "(void)ratings; (void)n; (void)mean;",
    )
    return text


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")


def main():
    write("final.cpp", BASE)
    write("dense_item.cpp", with_npz(BASE, "segment_base7_119.npz", stride=1, phase=0, table_size=65536))
    write("stride2.cpp", with_npz(BASE, "stride_scaled_128.npz", table_size=65536))
    write("stride4.cpp", with_npz(BASE, "stride4_phase2_128.npz", table_size=65536))
    write("stride16.cpp", with_npz(BASE, "stride16_phase13_128.npz", table_size=65536))
    write("no_segment_prior.cpp", no_segment_prior(BASE))
    write("no_count_terms.cpp", no_count_terms(BASE))
    write("no_user_residual.cpp", no_user_residual(BASE))
    write("no_item_residual.cpp", no_item_residual(BASE))
    write("item_only.cpp", item_only(BASE))
    write("constant.cpp", constant_predictor(BASE))
    shutil.copy2(ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp", OUT / "submission_final.cpp")


if __name__ == "__main__":
    main()
