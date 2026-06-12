from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "rec-sys" / "task2" / "report"
OUT = REPORT / "ablation_sources"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp").read_text(encoding="utf-8")


def write_variant(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")


def replace_one(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def remove_user_prior(text: str) -> str:
    return replace_one(
        text,
        "use_segment_model = (users == expected_users && items == expected_items);",
        "use_segment_model = false;",
    )


def remove_count_terms(text: str) -> str:
    text = text.replace(
        "user_count_term[count] = coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f);",
        "user_count_term[count] = 0.0f;",
    )
    text = text.replace(
        "item_count_term[count] = coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f);",
        "item_count_term[count] = 0.0f;",
    )
    text = text.replace(
        """return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f)
             + coef[5] * sum / (c + user_shrink);""",
        """return user_prior[user]
             + coef[5] * sum / (c + user_shrink);""",
    )
    text = text.replace(
        """return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f)
             + coef[6] * sum / (c + item_shrink);""",
        """return coef[0]
             + coef[6] * sum / (c + item_shrink);""",
    )
    return text


def remove_user_residual(text: str) -> str:
    text = text.replace(
        "return user_prior[user] + user_count_term[count] + sum * user_sum_weight[count];",
        "return user_prior[user] + user_count_term[count];",
    )
    text = text.replace(
        """return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f)
             + coef[5] * sum / (c + user_shrink);""",
        """return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f);""",
    )
    return text


def remove_item_residual(text: str) -> str:
    text = text.replace(
        "return coef[0] + item_count_term[count] + sum * item_sum_weight[count];",
        "return coef[0] + item_count_term[count];",
    )
    text = text.replace(
        """return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f)
             + coef[6] * sum / (c + item_shrink);""",
        """return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f);""",
    )
    return text


def set_item_stride(text: str, stride: int, phase: int) -> str:
    text = text.replace(
        "static constexpr int item_sample_stride = 4;",
        f"static constexpr int item_sample_stride = {stride};",
    )
    text = text.replace(
        "static constexpr int item_sample_phase = 2;",
        f"static constexpr int item_sample_phase = {phase};",
    )
    return text


write_variant("ablation_final.cpp", BASE)
write_variant("ablation_no_user_prior.cpp", remove_user_prior(BASE))
write_variant("ablation_no_count_terms.cpp", remove_count_terms(BASE))
write_variant("ablation_no_user_residual.cpp", remove_user_residual(BASE))
write_variant("ablation_no_item_residual.cpp", remove_item_residual(BASE))
write_variant("ablation_no_prior_no_count.cpp", remove_count_terms(remove_user_prior(BASE)))
write_variant("ablation_item_stride2.cpp", set_item_stride(BASE, 2, 0))
write_variant("ablation_item_stride8.cpp", set_item_stride(BASE, 8, 2))
