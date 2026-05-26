from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from data import (
    RequestGroup,
    Rating,
    compute_stats,
    grouped_rating_counts,
    rating_histogram,
    read_test,
    read_train,
    split_ratings,
    write_predictions,
)
from metrics import rmse, timed
from models import (
    BlendedModel,
    MFConfig,
    MatrixFactorizationModel,
    MeanBaselineModel,
    Recommender,
    ResidualMatrixFactorizationModel,
)


ModelName = Literal["baseline", "mf", "blend", "residual"]


@dataclass(frozen=True)
class Paths:
    task_root: Path
    train_path: Path
    test_path: Path

    @classmethod
    def from_task_root(cls, task_root: Path) -> "Paths":
        data_dir = task_root / "data"
        return cls(
            task_root=task_root,
            train_path=data_dir / "train.txt",
            test_path=data_dir / "test.txt",
        )


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    train_ratings: int
    validation_ratings: int
    rmse: float
    train_seconds: float


def load_inputs(paths: Paths) -> tuple[list[Rating], list[RequestGroup]]:
    return read_train(paths.train_path), read_test(paths.test_path)


def make_model(model_name: ModelName, mf_config: MFConfig, shrinkage: float) -> Recommender:
    if model_name == "baseline":
        return MeanBaselineModel(shrinkage=shrinkage, iterations=mf_config.bias_iterations)
    if model_name == "mf":
        return MatrixFactorizationModel(config=mf_config)
    if model_name == "blend":
        return BlendedModel(
            baseline=MeanBaselineModel(shrinkage=shrinkage, iterations=mf_config.bias_iterations),
            mf=MatrixFactorizationModel(config=mf_config),
            mf_weight=mf_config.blend_weight,
        )
    if model_name == "residual":
        return ResidualMatrixFactorizationModel(
            baseline=MeanBaselineModel(shrinkage=shrinkage, iterations=mf_config.bias_iterations),
            config=mf_config,
        )
    raise ValueError(f"unknown model {model_name!r}")


def describe_dataset(paths: Paths) -> dict[str, object]:
    ratings, test_groups = load_inputs(paths)
    stats = compute_stats(ratings, test_groups)
    return {
        **asdict(stats),
        "rating_histogram": rating_histogram(ratings),
    }


def analyze_dataset(paths: Paths) -> dict[str, object]:
    ratings, test_groups = load_inputs(paths)
    train_users = {user_id for user_id, _, _ in ratings}
    train_items = {item_id for _, item_id, _ in ratings}
    test_users = {user_id for user_id, _ in test_groups}
    test_items = {item_id for _, items in test_groups for item_id in items}
    test_pairs = [(user_id, item_id) for user_id, items in test_groups for item_id in items]
    user_counts, item_counts = grouped_rating_counts(ratings)

    def count_at_most(values: list[int], threshold: int) -> int:
        return sum(1 for value in values if value <= threshold)

    user_count_values = list(user_counts.values())
    item_count_values = list(item_counts.values())
    cold_user_pairs = sum(1 for user_id, _ in test_pairs if user_id not in train_users)
    cold_item_pairs = sum(1 for _, item_id in test_pairs if item_id not in train_items)
    fully_known_pairs = sum(1 for user_id, item_id in test_pairs if user_id in train_users and item_id in train_items)

    return {
        "train_users": len(train_users),
        "train_items": len(train_items),
        "train_ratings": len(ratings),
        "test_users": len(test_users),
        "test_items": len(test_items),
        "test_pairs": len(test_pairs),
        "new_test_users": len(test_users - train_users),
        "new_test_items": len(test_items - train_items),
        "cold_user_pairs": cold_user_pairs,
        "cold_item_pairs": cold_item_pairs,
        "fully_known_pairs": fully_known_pairs,
        "fully_known_pair_ratio": fully_known_pairs / len(test_pairs) if test_pairs else 0.0,
        "user_rating_count_min": min(user_count_values),
        "user_rating_count_max": max(user_count_values),
        "item_rating_count_min": min(item_count_values),
        "item_rating_count_max": max(item_count_values),
        "users_with_at_most_20_ratings": count_at_most(user_count_values, 20),
        "users_with_at_most_50_ratings": count_at_most(user_count_values, 50),
        "users_with_at_most_100_ratings": count_at_most(user_count_values, 100),
        "items_with_one_rating": count_at_most(item_count_values, 1),
        "items_with_at_most_5_ratings": count_at_most(item_count_values, 5),
    }


def evaluate(
    paths: Paths,
    model_name: ModelName,
    validation_ratio: float,
    seed: int,
    mf_config: MFConfig,
    shrinkage: float,
) -> EvaluationResult:
    ratings = read_train(paths.train_path)
    train_ratings, validation_ratings = split_ratings(ratings, validation_ratio=validation_ratio, seed=seed)
    model = make_model(model_name, mf_config=mf_config, shrinkage=shrinkage)
    _, train_seconds = timed(lambda: model.fit(train_ratings))
    validation_rmse = rmse(validation_ratings, model.predict)
    return EvaluationResult(
        model=model_name,
        train_ratings=len(train_ratings),
        validation_ratings=len(validation_ratings),
        rmse=validation_rmse,
        train_seconds=train_seconds,
    )


def predict_test(
    paths: Paths,
    model_name: ModelName,
    output_path: Path,
    mf_config: MFConfig,
    shrinkage: float,
    round_scores: bool,
) -> tuple[int, float, float]:
    ratings, test_groups = load_inputs(paths)
    model = make_model(model_name, mf_config=mf_config, shrinkage=shrinkage)
    _, train_seconds = timed(lambda: model.fit(ratings))

    def build_predictions() -> dict[tuple[int, int], float]:
        predictions: dict[tuple[int, int], float] = {}
        for user_id, items in test_groups:
            for item_id in items:
                predictions[(user_id, item_id)] = model.predict(user_id, item_id)
        return predictions

    predictions, predict_seconds = timed(build_predictions)
    write_predictions(output_path, test_groups, predictions, round_scores=round_scores)
    return len(predictions), train_seconds, predict_seconds
