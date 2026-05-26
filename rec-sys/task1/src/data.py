from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable


Rating = tuple[int, int, float]
RequestGroup = tuple[int, list[int]]


@dataclass(frozen=True)
class DatasetStats:
    users: int
    items: int
    ratings: int
    min_rating: float
    max_rating: float
    mean_rating: float
    sparsity: float
    test_users: int
    test_pairs: int


def _split_header(line: str, path: Path, line_number: int) -> tuple[int, int]:
    try:
        user_raw, count_raw = line.split("|", 1)
        return int(user_raw.strip()), int(count_raw.strip())
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid user header {line!r}") from exc


def read_train(path: Path) -> list[Rating]:
    ratings: list[Rating] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue

        user_id, expected_count = _split_header(line, path, index)
        for offset in range(expected_count):
            if index >= len(lines):
                raise ValueError(f"{path}: user {user_id} expected {expected_count} ratings but file ended early")
            item_line = lines[index].strip()
            index += 1
            parts = item_line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{index}: invalid rating row {item_line!r}")
            ratings.append((user_id, int(parts[0]), float(parts[1])))

    return ratings


def read_test(path: Path) -> list[RequestGroup]:
    groups: list[RequestGroup] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue

        user_id, expected_count = _split_header(line, path, index)
        items: list[int] = []
        for _ in range(expected_count):
            if index >= len(lines):
                raise ValueError(f"{path}: user {user_id} expected {expected_count} items but file ended early")
            item_line = lines[index].strip()
            index += 1
            parts = item_line.split()
            if not parts:
                raise ValueError(f"{path}:{index}: empty test item row")
            items.append(int(parts[0]))
        groups.append((user_id, items))

    return groups


def write_predictions(path: Path, groups: Iterable[RequestGroup], predictions: dict[tuple[int, int], float], round_scores: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for user_id, items in groups:
        lines.append(f"{user_id}|{len(items)}")
        for item_id in items:
            score = predictions[(user_id, item_id)]
            rendered = str(int(round(score))) if round_scores else f"{score:.6f}".rstrip("0").rstrip(".")
            lines.append(f"{item_id} {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_ratings(ratings: list[Rating], validation_ratio: float, seed: int) -> tuple[list[Rating], list[Rating]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    shuffled = list(ratings)
    Random(seed).shuffle(shuffled)
    valid_count = max(1, int(len(shuffled) * validation_ratio))
    return shuffled[valid_count:], shuffled[:valid_count]


def grouped_rating_counts(ratings: Iterable[Rating]) -> tuple[Counter[int], Counter[int]]:
    user_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    for user_id, item_id, _ in ratings:
        user_counts[user_id] += 1
        item_counts[item_id] += 1
    return user_counts, item_counts


def compute_stats(ratings: list[Rating], test_groups: list[RequestGroup]) -> DatasetStats:
    users = {user_id for user_id, _, _ in ratings}
    items = {item_id for _, item_id, _ in ratings}
    values = [rating for _, _, rating in ratings]
    test_pairs = sum(len(items_) for _, items_ in test_groups)
    matrix_size = len(users) * len(items)
    sparsity = 1.0 - (len(ratings) / matrix_size) if matrix_size else 0.0
    return DatasetStats(
        users=len(users),
        items=len(items),
        ratings=len(ratings),
        min_rating=min(values),
        max_rating=max(values),
        mean_rating=sum(values) / len(values),
        sparsity=sparsity,
        test_users=len(test_groups),
        test_pairs=test_pairs,
    )


def rating_histogram(ratings: Iterable[Rating]) -> dict[int, int]:
    histogram: defaultdict[int, int] = defaultdict(int)
    for _, _, rating in ratings:
        histogram[int(round(rating))] += 1
    return dict(sorted(histogram.items()))
