from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Callable, TypeVar

from data import Rating


T = TypeVar("T")


def rmse(actual: list[Rating], predict: Callable[[int, int], float]) -> float:
    if not actual:
        raise ValueError("cannot compute RMSE on an empty rating set")
    total = 0.0
    for user_id, item_id, rating in actual:
        error = predict(user_id, item_id) - rating
        total += error * error
    return sqrt(total / len(actual))


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start
