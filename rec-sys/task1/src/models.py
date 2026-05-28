from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from data import Rating


class Recommender:
    min_rating: float
    max_rating: float

    def fit(self, ratings: list[Rating]) -> None:
        raise NotImplementedError

    def predict(self, user_id: int, item_id: int) -> float:
        raise NotImplementedError

    def _clip(self, score: float) -> float:
        return min(self.max_rating, max(self.min_rating, score))


class MeanBaselineModel(Recommender):
    def __init__(self, shrinkage: float = 10.0, iterations: int = 5) -> None:
        self.shrinkage = shrinkage
        self.iterations = iterations
        self.global_mean = 0.0
        self.user_bias: dict[int, float] = {}
        self.item_bias: dict[int, float] = {}
        self.min_rating = 0.0
        self.max_rating = 100.0

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("cannot train baseline on empty ratings")
        values = np.array([rating for _, _, rating in ratings], dtype=np.float64)
        self.global_mean = float(values.mean())
        self.min_rating = float(values.min())
        self.max_rating = float(values.max())

        self.user_bias = {}
        self.item_bias = {}

        for _ in range(max(1, self.iterations)):
            user_sum: defaultdict[int, float] = defaultdict(float)
            user_count: defaultdict[int, int] = defaultdict(int)
            for user_id, item_id, rating in ratings:
                user_sum[user_id] += rating - self.global_mean - self.item_bias.get(item_id, 0.0)
                user_count[user_id] += 1
            self.user_bias = {
                user_id: user_sum[user_id] / (self.shrinkage + count)
                for user_id, count in user_count.items()
            }

            item_sum: defaultdict[int, float] = defaultdict(float)
            item_count: defaultdict[int, int] = defaultdict(int)
            for user_id, item_id, rating in ratings:
                item_sum[item_id] += rating - self.global_mean - self.user_bias.get(user_id, 0.0)
                item_count[item_id] += 1
            self.item_bias = {
                item_id: item_sum[item_id] / (self.shrinkage + count)
                for item_id, count in item_count.items()
            }

    def predict(self, user_id: int, item_id: int) -> float:
        score = self.global_mean + self.user_bias.get(user_id, 0.0) + self.item_bias.get(item_id, 0.0)
        return self._clip(score)


@dataclass(frozen=True)
class MFConfig:
    factors: int = 24
    epochs: int = 20
    learning_rate: float = 0.008
    regularization: float = 0.05
    blend_weight: float = 1.0
    residual_weight: float = 1.0
    bias_iterations: int = 5
    seed: int = 42
    shuffle: bool = True


class MatrixFactorizationModel(Recommender):
    def __init__(self, config: MFConfig) -> None:
        self.config = config
        self.global_mean = 0.0
        self.min_rating = 0.0
        self.max_rating = 100.0
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("cannot train matrix factorization on empty ratings")
        users = sorted({user_id for user_id, _, _ in ratings})
        items = sorted({item_id for _, item_id, _ in ratings})
        self.user_index = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_index = {item_id: idx for idx, item_id in enumerate(items)}

        values = np.array([rating for _, _, rating in ratings], dtype=np.float64)
        self.global_mean = float(values.mean())
        self.min_rating = float(values.min())
        self.max_rating = float(values.max())

        rng = np.random.default_rng(self.config.seed)
        scale = 0.1
        self.user_bias = np.zeros(len(users), dtype=np.float64)
        self.item_bias = np.zeros(len(items), dtype=np.float64)
        self.user_factors = rng.normal(0.0, scale, size=(len(users), self.config.factors))
        self.item_factors = rng.normal(0.0, scale, size=(len(items), self.config.factors))

        encoded = np.array(
            [
                (self.user_index[user_id], self.item_index[item_id], rating)
                for user_id, item_id, rating in ratings
            ],
            dtype=np.float64,
        )

        for _ in range(self.config.epochs):
            if self.config.shuffle:
                rng.shuffle(encoded)
            for user_pos_raw, item_pos_raw, rating in encoded:
                user_pos = int(user_pos_raw)
                item_pos = int(item_pos_raw)
                prediction = self._predict_known(user_pos, item_pos)
                error = rating - prediction

                user_vector = self.user_factors[user_pos].copy()
                item_vector = self.item_factors[item_pos].copy()
                lr = self.config.learning_rate
                reg = self.config.regularization

                self.user_bias[user_pos] += lr * (error - reg * self.user_bias[user_pos])
                self.item_bias[item_pos] += lr * (error - reg * self.item_bias[item_pos])
                self.user_factors[user_pos] += lr * (error * item_vector - reg * user_vector)
                self.item_factors[item_pos] += lr * (error * user_vector - reg * item_vector)

    def _predict_known(self, user_pos: int, item_pos: int) -> float:
        assert self.user_bias is not None
        assert self.item_bias is not None
        assert self.user_factors is not None
        assert self.item_factors is not None
        return float(
            self.global_mean
            + self.user_bias[user_pos]
            + self.item_bias[item_pos]
            + np.dot(self.user_factors[user_pos], self.item_factors[item_pos])
        )

    def predict(self, user_id: int, item_id: int) -> float:
        user_pos = self.user_index.get(user_id)
        item_pos = self.item_index.get(item_id)
        if user_pos is None or item_pos is None:
            score = self.global_mean
            if user_pos is not None and self.user_bias is not None:
                score += float(self.user_bias[user_pos])
            if item_pos is not None and self.item_bias is not None:
                score += float(self.item_bias[item_pos])
            return self._clip(score)
        return self._clip(self._predict_known(user_pos, item_pos))


class BlendedModel(Recommender):
    def __init__(self, baseline: MeanBaselineModel, mf: MatrixFactorizationModel, mf_weight: float) -> None:
        if not 0.0 <= mf_weight <= 1.0:
            raise ValueError("mf_weight must be between 0 and 1")
        self.baseline = baseline
        self.mf = mf
        self.mf_weight = mf_weight
        self.min_rating = 0.0
        self.max_rating = 100.0

    def fit(self, ratings: list[Rating]) -> None:
        self.baseline.fit(ratings)
        self.mf.fit(ratings)
        self.min_rating = self.baseline.min_rating
        self.max_rating = self.baseline.max_rating

    def predict(self, user_id: int, item_id: int) -> float:
        baseline_score = self.baseline.predict(user_id, item_id)
        mf_score = self.mf.predict(user_id, item_id)
        score = (1.0 - self.mf_weight) * baseline_score + self.mf_weight * mf_score
        return self._clip(score)


class WeightedEnsembleModel(Recommender):
    def __init__(self, weighted_models: list[tuple[float, Recommender]]) -> None:
        if not weighted_models:
            raise ValueError("ensemble requires at least one model")
        total_weight = sum(weight for weight, _ in weighted_models)
        if total_weight <= 0.0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.weighted_models = [(weight / total_weight, model) for weight, model in weighted_models]
        self.min_rating = 0.0
        self.max_rating = 100.0

    def fit(self, ratings: list[Rating]) -> None:
        for _, model in self.weighted_models:
            model.fit(ratings)
        first_model = self.weighted_models[0][1]
        self.min_rating = first_model.min_rating
        self.max_rating = first_model.max_rating

    def predict(self, user_id: int, item_id: int) -> float:
        score = sum(weight * model.predict(user_id, item_id) for weight, model in self.weighted_models)
        return self._clip(score)


class UserResidualKNNModel(Recommender):
    def __init__(
        self,
        baseline: MeanBaselineModel,
        neighbors: int = 40,
        residual_shrinkage: float = 1.0,
        use_absolute_similarity: bool = True,
    ) -> None:
        self.baseline = baseline
        self.neighbors = neighbors
        self.residual_shrinkage = residual_shrinkage
        self.use_absolute_similarity = use_absolute_similarity
        self.min_rating = 0.0
        self.max_rating = 100.0
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.residual_matrix: np.ndarray | None = None
        self.user_similarity: np.ndarray | None = None
        self.item_users: list[np.ndarray] = []

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("cannot train user residual KNN on empty ratings")
        self.baseline.fit(ratings)
        self.min_rating = self.baseline.min_rating
        self.max_rating = self.baseline.max_rating

        users = sorted({user_id for user_id, _, _ in ratings})
        items = sorted({item_id for _, item_id, _ in ratings})
        self.user_index = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_index = {item_id: idx for idx, item_id in enumerate(items)}

        residuals = np.zeros((len(users), len(items)), dtype=np.float32)
        observed = np.zeros((len(users), len(items)), dtype=bool)
        for user_id, item_id, rating in ratings:
            user_pos = self.user_index[user_id]
            item_pos = self.item_index[item_id]
            residuals[user_pos, item_pos] = rating - self.baseline.predict(user_id, item_id)
            observed[user_pos, item_pos] = True

        norms = np.sqrt(np.sum(residuals * residuals, axis=1, dtype=np.float32))
        similarity = residuals @ residuals.T
        similarity /= norms[:, None] * norms[None, :] + 1e-6
        np.fill_diagonal(similarity, 0.0)

        self.residual_matrix = residuals
        self.user_similarity = similarity
        self.item_users = [np.flatnonzero(observed[:, item_pos]) for item_pos in range(len(items))]

    def predict(self, user_id: int, item_id: int) -> float:
        assert self.residual_matrix is not None
        assert self.user_similarity is not None
        score = self.baseline.predict(user_id, item_id)
        user_pos = self.user_index.get(user_id)
        item_pos = self.item_index.get(item_id)
        if user_pos is None or item_pos is None:
            return self._clip(score)

        users_for_item = self.item_users[item_pos]
        if users_for_item.size == 0:
            return self._clip(score)

        similarities = self.user_similarity[user_pos, users_for_item]
        residuals = self.residual_matrix[users_for_item, item_pos]
        if self.neighbors < similarities.size:
            ranking_values = np.abs(similarities) if self.use_absolute_similarity else similarities
            selected = np.argpartition(ranking_values, -self.neighbors)[-self.neighbors:]
            similarities = similarities[selected]
            residuals = residuals[selected]

        denominator = float(np.sum(np.abs(similarities))) + self.residual_shrinkage
        if denominator > 1e-8:
            score += float(np.dot(similarities, residuals) / denominator)
        return self._clip(score)


class ResidualMatrixFactorizationModel(Recommender):
    def __init__(self, baseline: MeanBaselineModel, config: MFConfig) -> None:
        if not 0.0 <= config.residual_weight <= 2.0:
            raise ValueError("residual_weight must be between 0 and 2")
        self.baseline = baseline
        self.config = config
        self.min_rating = 0.0
        self.max_rating = 100.0
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("cannot train residual matrix factorization on empty ratings")
        self.baseline.fit(ratings)
        self.min_rating = self.baseline.min_rating
        self.max_rating = self.baseline.max_rating

        users = sorted({user_id for user_id, _, _ in ratings})
        items = sorted({item_id for _, item_id, _ in ratings})
        self.user_index = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_index = {item_id: idx for idx, item_id in enumerate(items)}

        rng = np.random.default_rng(self.config.seed)
        scale = 0.05
        self.user_bias = np.zeros(len(users), dtype=np.float64)
        self.item_bias = np.zeros(len(items), dtype=np.float64)
        self.user_factors = rng.normal(0.0, scale, size=(len(users), self.config.factors))
        self.item_factors = rng.normal(0.0, scale, size=(len(items), self.config.factors))

        encoded = np.array(
            [
                (
                    self.user_index[user_id],
                    self.item_index[item_id],
                    rating - self.baseline.predict(user_id, item_id),
                )
                for user_id, item_id, rating in ratings
            ],
            dtype=np.float64,
        )

        for _ in range(self.config.epochs):
            if self.config.shuffle:
                rng.shuffle(encoded)
            for user_pos_raw, item_pos_raw, residual in encoded:
                user_pos = int(user_pos_raw)
                item_pos = int(item_pos_raw)
                prediction = self._predict_residual_known(user_pos, item_pos)
                error = residual - prediction

                user_vector = self.user_factors[user_pos].copy()
                item_vector = self.item_factors[item_pos].copy()
                lr = self.config.learning_rate
                reg = self.config.regularization

                self.user_bias[user_pos] += lr * (error - reg * self.user_bias[user_pos])
                self.item_bias[item_pos] += lr * (error - reg * self.item_bias[item_pos])
                self.user_factors[user_pos] += lr * (error * item_vector - reg * user_vector)
                self.item_factors[item_pos] += lr * (error * user_vector - reg * item_vector)

    def _predict_residual_known(self, user_pos: int, item_pos: int) -> float:
        assert self.user_bias is not None
        assert self.item_bias is not None
        assert self.user_factors is not None
        assert self.item_factors is not None
        return float(
            self.user_bias[user_pos]
            + self.item_bias[item_pos]
            + np.dot(self.user_factors[user_pos], self.item_factors[item_pos])
        )

    def _predict_residual(self, user_id: int, item_id: int) -> float:
        assert self.user_bias is not None
        assert self.item_bias is not None
        assert self.user_factors is not None
        assert self.item_factors is not None
        user_pos = self.user_index.get(user_id)
        item_pos = self.item_index.get(item_id)
        residual = 0.0
        if user_pos is not None:
            residual += float(self.user_bias[user_pos])
        if item_pos is not None:
            residual += float(self.item_bias[item_pos])
        if user_pos is not None and item_pos is not None:
            residual += float(np.dot(self.user_factors[user_pos], self.item_factors[item_pos]))
        return residual

    def predict(self, user_id: int, item_id: int) -> float:
        score = self.baseline.predict(user_id, item_id)
        score += self.config.residual_weight * self._predict_residual(user_id, item_id)
        return self._clip(score)
