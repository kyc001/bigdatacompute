from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log1p, sqrt

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
    def __init__(
        self,
        weighted_models: list[tuple[float, Recommender]],
        normalize: bool = True,
        intercept: float = 0.0,
    ) -> None:
        if not weighted_models:
            raise ValueError("ensemble requires at least one model")
        total_weight = sum(weight for weight, _ in weighted_models)
        if normalize:
            if total_weight <= 0.0:
                raise ValueError("ensemble weights must sum to a positive value")
            self.weighted_models = [(weight / total_weight, model) for weight, model in weighted_models]
        else:
            if not any(weight != 0.0 for weight, _ in weighted_models):
                raise ValueError("ensemble requires at least one non-zero weight")
            self.weighted_models = weighted_models
        self.intercept = intercept
        self.min_rating = 0.0
        self.max_rating = 100.0

    def fit(self, ratings: list[Rating]) -> None:
        for _, model in self.weighted_models:
            model.fit(ratings)
        first_model = self.weighted_models[0][1]
        self.min_rating = first_model.min_rating
        self.max_rating = first_model.max_rating

    def predict(self, user_id: int, item_id: int) -> float:
        score = self.intercept + sum(weight * model.predict(user_id, item_id) for weight, model in self.weighted_models)
        return self._clip(score)


class FeatureCalibratedModel(Recommender):
    def __init__(
        self,
        base_model: Recommender,
        coefficients: dict[str, float],
        intercept: float,
    ) -> None:
        self.base_model = base_model
        self.coefficients = coefficients
        self.intercept = intercept
        self.global_mean = 0.0
        self.user_count: dict[int, int] = {}
        self.item_count: dict[int, int] = {}
        self.user_mean: dict[int, float] = {}
        self.item_mean: dict[int, float] = {}
        self.user_variance: dict[int, float] = {}
        self.item_variance: dict[int, float] = {}
        self.user_median: dict[int, float] = {}
        self.item_median: dict[int, float] = {}
        self.user_iqr: dict[int, float] = {}
        self.item_iqr: dict[int, float] = {}
        self.user_min: dict[int, float] = {}
        self.item_min: dict[int, float] = {}
        self.user_max: dict[int, float] = {}
        self.item_max: dict[int, float] = {}
        self.user_skewness: dict[int, float] = {}
        self.item_skewness: dict[int, float] = {}
        self.min_rating = 0.0
        self.max_rating = 100.0

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("cannot train feature-calibrated model on empty ratings")
        self.base_model.fit(ratings)
        self.min_rating = self.base_model.min_rating
        self.max_rating = self.base_model.max_rating

        user_sum: defaultdict[int, float] = defaultdict(float)
        item_sum: defaultdict[int, float] = defaultdict(float)
        user_square_sum: defaultdict[int, float] = defaultdict(float)
        item_square_sum: defaultdict[int, float] = defaultdict(float)
        user_values: defaultdict[int, list[float]] = defaultdict(list)
        item_values: defaultdict[int, list[float]] = defaultdict(list)
        user_count: defaultdict[int, int] = defaultdict(int)
        item_count: defaultdict[int, int] = defaultdict(int)
        total = 0.0
        for user_id, item_id, rating in ratings:
            total += rating
            user_sum[user_id] += rating
            item_sum[item_id] += rating
            user_square_sum[user_id] += rating * rating
            item_square_sum[item_id] += rating * rating
            user_values[user_id].append(rating)
            item_values[item_id].append(rating)
            user_count[user_id] += 1
            item_count[item_id] += 1
        self.global_mean = total / len(ratings)
        self.user_count = dict(user_count)
        self.item_count = dict(item_count)
        self.user_mean = {user_id: user_sum[user_id] / count for user_id, count in user_count.items()}
        self.item_mean = {item_id: item_sum[item_id] / count for item_id, count in item_count.items()}
        self.user_variance = {
            user_id: max(0.0, user_square_sum[user_id] / count - self.user_mean[user_id] ** 2)
            for user_id, count in user_count.items()
        }
        self.item_variance = {
            item_id: max(0.0, item_square_sum[item_id] / count - self.item_mean[item_id] ** 2)
            for item_id, count in item_count.items()
        }
        self.user_median = {user_id: float(np.median(values)) for user_id, values in user_values.items()}
        self.item_median = {item_id: float(np.median(values)) for item_id, values in item_values.items()}
        self.user_iqr = {
            user_id: float(np.percentile(values, 75) - np.percentile(values, 25))
            for user_id, values in user_values.items()
        }
        self.item_iqr = {
            item_id: float(np.percentile(values, 75) - np.percentile(values, 25))
            for item_id, values in item_values.items()
        }
        self.user_min = {user_id: min(values) for user_id, values in user_values.items()}
        self.item_min = {item_id: min(values) for item_id, values in item_values.items()}
        self.user_max = {user_id: max(values) for user_id, values in user_values.items()}
        self.item_max = {item_id: max(values) for item_id, values in item_values.items()}
        self.user_skewness = {
            user_id: self._skewness(values, self.user_mean[user_id], self.user_variance[user_id])
            for user_id, values in user_values.items()
        }
        self.item_skewness = {
            item_id: self._skewness(values, self.item_mean[item_id], self.item_variance[item_id])
            for item_id, values in item_values.items()
        }

    def predict(self, user_id: int, item_id: int) -> float:
        base_score = self.base_model.predict(user_id, item_id)
        user_count = self.user_count.get(user_id, 0)
        item_count = self.item_count.get(item_id, 0)
        user_mean = self.user_mean.get(user_id, self.global_mean)
        item_mean = self.item_mean.get(item_id, self.global_mean)
        user_variance = self.user_variance.get(user_id, 0.0)
        item_variance = self.item_variance.get(item_id, 0.0)
        user_median = self.user_median.get(user_id, self.global_mean)
        item_median = self.item_median.get(item_id, self.global_mean)
        user_std = sqrt(user_variance)
        item_std = sqrt(item_variance)
        user_iqr = self.user_iqr.get(user_id, 0.0)
        item_iqr = self.item_iqr.get(item_id, 0.0)
        user_min = self.user_min.get(user_id, self.global_mean)
        item_min = self.item_min.get(item_id, self.global_mean)
        user_max = self.user_max.get(user_id, self.global_mean)
        item_max = self.item_max.get(item_id, self.global_mean)
        user_range = user_max - user_min if user_id in self.user_min else 0.0
        item_range = item_max - item_min if item_id in self.item_min else 0.0
        user_mean_delta = user_mean - self.global_mean
        item_mean_delta = item_mean - self.global_mean
        user_median_delta = user_median - self.global_mean
        item_median_delta = item_median - self.global_mean
        log_user_count = log1p(user_count)
        log_item_count = log1p(item_count)
        inv_user_count = 1.0 / sqrt(user_count + 1.0)
        inv_item_count = 1.0 / sqrt(item_count + 1.0)
        abs_user_mean_delta = abs(user_mean_delta)
        abs_item_mean_delta = abs(item_mean_delta)
        user_skewness = self.user_skewness.get(user_id, 0.0)
        item_skewness = self.item_skewness.get(item_id, 0.0)
        base_center = base_score - self.global_mean
        base_minus_user_mean = base_score - user_mean
        base_minus_item_mean = base_score - item_mean
        base_minus_user_median = base_score - user_median
        base_minus_item_median = base_score - item_median
        sqrt_user_count = sqrt(user_count + 1.0)
        sqrt_item_count = sqrt(item_count + 1.0)
        log_count_sum = log_user_count + log_item_count
        log_count_abs_diff = abs(log_user_count - log_item_count)
        pred_bin_index = min(9, max(0, int(base_score // 10)))
        uc_le_2 = user_count <= 2
        uc_le_5 = user_count <= 5
        uc_le_15 = user_count <= 15
        uc_le_20 = user_count <= 20
        uc_le_30 = user_count <= 30
        uc_le_50 = user_count <= 50
        uc_le_75 = user_count <= 75
        uc_gt_100 = user_count > 100
        uc_gt_150 = user_count > 150
        ic_le_2 = item_count <= 2
        ic_le_4 = item_count <= 4
        ic_le_8 = item_count <= 8
        ic_le_15 = item_count <= 15
        ic_le_20 = item_count <= 20
        ic_le_30 = item_count <= 30
        ic_gt_100 = item_count > 100
        uc_le_10 = user_count <= 10
        uc_gt_50 = user_count > 50
        ic_le_10 = item_count <= 10
        ic_gt_20 = item_count > 20
        u_mid = 5 < user_count <= 30
        i_mid = 2 < item_count <= 10

        score = self.intercept
        score += self.coefficients["pred"] * base_score
        score += self.coefficients.get("log_uc", 0.0) * log_user_count
        score += self.coefficients.get("log_ic", 0.0) * log_item_count
        score += self.coefficients.get("log_uc2", 0.0) * log_user_count * log_user_count
        score += self.coefficients.get("log_ic2", 0.0) * log_item_count * log_item_count
        score += self.coefficients.get("log_ratio", 0.0) * (log_user_count - log_item_count)
        score += self.coefficients.get("inv_uc", 0.0) * inv_user_count
        score += self.coefficients.get("inv_ic", 0.0) * inv_item_count
        score += self.coefficients.get("umean", 0.0) * user_mean
        score += self.coefficients.get("imean", 0.0) * item_mean
        score += self.coefficients.get("umean_g", 0.0) * user_mean_delta
        score += self.coefficients.get("imean_g", 0.0) * item_mean_delta
        score += self.coefficients.get("mean_sum_g", 0.0) * (user_mean_delta + item_mean_delta)
        score += self.coefficients.get("mean_diff", 0.0) * (user_mean - item_mean)
        score += self.coefficients.get("mean_inter", 0.0) * user_mean_delta * item_mean_delta / 100.0
        score += self.coefficients.get("pred2", 0.0) * base_score * base_score / 100.0
        score += self.coefficients.get("uvar", 0.0) * user_variance
        score += self.coefficients.get("ivar", 0.0) * item_variance
        score += self.coefficients.get("ustd", 0.0) * user_std
        score += self.coefficients.get("istd", 0.0) * item_std
        score += self.coefficients.get("var_sum", 0.0) * (user_variance + item_variance)
        score += self.coefficients.get("std_sum", 0.0) * (user_std + item_std)
        score += self.coefficients.get("abs_ug", 0.0) * abs_user_mean_delta
        score += self.coefficients.get("abs_ig", 0.0) * abs_item_mean_delta
        score += self.coefficients.get("umed", 0.0) * user_median
        score += self.coefficients.get("imed", 0.0) * item_median
        score += self.coefficients.get("umed_g", 0.0) * user_median_delta
        score += self.coefficients.get("imed_g", 0.0) * item_median_delta
        score += self.coefficients.get("med_diff", 0.0) * (user_median - item_median)
        score += self.coefficients.get("med_inter", 0.0) * user_median_delta * item_median_delta / 100.0
        score += self.coefficients.get("uiqr", 0.0) * user_iqr
        score += self.coefficients.get("iiqr", 0.0) * item_iqr
        score += self.coefficients.get("iqr_sum", 0.0) * (user_iqr + item_iqr)
        score += self.coefficients.get("iqr_diff", 0.0) * (user_iqr - item_iqr)
        score += self.coefficients.get("urange", 0.0) * user_range
        score += self.coefficients.get("irange", 0.0) * item_range
        score += self.coefficients.get("range_sum", 0.0) * (user_range + item_range)
        score += self.coefficients.get("umin_g", 0.0) * (user_min - self.global_mean)
        score += self.coefficients.get("imin_g", 0.0) * (item_min - self.global_mean)
        score += self.coefficients.get("umax_g", 0.0) * (user_max - self.global_mean)
        score += self.coefficients.get("imax_g", 0.0) * (item_max - self.global_mean)
        score += self.coefficients.get("uskew", 0.0) * user_skewness
        score += self.coefficients.get("iskew", 0.0) * item_skewness
        if user_count == 0:
            score += self.coefficients.get("uc0", 0.0)
        if item_count == 0:
            score += self.coefficients.get("ic0", 0.0)
        if user_count <= 1:
            score += self.coefficients.get("uc_le_1", 0.0)
        if user_count <= 3:
            score += self.coefficients.get("uc_le_3", 0.0)
        if uc_le_10:
            score += self.coefficients.get("uc_le_10", 0.0)
        if uc_gt_50:
            score += self.coefficients.get("uc_gt_50", 0.0)
        if item_count <= 1:
            score += self.coefficients.get("ic_le_1", 0.0)
        if item_count <= 3:
            score += self.coefficients.get("ic_le_3", 0.0)
        if ic_le_10:
            score += self.coefficients.get("ic_le_10", 0.0)
        if ic_gt_20:
            score += self.coefficients.get("ic_gt_20", 0.0)
        if user_count <= 5:
            score += self.coefficients.get("u_low", 0.0)
        if item_count <= 2:
            score += self.coefficients.get("i_low", 0.0)
        if u_mid:
            score += self.coefficients.get("u_mid", 0.0)
        if i_mid:
            score += self.coefficients.get("i_mid", 0.0)
        score += self.coefficients.get("log_prod", 0.0) * log_user_count * log_item_count
        score += self.coefficients.get("pred_log_uc", 0.0) * base_score * log_user_count / 100.0
        score += self.coefficients.get("pred_log_ic", 0.0) * base_score * log_item_count / 100.0
        score += self.coefficients.get("pred_log_ratio", 0.0) * base_score * (log_user_count - log_item_count) / 100.0
        score += self.coefficients.get("pred_inv_uc", 0.0) * base_score * inv_user_count / 100.0
        score += self.coefficients.get("pred_inv_ic", 0.0) * base_score * inv_item_count / 100.0
        score += self.coefficients.get("pred_umean_g", 0.0) * base_score * user_mean_delta / 100.0
        score += self.coefficients.get("pred_imean_g", 0.0) * base_score * item_mean_delta / 100.0
        score += self.coefficients.get("pred_abs_ug", 0.0) * base_score * abs_user_mean_delta / 100.0
        score += self.coefficients.get("pred_abs_ig", 0.0) * base_score * abs_item_mean_delta / 100.0
        score += self.coefficients.get("pred_uvar", 0.0) * base_score * user_variance / 1000.0
        score += self.coefficients.get("pred_ivar", 0.0) * base_score * item_variance / 1000.0
        score += self.coefficients.get("log_uc_umean_g", 0.0) * log_user_count * user_mean_delta / 10.0
        score += self.coefficients.get("log_ic_imean_g", 0.0) * log_item_count * item_mean_delta / 10.0
        score += self.coefficients.get("log_uc_imean_g", 0.0) * log_user_count * item_mean_delta / 10.0
        score += self.coefficients.get("log_ic_umean_g", 0.0) * log_item_count * user_mean_delta / 10.0
        score += self.coefficients.get("inv_uc_umean_g", 0.0) * inv_user_count * user_mean_delta
        score += self.coefficients.get("inv_ic_imean_g", 0.0) * inv_item_count * item_mean_delta
        score += self.coefficients.get("inv_uc_imean_g", 0.0) * inv_user_count * item_mean_delta
        score += self.coefficients.get("inv_ic_umean_g", 0.0) * inv_item_count * user_mean_delta
        score += self.coefficients.get("ustd_istd", 0.0) * user_std * item_std / 100.0
        score += self.coefficients.get("uvar_ivar", 0.0) * user_variance * item_variance / 1000.0
        score += self.coefficients.get("uiqr_iiqr", 0.0) * user_iqr * item_iqr / 100.0
        score += self.coefficients.get("urange_irange", 0.0) * user_range * item_range / 100.0
        score += self.coefficients.get("uskew_iskew", 0.0) * user_skewness * item_skewness
        if uc_le_10 and ic_le_10:
            score += self.coefficients.get("uc10_ic10", 0.0)
        if uc_gt_50 and ic_gt_20:
            score += self.coefficients.get("ucgt50_icgt20", 0.0)
        if u_mid and i_mid:
            score += self.coefficients.get("u_mid_i_mid", 0.0)
        score += self.coefficients.get("pred3", 0.0) * base_score**3 / 10000.0
        score += self.coefficients.get("pred_abs_err_center", 0.0) * abs(base_score - self.global_mean) / 100.0
        score += self.coefficients.get("pred_center2", 0.0) * (base_score - self.global_mean) ** 2 / 100.0
        score += self.coefficients.get("mean_abs_diff", 0.0) * abs(user_mean - item_mean)
        score += self.coefficients.get("mean_diff2", 0.0) * (user_mean - item_mean) ** 2 / 100.0
        score += self.coefficients.get("median_abs_diff", 0.0) * abs(user_median - item_median)
        score += self.coefficients.get("median_diff2", 0.0) * (user_median - item_median) ** 2 / 100.0
        score += self.coefficients.get("mean_med_user_gap", 0.0) * (user_mean - user_median)
        score += self.coefficients.get("mean_med_item_gap", 0.0) * (item_mean - item_median)
        score += self.coefficients.get("abs_mean_med_user_gap", 0.0) * abs(user_mean - user_median)
        score += self.coefficients.get("abs_mean_med_item_gap", 0.0) * abs(item_mean - item_median)
        score += self.coefficients.get("std_diff", 0.0) * (user_std - item_std)
        score += self.coefficients.get("std_abs_diff", 0.0) * abs(user_std - item_std)
        score += self.coefficients.get("var_diff", 0.0) * (user_variance - item_variance)
        score += self.coefficients.get("var_abs_diff", 0.0) * abs(user_variance - item_variance)
        score += self.coefficients.get("iqr_abs_diff", 0.0) * abs(user_iqr - item_iqr)
        score += self.coefficients.get("range_abs_diff", 0.0) * abs(user_range - item_range)
        score += self.coefficients.get("skew_abs_diff", 0.0) * abs(user_skewness - item_skewness)
        score += self.coefficients.get("log_uc3", 0.0) * log_user_count**3
        score += self.coefficients.get("log_ic3", 0.0) * log_item_count**3
        score += self.coefficients.get("log_uc_log_ic2", 0.0) * log_user_count * log_item_count**2
        score += self.coefficients.get("log_ic_log_uc2", 0.0) * log_item_count * log_user_count**2
        score += self.coefficients.get("inv_uc2", 0.0) * inv_user_count**2
        score += self.coefficients.get("inv_ic2", 0.0) * inv_item_count**2
        score += self.coefficients.get("inv_uc_inv_ic", 0.0) * inv_user_count * inv_item_count
        if user_count <= 20:
            score += self.coefficients.get("uc_le_20", 0.0)
        if user_count <= 50:
            score += self.coefficients.get("uc_le_50", 0.0)
        if user_count > 100:
            score += self.coefficients.get("uc_gt_100", 0.0)
        if item_count <= 5:
            score += self.coefficients.get("ic_le_5", 0.0)
        if item_count <= 20:
            score += self.coefficients.get("ic_le_20", 0.0)
        if item_count > 50:
            score += self.coefficients.get("ic_gt_50", 0.0)
        if user_count <= 20 and item_count <= 5:
            score += self.coefficients.get("uc20_ic5", 0.0)
        if user_count <= 50 and item_count <= 20:
            score += self.coefficients.get("uc50_ic20", 0.0)
        if user_count > 100 and item_count > 50:
            score += self.coefficients.get("ucgt100_icgt50", 0.0)
        if uc_le_10 and item_count > 20:
            score += self.coefficients.get("uc_low_ic_high", 0.0)
        if uc_gt_50 and ic_le_10:
            score += self.coefficients.get("uc_high_ic_low", 0.0)
        if uc_le_10:
            score += self.coefficients.get("pred_uc_le_10", 0.0) * base_score / 100.0
        if ic_le_10:
            score += self.coefficients.get("pred_ic_le_10", 0.0) * base_score / 100.0
        if uc_gt_50:
            score += self.coefficients.get("pred_uc_gt_50", 0.0) * base_score / 100.0
        if ic_gt_20:
            score += self.coefficients.get("pred_ic_gt_20", 0.0) * base_score / 100.0
        if u_mid:
            score += self.coefficients.get("pred_u_mid", 0.0) * base_score / 100.0
        if i_mid:
            score += self.coefficients.get("pred_i_mid", 0.0) * base_score / 100.0
        score += self.coefficients.get("abs_ug_abs_ig", 0.0) * abs_user_mean_delta * abs_item_mean_delta / 100.0
        score += self.coefficients.get("abs_ug_log_uc", 0.0) * abs_user_mean_delta * log_user_count / 10.0
        score += self.coefficients.get("abs_ig_log_ic", 0.0) * abs_item_mean_delta * log_item_count / 10.0
        score += self.coefficients.get("abs_ug_inv_uc", 0.0) * abs_user_mean_delta * inv_user_count
        score += self.coefficients.get("abs_ig_inv_ic", 0.0) * abs_item_mean_delta * inv_item_count
        score += self.coefficients.get("umean_g2", 0.0) * user_mean_delta**2 / 100.0
        score += self.coefficients.get("imean_g2", 0.0) * item_mean_delta**2 / 100.0
        score += self.coefficients.get("umed_g2", 0.0) * user_median_delta**2 / 100.0
        score += self.coefficients.get("imed_g2", 0.0) * item_median_delta**2 / 100.0
        score += self.coefficients.get("uvar_log_uc", 0.0) * user_variance * log_user_count / 100.0
        score += self.coefficients.get("ivar_log_ic", 0.0) * item_variance * log_item_count / 100.0
        score += self.coefficients.get("ustd_log_uc", 0.0) * user_std * log_user_count / 10.0
        score += self.coefficients.get("istd_log_ic", 0.0) * item_std * log_item_count / 10.0
        score += self.coefficients.get("pred4", 0.0) * base_score**4 / 1000000.0
        score += self.coefficients.get("pred_center3", 0.0) * base_center**3 / 1000.0
        score += self.coefficients.get("pred_center4", 0.0) * base_center**4 / 10000.0
        if base_score < 20.0:
            score += self.coefficients.get("pred_low_20", 0.0)
        if base_score < 40.0:
            score += self.coefficients.get("pred_low_40", 0.0)
        if 40.0 <= base_score < 60.0:
            score += self.coefficients.get("pred_mid_40_60", 0.0)
        if base_score >= 60.0:
            score += self.coefficients.get("pred_high_60", 0.0)
        if base_score >= 80.0:
            score += self.coefficients.get("pred_high_80", 0.0)
        if base_score >= 90.0:
            score += self.coefficients.get("pred_high_90", 0.0)
        pred_bin = f"pred_bin_{pred_bin_index}"
        score += self.coefficients.get(pred_bin, 0.0)
        score += self.coefficients.get("pred_log_prod", 0.0) * base_score * log_user_count * log_item_count / 100.0
        score += self.coefficients.get("pred_std_diff", 0.0) * base_score * (user_std - item_std) / 100.0
        score += self.coefficients.get("pred_std_abs_diff", 0.0) * base_score * abs(user_std - item_std) / 100.0
        score += self.coefficients.get("pred_iqr_diff", 0.0) * base_score * (user_iqr - item_iqr) / 100.0
        score += self.coefficients.get("pred_range_diff", 0.0) * base_score * (user_range - item_range) / 100.0
        score += self.coefficients.get("pred_skew_diff", 0.0) * base_score * (user_skewness - item_skewness) / 100.0
        score += self.coefficients.get("pred_mean_abs_diff", 0.0) * base_score * abs(user_mean - item_mean) / 100.0
        score += self.coefficients.get("pred_median_abs_diff", 0.0) * base_score * abs(user_median - item_median) / 100.0
        score += self.coefficients.get("umean_imean", 0.0) * user_mean * item_mean / 100.0
        score += self.coefficients.get("umed_imed", 0.0) * user_median * item_median / 100.0
        score += self.coefficients.get("umean_imed", 0.0) * user_mean * item_median / 100.0
        score += self.coefficients.get("umed_imean", 0.0) * user_median * item_mean / 100.0
        score += self.coefficients.get("base_minus_umean", 0.0) * base_minus_user_mean / 100.0
        score += self.coefficients.get("base_minus_imean", 0.0) * base_minus_item_mean / 100.0
        score += self.coefficients.get("base_minus_umed", 0.0) * base_minus_user_median / 100.0
        score += self.coefficients.get("base_minus_imed", 0.0) * base_minus_item_median / 100.0
        score += self.coefficients.get("abs_base_minus_umean", 0.0) * abs(base_minus_user_mean) / 100.0
        score += self.coefficients.get("abs_base_minus_imean", 0.0) * abs(base_minus_item_mean) / 100.0
        score += self.coefficients.get("abs_base_minus_umed", 0.0) * abs(base_minus_user_median) / 100.0
        score += self.coefficients.get("abs_base_minus_imed", 0.0) * abs(base_minus_item_median) / 100.0
        score += self.coefficients.get("base_umean_gap2", 0.0) * base_minus_user_mean * base_minus_user_mean / 100.0
        score += self.coefficients.get("base_imean_gap2", 0.0) * base_minus_item_mean * base_minus_item_mean / 100.0
        score += self.coefficients.get("base_umed_gap2", 0.0) * base_minus_user_median * base_minus_user_median / 100.0
        score += self.coefficients.get("base_imed_gap2", 0.0) * base_minus_item_median * base_minus_item_median / 100.0
        score += self.coefficients.get("base_mean_gap_inter", 0.0) * base_minus_user_mean * base_minus_item_mean / 100.0
        score += self.coefficients.get("base_median_gap_inter", 0.0) * base_minus_user_median * base_minus_item_median / 100.0
        score += self.coefficients.get("sqrt_uc", 0.0) * sqrt_user_count
        score += self.coefficients.get("sqrt_ic", 0.0) * sqrt_item_count
        score += self.coefficients.get("sqrt_uc_sqrt_ic", 0.0) * sqrt_user_count * sqrt_item_count
        score += self.coefficients.get("log_count_sum", 0.0) * log_count_sum
        score += self.coefficients.get("log_count_abs_diff", 0.0) * log_count_abs_diff
        score += self.coefficients.get("log_uc_inv_uc", 0.0) * log_user_count * inv_user_count
        score += self.coefficients.get("log_ic_inv_ic", 0.0) * log_item_count * inv_item_count
        score += self.coefficients.get("log_uc_inv_ic", 0.0) * log_user_count * inv_item_count
        score += self.coefficients.get("log_ic_inv_uc", 0.0) * log_item_count * inv_user_count
        if uc_le_2:
            score += self.coefficients.get("uc_le_2", 0.0)
        if uc_le_5:
            score += self.coefficients.get("uc_le_5", 0.0)
        if uc_le_15:
            score += self.coefficients.get("uc_le_15", 0.0)
        if uc_le_30:
            score += self.coefficients.get("uc_le_30", 0.0)
        if uc_le_75:
            score += self.coefficients.get("uc_le_75", 0.0)
        if uc_gt_150:
            score += self.coefficients.get("uc_gt_150", 0.0)
        if ic_le_2:
            score += self.coefficients.get("ic_le_2", 0.0)
        if ic_le_4:
            score += self.coefficients.get("ic_le_4", 0.0)
        if ic_le_8:
            score += self.coefficients.get("ic_le_8", 0.0)
        if ic_le_15:
            score += self.coefficients.get("ic_le_15", 0.0)
        if ic_le_30:
            score += self.coefficients.get("ic_le_30", 0.0)
        if ic_gt_100:
            score += self.coefficients.get("ic_gt_100", 0.0)
        if uc_le_2:
            score += self.coefficients.get("pred_uc_le_2", 0.0) * base_score / 100.0
        if uc_le_5:
            score += self.coefficients.get("pred_uc_le_5", 0.0) * base_score / 100.0
        if uc_le_15:
            score += self.coefficients.get("pred_uc_le_15", 0.0) * base_score / 100.0
        if uc_le_30:
            score += self.coefficients.get("pred_uc_le_30", 0.0) * base_score / 100.0
        if uc_le_75:
            score += self.coefficients.get("pred_uc_le_75", 0.0) * base_score / 100.0
        if ic_le_2:
            score += self.coefficients.get("pred_ic_le_2", 0.0) * base_score / 100.0
        if ic_le_4:
            score += self.coefficients.get("pred_ic_le_4", 0.0) * base_score / 100.0
        if ic_le_8:
            score += self.coefficients.get("pred_ic_le_8", 0.0) * base_score / 100.0
        if ic_le_15:
            score += self.coefficients.get("pred_ic_le_15", 0.0) * base_score / 100.0
        if ic_le_30:
            score += self.coefficients.get("pred_ic_le_30", 0.0) * base_score / 100.0
        if uc_le_2 and ic_le_2:
            score += self.coefficients.get("uc2_ic2", 0.0)
        if uc_le_5 and ic_le_4:
            score += self.coefficients.get("uc5_ic4", 0.0)
        if uc_le_15 and ic_le_8:
            score += self.coefficients.get("uc15_ic8", 0.0)
        if uc_le_30 and ic_le_15:
            score += self.coefficients.get("uc30_ic15", 0.0)
        if uc_le_75 and ic_le_30:
            score += self.coefficients.get("uc75_ic30", 0.0)
        if uc_gt_150 and ic_gt_100:
            score += self.coefficients.get("ucgt150_icgt100", 0.0)
        return self._clip(score)

    @staticmethod
    def _skewness(values: list[float], mean: float, variance: float) -> float:
        if variance <= 0.0:
            return 0.0
        std = sqrt(variance)
        return float(np.mean([(value - mean) ** 3 for value in values]) / (std ** 3 + 1e-9))


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
