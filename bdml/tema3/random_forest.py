from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, sqrt
from typing import Any

import numpy as np


@dataclass
class _TreeNode:
    prediction: Any
    feature_index: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.root: _TreeNode | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.root = self._build_tree(X, y_encoded, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None or self.classes_ is None:
            raise ValueError("The tree must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        predictions = [self.classes_[self._predict_one(row, self.root)] for row in X]
        return np.asarray(predictions)

    def _predict_one(self, row: np.ndarray, node: _TreeNode):
        current = node
        while current.feature_index is not None and current.threshold is not None:
            if row[current.feature_index] <= current.threshold:
                current = current.left  # type: ignore[assignment]
            else:
                current = current.right  # type: ignore[assignment]
        return current.prediction

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        prediction = self._majority_class(y)
        node = _TreeNode(prediction=prediction)

        if self._should_stop(X, y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
        if len(np.unique(y)) == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if X.shape[1] == 0:
            return True
        return False

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        feature_count = self._resolve_max_features(n_features)
        if feature_count <= 0:
            return None

        feature_indices = self.rng.choice(n_features, size=feature_count, replace=False)
        best_gain = -np.inf
        best_split = None
        parent_impurity = self._gini(y)

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])
            if len(values) <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_count = left_mask.sum()
                right_count = right_mask.sum()
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                gain = parent_impurity - (
                    (left_count / n_samples) * self._gini(y[left_mask])
                    + (right_count / n_samples) * self._gini(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, float(threshold))

        return best_split

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None or self.max_features == "auto":
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(log2(n_features)))
            raise ValueError(f"Unknown max_features value: {self.max_features}")
        if isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("float max_features must be in the interval (0, 1]")
            return max(1, int(ceil(self.max_features * n_features)))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        raise TypeError("max_features must be int, float, str, or None")

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities**2)

    def _majority_class(self, y: np.ndarray):
        counts = np.bincount(y)
        return int(np.argmax(counts))


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        sample_fraction: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the interval (0, 1]")

        self.classes_ = np.unique(y)
        self.trees = []
        sample_size = max(1, int(round(len(X) * self.sample_fraction)))

        for _ in range(self.n_estimators):
            tree_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            if self.bootstrap:
                indices = self.rng.integers(0, len(X), size=sample_size)
            else:
                indices = self.rng.choice(len(X), size=sample_size, replace=False)

            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("The forest must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        votes = np.asarray([tree.predict(X) for tree in self.trees])
        predictions = [
            self._majority_vote(votes[:, idx]) for idx in range(votes.shape[1])
        ]
        return np.asarray(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("The forest must be fitted before calling predict_proba")

        X = np.asarray(X, dtype=float)
        votes = np.asarray([tree.predict(X) for tree in self.trees])
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)

        for sample_index in range(len(X)):
            sample_votes = votes[:, sample_index]
            for class_index, label in enumerate(self.classes_):
                probabilities[sample_index, class_index] = np.mean(
                    sample_votes == label
                )

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def _majority_vote(self, values: np.ndarray):
        unique_values, counts = np.unique(values, return_counts=True)
        return unique_values[np.argmax(counts)]


def make_toy_dataset(n_samples: int = 300, random_state: int | None = 0):
    rng = np.random.default_rng(random_state)
    x1 = rng.normal(loc=(-2.0, -1.5), scale=0.9, size=(n_samples // 3, 2))
    x2 = rng.normal(loc=(2.2, 1.8), scale=0.8, size=(n_samples // 3, 2))
    x3 = rng.normal(
        loc=(0.5, 3.2), scale=0.7, size=(n_samples - 2 * (n_samples // 3), 2)
    )

    X = np.vstack((x1, x2, x3))
    y = np.concatenate(
        (
            np.full(len(x1), 0, dtype=int),
            np.full(len(x2), 1, dtype=int),
            np.full(len(x3), 2, dtype=int),
        )
    )

    permutation = rng.permutation(len(X))
    return X[permutation], y[permutation]
