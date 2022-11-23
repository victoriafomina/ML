from typing import Any

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class DTC(BaseEstimator):
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_class_labels = None
        self.n_samples = None
        self.n_features = None
        self.n_features_in_ = None
        self.root = None

    def createTree(self, x, y, depth=0):
        self.n_samples, self.n_features = x.shape
        self.n_class_labels = len(np.unique(y))

        if self.isFinished(depth):
            return Node(value=np.argmax(np.bincount(y)))

        best_feat, best_thresh = bestSplitting(x, y, list(range(self.n_features)))

        left_idx, right_idx = createSplitting(x[:, best_feat], best_thresh)
        left_child = self.createTree(x[left_idx, :], y[left_idx], depth + 1)
        right_child = self.createTree(x[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        self.n_features_in_ = x.shape[1]
        self.root = self.createTree(x, np.array(y, dtype='int64'))

        return self

    def isFinished(self, depth) -> bool:
        return (
            depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split
        )

    def predict(self, x) -> ndarray:
        check_is_fitted(self)

        return np.array([self.traverseTree(elem, self.root) for elem in check_array(x)])

    def score(self, x_test, y_test) -> float:
        return accuracy_score(y_test, self.predict(x_test))

    def traverseTree(self, x, node) -> Any:
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.left)

        return self.traverseTree(x, node.right)


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value


def bestSplitting(x, y, features):
    split = {'score': -1, 'feat': None, 'thresh': None}

    for feat in features:
        x_feat = x[:, feat]
        thresholds = np.unique(x_feat)
        for thresh in thresholds:
            score = informationGrowth(x_feat, y, thresh)

            if score > split['score']:
                split['score'] = score
                split['feat'] = feat
                split['thresh'] = thresh

    return split['feat'], split['thresh']


def createSplitting(x, thresh) -> tuple:
    return np.argwhere(x <= thresh).flatten(), np.argwhere(x > thresh).flatten()


def getEntropy(y) -> Any:
    return -np.sum([p * np.log2(p) for p in np.bincount(y) / len(y) if p > 0])


def informationGrowth(x, y, thresh):
    left_idx, right_idx = createSplitting(x, thresh)

    if len(left_idx) == 0 or len(right_idx) == 0:
        return 0

    return getEntropy(y) - (len(left_idx) / len(y)) * getEntropy(y[left_idx]) + (len(right_idx) / len(y)) * getEntropy(
        y[right_idx]
    )
