from typing import Any

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class KNN(BaseEstimator):

    def __init__(self, k=3, p_minkowski=2):
        self.k = k
        self.n_features_in_ = None
        self.p_minkowski = p_minkowski
        self.x = None
        self.y = None

    def predict(self, x) -> ndarray:
        check_is_fitted(self)

        return np.array([self.singlePrediction(x_test_row) for x_test_row in check_array(x)])

    def singlePrediction(self, x_test_row) -> Any:
        distances = [
            minkowskiDistance(x_test_row, x_train_row, self.p_minkowski)
            for x_train_row in self.x
        ]
        k_index = np.argsort(distances)[: self.k]
        k_labels = [self.y[idx] for idx in k_index]
        return np.argmax(np.bincount(k_labels))

    def score(self, x_test, y_test) -> float:
        return accuracy_score(y_test, self.predict(x_test))

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        self.n_features_in_ = x.shape[1]
        self.x = x
        self.y = y

        return self


def minkowskiDistance(x1, x2, p) -> ndarray:
    return np.power(np.abs(np.sum((x1 - x2) ** p)), 1 / p)
