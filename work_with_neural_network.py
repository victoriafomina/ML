import math

import numpy as np
from numpy import linalg

epsilon = 0.000001


def gradient_descent(
    x: np.array, y: np.array, initial_weights: np.array, step_function, stop_criterion
) -> np.array:
    current_weights = initial_weights
    num_x = len(x)

    num_step = 1
    while True:
        current_prediction = x.dot(current_weights)
        error = current_prediction - y
        current_gradient = 2 * x.T.dot(error) / num_x
        new_weights = current_weights - step_function(num_step) * current_gradient

        if stop_criterion(num_step, current_weights, new_weights, current_gradient):
            return new_weights

        current_weights = new_weights
        num_step += 1


def step(num_step) -> float:
    return 1.0 / num_step


def get_stop_criterion(num_step, weight0, weight1, b) -> bool:
    return num_step != 1 and linalg.norm(weight1 - weight0) < epsilon


def add_ones_column(x) -> np.hstack:
    ones = np.ones((x.shape[0], 1))
    return np.hstack((x, ones))


def get_linear_regression(x, y) -> np.array:
    x = add_ones_column(x)
    return gradient_descent(x, y, np.zeros(x.shape[1]), step, get_stop_criterion)


def get_predict(x, weights) -> np.array:
    return x.dot(weights)


def MSE(ground_truth, prediction):
    return np.square(ground_truth - prediction).mean()


def RMSE(ground_truth, prediction) -> float:
    return math.sqrt(MSE(ground_truth, prediction))


def r_squared(ground_truth, prediction) -> float:
    result = np.square(ground_truth - prediction).sum()
    total = np.square(ground_truth - ground_truth.mean()).sum()
    return 1.0 - result / total


def evaluate_metrics(x, y, weights) -> tuple:
    x = add_ones_column(x)
    prediction = get_predict(x, weights)
    mse = MSE(y, prediction)
    rmse = math.sqrt(mse)
    r_sq = r_squared(y, prediction)
    return mse, rmse, r_sq
