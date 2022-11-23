import pandas
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from work_with_data import get_normalize_data
from work_with_neural_network import evaluate_metrics, get_linear_regression

data = get_normalize_data("car_price.csv")

y = data["price"].to_numpy()
X = data.drop(columns=["price"]).to_numpy()

table_data = {
    "": [
        "mse-learning",
        "mse-test",
        "rmse-learning",
        "rmse-test",
        "r2-learning",
        "r2-test",
    ]
}
table_data_sklearn = {
    "": [
        "mse-learning",
        "mse-test",
        "rmse-learning",
        "rmse-test",
        "r2-learning",
        "r2-test",
    ]
}

k_fold = KFold(n_splits=5, random_state=None, shuffle=True)
fold_step = 1
for learning_index, test_index in k_fold.split(X):
    x_learning, x_test = X[learning_index], X[test_index]
    y_learning, y_test = y[learning_index], y[test_index]

    weights = get_linear_regression(x_learning, y_learning)
    learning_mse, learning_rmse, learning_r_sq = evaluate_metrics(
        x_learning, y_learning, weights
    )
    test_mse, test_rmse, test_r_sq = evaluate_metrics(x_test, y_test, weights)

    table_data["Fold" + str(fold_step)] = [
        learning_mse,
        test_mse,
        learning_rmse,
        test_rmse,
        learning_r_sq,
        test_r_sq,
    ]

    model = Ridge()

    model.fit(x_learning, y_learning)

    y_learning_predict = model.predict(x_learning)
    y_test_predict = model.predict(x_test)

    table_data_sklearn["Fold" + str(fold_step)] = [
        mean_squared_error(y_learning, y_learning_predict, squared=True),
        mean_squared_error(y_test, y_test_predict, squared=True),
        mean_squared_error(y_learning, y_learning_predict, squared=False),
        mean_squared_error(y_test, y_test_predict, squared=False),
        r2_score(y_learning, y_learning_predict),
        r2_score(y_test, y_test_predict),
    ]

    fold_step += 1

table_data_frame = pandas.DataFrame(data=table_data)
table_data_frame["E"] = table_data_frame.drop(
    columns=table_data_frame.columns[0], axis=1
).mean(axis=1)
table_data_frame["STD"] = table_data_frame.drop(
    columns=table_data_frame.columns[0], axis=1
).std(axis=1)

table_data_frame_sklearn = pandas.DataFrame(data=table_data_sklearn)
table_data_frame_sklearn["E"] = table_data_frame_sklearn.drop(
    columns=table_data_frame_sklearn.columns[0], axis=1
).mean(axis=1)
table_data_frame_sklearn["STD"] = table_data_frame_sklearn.drop(
    columns=table_data_frame_sklearn.columns[0], axis=1
).std(axis=1)

print(table_data_frame.head())
print("--------------------------------------------------------")
print(table_data_frame_sklearn.head())
