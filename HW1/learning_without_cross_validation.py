from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from work_with_data import get_normalize_data
from work_with_neural_network import evaluate_metrics, get_linear_regression

data = get_normalize_data("car_price.csv")

y = data["price"]
x = data.drop(columns=["price"])
x_learning_p, x_test_p, y_learning_p, y_test_p = train_test_split(
    x, y, test_size=0.3, random_state=42
)
x_learning = x_learning_p.to_numpy()
x_test = x_test_p.to_numpy()
y_learning = y_learning_p.to_numpy()
y_test = y_test_p.to_numpy()

weights = get_linear_regression(x_learning, y_learning)
learning_mse, learning_rmse, learning_r_sq = evaluate_metrics(
    x_learning, y_learning, weights
)
test_mse, test_rmse, test_r_sq = evaluate_metrics(x_test, y_test, weights)

model = Ridge()
model.fit(x_learning, y_learning)

y_learning_predict = model.predict(x_learning)
y_test_predict = model.predict(x_test)

print("--------------------------------------------------------")
print("Mean Square Error (learning): ", learning_mse)
print("Root Mean Square Error (learning): ", learning_rmse)
print("R^2 (learning): ", learning_r_sq)

print("--------------------------------------------------------")
print("Mean Square Error (test): ", test_mse)
print("Root Mean Square Error (test): ", test_rmse)
print("R^2 (test): ", test_r_sq)

print("--------------------------------------------------------")
print(
    "Mean Square Error (learning, sklearn): ",
    mean_squared_error(y_learning, y_learning_predict, squared=True),
)
print(
    "Root Mean Square Error (learning, sklearn): ",
    mean_squared_error(y_learning, y_learning_predict, squared=False),
)
print("R^2 (learning, sklearn): ", r2_score(y_learning, y_learning_predict))

print("--------------------------------------------------------")
print(
    "Mean Square Error (test, sklearn): ",
    mean_squared_error(y_test, y_test_predict, squared=True),
)
print(
    "Root Mean Square Error (test, sklearn): ",
    mean_squared_error(y_test, y_test_predict, squared=False),
)
print("R^2 (test, sklearn): ", r2_score(y_test, y_test_predict))
