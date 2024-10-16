from unittest.mock import patch
import numpy as np

from machine_learning import (
    linear_regression, linear_regression_predict, mean_squared_error, 
    logistic_regression_gradient_descent, logistic_regression_predict,
    binarize, split_data, standardize, euclidean_distance,
    cosine_distance, knn
)

def test_binarize():
    assert binarize(["Chinstrap", "Adelie", "Chinstrap", "Adelie"]).tolist() == [1, 0, 1, 0]

def test_split_data_correct():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    x_train, x_test, y_train, y_test = split_data(x, y)
    assert np.allclose(x_train, np.array([[7, 8], [1, 2], [5, 6]]))
    assert np.allclose(x_test, np.array([[3, 4]]))
    assert np.allclose(y_train, np.array([1, 0, 0]))
    assert np.allclose(y_test, np.array([1]))

@patch("sklearn.model_selection.train_test_split")
def test_split_data_sklearn_used(train_test_split_patch):
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    split_data(x, y)
    assert train_test_split_patch.called_with(x, y, test_size=0.2, random_state=42, shuffle=True)

def test_standardize():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    X_test = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    X_train, X_test = standardize(X_train, X_test)
    
    # ensure the mean is 0 and std is 1 for training set.
    assert np.allclose(X_train.mean(axis=0), 0.0)
    assert np.allclose(X_train.std(axis=0), 1.0)

    # standardize all together.
    X = np.vstack([X_train, X_test])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    X_combined = np.vstack([X_train, X_test])
    assert not np.allclose(X_combined, X)

def test_euclidean_distance():
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    assert np.isclose(euclidean_distance(x1, x2), 5.196152422706632)

def test_cosine_distance():
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    assert np.isclose(cosine_distance(x1, x2), 0.025368153802923787)

@patch("sklearn.neighbors.KNeighborsClassifier")
def test_knn(knn_patch):
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    assert knn(x, y, np.array([2, 3]), euclidean_distance, 3) == 0
    assert knn(x, y, np.array([6, 7]), euclidean_distance, 3) == 1
    assert not knn_patch.called

    
@patch("sklearn.linear_model.LinearRegression")
def test_linear_regression(lin_reg_patch):
    train_x = [0.32084576,  1.30057122,  0.60076732,  1.02064966,  1.58049278, -0.65887969, -0.51891891,  0.60076732]
    train_x = np.array(train_x).reshape(-1, 2)
    train_y = np.array([0, 1, 0, 1])
    weights = linear_regression(train_x, train_y)
    assert np.allclose(weights, [0.76422991, -0.45091607, -0.07187847])
    assert not lin_reg_patch.called

@patch("sklearn.linear_model.LinearRegression")
def test_linear_regression_predict(lin_reg_patch):
    train_x = [0.32084576,  1.30057122,  0.60076732,  1.02064966,  1.58049278, -0.65887969, -0.51891891,  0.60076732]
    train_x = np.array(train_x).reshape(-1, 2)
    train_y = np.array([0, 1, 0, 1])
    weights = linear_regression(train_x, train_y)
    predictions = linear_regression_predict(train_x, weights)
    assert np.allclose(predictions, [0.52607233, 0.41997153, 0.09891958, 0.95503655])
    assert not lin_reg_patch.called

@patch("sklearn.metrics.mean_squared_error")
def test_mean_squared_error(mse_patch):
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 3, 3, 5])
    assert np.isclose(mean_squared_error(y_true, y_pred), 0.5)
    assert not mse_patch.called

@patch("sklearn.linear_model.LogisticRegression")
def test_logistic_regression(lr_patch):
    train_x = [0.32084576,  1.30057122,  0.60076732,  1.02064966,  1.58049278, -0.65887969, -0.51891891,  0.60076732]
    train_x = np.array(train_x).reshape(-1, 2)
    train_y = np.array([0, 1, 0, 1])
    weights = logistic_regression_gradient_descent(train_x, train_y)
    assert np.allclose(weights, [0.77362687, -1.97910505, 0.12705975])
    assert not lr_patch.called

def test_logistic_regression_predict():
    train_x = [0.32084576,  1.30057122,  0.60076732,  1.02064966,  1.58049278, -0.65887969, -0.51891891,  0.60076732]
    train_x = np.array(train_x).reshape(-1, 2)
    train_y = np.array([0, 1, 0, 1])
    weights = logistic_regression_gradient_descent(train_x, train_y)
    predictions = logistic_regression_predict(train_x, weights)
    assert np.allclose(predictions, [0.5753931, 0.42906394, 0.08031673, 0.86726101])