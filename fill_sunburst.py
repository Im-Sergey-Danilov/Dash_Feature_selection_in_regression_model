from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 1130

import numpy as np

random_state = RANDOM_STATE
def make_dataset(name, random_state):
    np.random.seed(1130)
    if name == "INDUS":
        X = load_boston().data[:, 2].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'NOX':
        X = load_boston().data[:, 4].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'LSTAT':
        X = load_boston().data[:, -1].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'RM':
        X = load_boston().data[:, 5].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'AGE':
        X = load_boston().data[:, 6].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'TAX':
        X = load_boston().data[:, -4].reshape(-1, 1)
        y = load_boston().target
        return X, y


def fill_sunburst_scores_errors(regression_model_name, feature_names, error_bool, score_bool):
    x1, y1 = make_dataset(name=feature_names, random_state=RANDOM_STATE)
    x_trains, x_tests, y_trains, y_tests = \
        train_test_split(x1, y1, test_size=100, random_state=RANDOM_STATE)
    if regression_model_name == 'lasso':
        models = Lasso(alpha=0.1, normalize=True)
    elif regression_model_name == 'ridge':
        models = Ridge(alpha=0.1, normalize=True)
    elif regression_model_name == 'elastic_net':
        models = ElasticNet(alpha=0.1, normalize=True)
    else:
        models = LinearRegression(normalize=True)
    if regression_model_name =='elastic_net':
        poly = PolynomialFeatures(degree=8)
    else:
        poly = PolynomialFeatures(degree=1)
    x_train_poly = poly.fit_transform(x_trains)
    x_test_poly = poly.fit_transform(x_tests)
    models.fit(x_train_poly, y_trains)
    test_scores = models.score(x_train_poly, y_trains)
    test_errors = mean_squared_error(y_tests, models.predict(x_test_poly))
    if score_bool:
        return test_errors
    elif error_bool:
        return test_scores


print(fill_sunburst_scores_errors(regression_model_name='elastic', feature_names='LSTAT', error_bool=True, score_bool=False)
      )
