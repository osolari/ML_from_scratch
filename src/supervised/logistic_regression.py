import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils import check_X_y, check_array

from src.test.utils.mlutils import add_intercept_column


class LogisticRegression(LinearClassifierMixin, BaseEstimator):

    def __init__(
        self,
        fit_intercept=True,
        C: float = 1.0,
        learning_rate: float = 0.001,
        max_iter=100,
        tol: float = 1e-4,
    ):
        self.num_features = None
        self._coef = None
        self.fit_intercept = fit_intercept
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _xavier_initialization(self):

        return np.random.uniform(
            -1 / self.num_features, 1 / self.num_features, self.num_features
        )

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_intercept_column(X)

        self.num_features = X.shape[1]
        coef = self._xavier_initialization()

        for _ in range(self.max_iter):
            yhat = self.sigmoid(X @ coef)
            residuals = y - yhat
            dldcoef = residuals @ X

            old_coef = coef
            coef -= self.learning_rate * dldcoef

            if self._converged(old_coef, coef):
                break
        self._coef = coef

        return self

    def predict(self, X):
        self._is_fitted()
        X = check_array(X)
        if self.fit_intercept:
            X = add_intercept_column(X)
        yhat = self.sigmoid(X @ self._coef)

        return yhat

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def _converged(self, param, new_param):
        return np.linalg.norm(new_param - param) < self.tol

    def _is_fitted(self):
        if self._coef is None:
            raise ValueError("LinearRegression is not fitted")
