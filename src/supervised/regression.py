from typing import Callable

import numpy as np
from numpy._typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y, check_array


class Regression(RegressorMixin):

    def __init__(
        self,
        fit_intercept: bool,
        max_iter: int,
        learning_rate: float,
        regularization: Callable = None,
        tol: float = 1e-8,
    ):

        self.coef_ = None
        self.n_features = None
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.tol = tol

    def _xavier_initialization(self):

        return np.random.uniform(
            -1 / self.n_features, 1 / self.n_features, self.n_features
        )

    def fit(self, X: NDArray, y: NDArray):
        """

        :param X:
        :param y:
        :return:
        """

        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, self.n_features = X.shape

        coef_ = self._xavier_initialization()

        for _ in range(self.max_iter):
            yhat = X @ coef_
            residuals = yhat - y
            dldcoef = (
                X.T @ residuals / n_samples + self.regularization.grad(coef_)
                if self.regularization
                else 0
            )

            old_coef_ = coef_
            coef_ -= self.learning_rate * dldcoef

            if self._converged(old_coef_, coef_):
                print("CONVERGED!")
                break

        self.coef_ = coef_

    def predict(self, X: NDArray) -> NDArray:
        """

        :param X:
        :return:
        """
        check_array(X)
        return X @ self.coef_

    def _converged(self, param, new_param):

        return np.linalg.norm(new_param - param) < self.tol
