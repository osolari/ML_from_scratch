import numpy as np
from numpy._typing import NDArray
from sklearn.utils import check_X_y

from src.supervised.regression import Regression


class LinearRegressionGD(Regression):

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """

        :param fit_intercept:
        :param learning_rate:
        :param max_iter:
        """
        super().__init__()
        self._intercept = None
        self._coefs = None
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: NDArray, y: NDArray):
        """

        :param X:
        :param y:
        :return:
        """

        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        coefs = np.zeros(n_features)
        intercept = 0

        params = np.hstack([coefs, np.array(intercept)])

        for _ in range(self.max_iter):

            y_pred = X @ coefs + intercept
            residuals = y_pred - y

            dldcoef = X.T @ residuals / n_samples
            dldintercept = sum(residuals) / n_samples

            old_params = params

            coefs -= self.learning_rate * dldcoef
            intercept -= self.learning_rate * dldintercept

            params = np.hstack([coefs, np.array(intercept)])

            if self.converged(old_params, params):
                print("CONVERGED!")
                break

        self._coefs = coefs
        self._intercept = intercept

        return self

    def predict(self, X: NDArray) -> NDArray:
        """

        :param X:
        :return:
        """

        self._is_fitted()

        return X @ self._coefs + self._intercept

    def _is_fitted(self):
        """

        :return:
        """

        if self._coefs is None:
            raise ValueError("LinearRegression is not fitted")

    def converged(self, params, new_params):
        """

        :param params:
        :param new_params:
        :return:
        """

        return (
            np.sqrt(
                sum((params[0] - new_params[0]) ** 2) + (params[1] - new_params[1]) ** 2
            )
            < self.tol
        )
