from typing import Union

import numpy as np
from numpy._typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y, check_array

from src.test.utils.mlutils import add_intercept_column
from src.utils.losses import Loss, L1Loss, L2Loss, ElasticNetLoss


class _RegressionBase(RegressorMixin):

    def __init__(
        self,
        fit_intercept: bool,
        max_iter: int,
        learning_rate: float,
        regularization: Union[Loss, None] = None,
        tol: float = 1e-8,
    ):
        """

        :param fit_intercept:
        :param max_iter:
        :param learning_rate:
        :param regularization:
        :param tol:
        """

        self._coef = None
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
            X = add_intercept_column(X)
        n_samples, self.n_features = X.shape

        coef = self._xavier_initialization()

        for _ in range(self.max_iter):
            yhat = X @ coef
            residuals = yhat - y
            dldcoef = (
                X.T @ residuals / n_samples + self.regularization.grad(coef)
                if self.regularization
                else 0
            )

            old_coef_ = coef
            coef -= self.learning_rate * dldcoef

            if self._converged(old_coef_, coef):
                print("CONVERGED!")
                break

        self._coef = coef

    def predict(self, X: NDArray) -> NDArray:
        """

        :param X:
        :return:
        """
        X = check_array(X)
        if self.fit_intercept:
            X = add_intercept_column(X)
        return X @ self._coef

    def _converged(self, param, new_param):

        return np.linalg.norm(new_param - param) < self.tol

    def _is_fitted(self):

        if self._coef is None:
            raise ValueError("LinearRegression is not fitted")


class LinearRegression(_RegressionBase):

    def __init__(
        self,
        fit_intercept: bool = True,
        closed_form: bool = True,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """

        :param fit_intercept:
        :param closed_form:
        :param learning_rate:
        :param max_iter:
        """
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=None,
            tol=tol,
        )
        self._intercept = None
        self._coefs = None
        self.closed_form = closed_form

    def fit(self, X: NDArray, y: NDArray):
        """

        :param X:
        :param y:
        :return:
        """
        if self.closed_form:
            X, y = check_X_y(X, y)
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            n_samples, self.n_features = X.shape
            XtX = X.T @ X
            U, S, Vt = np.linalg.svd(XtX)
            XtXinv = U @ np.diag(1.0 / S) @ Vt
            self._coef = XtXinv @ (X.T @ y)
        else:
            super().fit(X, y)


class Lasso(_RegressionBase):
    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):

        self.regularization = L1Loss(alpha=alpha)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )


class Ridge(_RegressionBase):
    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):

        self.regularization = L2Loss(alpha=alpha)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )


class ElasticNet(_RegressionBase):
    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        ratio: float = 1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):

        self.regularization = ElasticNetLoss(alpha=alpha, ratio=ratio)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )
