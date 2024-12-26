from abc import abstractmethod
from typing import Tuple

import numpy as np
from numpy import ndarray, dtype, bool
from numpy._typing import NDArray


class Loss:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: NDArray) -> float:
        pass

    @abstractmethod
    def grad(self, X: NDArray) -> NDArray:
        pass


class L1Loss(Loss):

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: NDArray) -> float:
        return self.alpha * np.linalg.norm(x, ord=1)

    def grad(self, x: NDArray) -> NDArray:
        return self.alpha * np.sign(x)


class L2Loss(Loss):
    def __init__(self, alpha: float = 1.0, squared: bool = False):
        """

        :param alpha:
        """
        super().__init__()
        self.alpha = alpha
        self.squared = squared

    def __call__(self, x: NDArray) -> float:
        """

        :param x:
        :return:
        """

        return self.alpha * (
            (x.T @ x).sum() if self.squared else np.linalg.norm(x, ord=2)
        )

    def grad(self, x: NDArray) -> NDArray:
        """

        :param x:
        :return:
        """

        return self.alpha * x if self.squared else self.alpha * x / self(x)


class ElasticNetLoss(Loss):
    def __init__(self, alpha: float = 1.0, ratio: float = 1):
        """

        :param alpha: float loss coefficient
        :param ratio: float ratio of the l2 loss to the l2 loss
        """

        super().__init__()
        self.alpha = alpha

        if ratio < 0:
            raise ValueError("ratio must be positive")
        elif ratio == 0:
            contributions = np.array([0, 1])
        elif ratio == float("inf"):
            contributions = np.array([1, 0])
        else:
            contributions = np.array([ratio, 1]) / (ratio + 1)

        self.ratio = ratio
        self.contributions = contributions

        self.l2_loss = L2Loss(alpha=alpha)
        self.l1_loss = L1Loss(alpha=alpha)

    def __call__(self, x: NDArray) -> float:

        return self.contributions @ np.array([self.l2_loss(x), self.l1_loss(x)])

    def grad(self, x: NDArray) -> NDArray:

        return self.contributions @ np.vstack(
            [self.l2_loss.grad(x), self.l1_loss.grad(x)]
        )
