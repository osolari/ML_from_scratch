"""

"""

from collections import Counter
from typing import Union, Dict

import numpy as np
from numpy.typing import NDArray, ArrayLike

from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y, check_consistent_length, check_array

from src.utils.mlutils import matrix_euclidean_distance


class KNN(ClassifierMixin):

    def __init__(self, n_neighbors: int = 3):
        """
        K-Nearest Neighbors Classifier
        :param n_neighbors: number of nearest neighbors
        """

        self.n_neighbors = n_neighbors

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Fit K-Nearest Neighbors Classifier
        :param X: a nxp matrix of features
        :param y: a length n array of labels
        :param sample_weight: a length n array of sample weights
        :return: None
        """
        X, y = check_X_y(X, y)

        self.X_train = X
        self.y_train = y

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict labels for X
        :param X: An nxp matrix of features
        :return:
        """

        X_checked = check_array(X)
        dist = matrix_euclidean_distance(self.X_train, X_checked)
        assert dist.shape == (
            self.X_train.shape[0],
            X_checked.shape[0],
        ), f"Distance matrix must be of the shape {(self.X_train.shape[0], X_checked.shape[0])}"

        labels = self.infer_labels(dist)

        return labels

    def infer_labels(self, dist: NDArray) -> NDArray:
        """
        Infer labels from a distance matrix
        :param dist: n1xn2 matrix where dist[i,j] is the distance between X[i] and X[j]
        :return: labels
        """
        dist = dist.argsort(axis=0)
        dist = self.y_train[dist][: self.n_neighbors, :]
        # labels = np.array([Counter(d).most_common(1)[0][0] for d in dist.T])
        labels = np.array(
            [
                np.bincount(dist.astype(int)[:, i], minlength=self.n_neighbors)
                for i in range(dist.shape[1])
            ]
        ).argmax(axis=1)

        return labels
