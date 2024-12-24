import numpy as np
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array

from numpy.typing import NDArray, ArrayLike

from src.utils.mlutils import matrix_euclidean_distance
import seaborn as sns
import matplotlib.pyplot as plt


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters=2, max_iter=100, tol=0.0001, random_state=None):

        self._labels = None
        self._centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        """

        :param X:
        :return:
        """
        X = check_array(X)

        centroids = self._init_centroids_from_random_samples(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)

            old_centroids = centroids
            centroids = self._compute_centroids(X, labels)

            if self._converged(old_centroids, centroids):
                print("CONVERGED!")
                break

        self._centroids = centroids
        self._labels = self._assign_clusters(X, centroids)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """

        :param X:
        :return:
        """

        self._is_fitted()
        X = check_array(X)

        return self._assign_clusters(X, self._centroids)

    def _init_centroids_from_random_samples(self, X: NDArray):
        """

        :param X:
        :return:
        """
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)

        return X[idx]

    def _assign_clusters(self, X, centroids):
        """

        :param X:
        :param centroids:
        :return:
        """

        assert len(centroids) == self.n_clusters

        return matrix_euclidean_distance(X, centroids).argmin(axis=1).reshape([-1, 1])

    @staticmethod
    def _compute_centroids(X, labels):
        """

        :param X:
        :param labels:
        :return:
        """

        encoder = OneHotEncoder(sparse_output=False)
        labels = encoder.fit_transform(labels)
        sums = labels.sum(axis=0)
        sums[sums == 0] = 1
        labels /= sums

        return labels.T @ X

    @staticmethod
    def _converged(old_centroids: NDArray, new_centroids: NDArray):
        """
        :param old_centroids:
        :param new_centroids:
        :return:
        """

        return not (old_centroids - new_centroids).any()

    def _is_fitted(self):
        """

        :return:
        """

        if self._centroids is None:
            raise ValueError("KMeans is not fitted")

    @staticmethod
    def plot_2d(ax: plt.Axes, X: NDArray, **kwargs):
        """

        :param ax:
        :param X:
        :param kwargs:
        :return:
        """

        if len(X.shape) < 2 or X.shape[1] != 2:
            raise ValueError(
                "plot_2d only supports 2-dimensional data for plotting purposes. You can perform "
                "dimensionality reduction or just input the first two features "
            )

        kmeans = KMeans(**kwargs)
        kmeans.fit(X)

        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans._labels.flatten(), ax=ax)

        return ax
