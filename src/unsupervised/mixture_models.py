import logging
from types import NoneType

import numpy as np
from sklearn.base import DensityMixin, BaseEstimator
from sklearn.utils import check_array

from src.utils.dist import MultiVariateGaussian

logging.basicConfig(format="[%(asctime)s]-[%(name)-1s-%(levelname)2s]: %(message)s")
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

"""
https://github.com/zf109/algorithm_practice/blob/master/gaussian_mixture_model/gaussian_mixture_model/gaussian_mixture_model.pdf
"""


class _BaseMixtureModel:
    """
    Base class for mixture models.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    n_init : int
        Number of initializations to perform.
    init_params : str
        Method for initializing parameters ('kmeans', 'random').
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress messages.
    """

    def __init__(
        self,
        n_components,
        tol,
        max_iter,
        n_init,
        init_params,
        random_state,
        verbose,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose


class GMM(_BaseMixtureModel):

    def __init__(
        self,
        n_components=1,
        tol=1e-3,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=None,
        verbose=False,
    ):

        super(GMM, self).__init__(
            n_components, tol, max_iter, n_init, init_params, random_state, verbose
        )
        self._mixtures = None
        self._means = None
        self._covariances = None
        self.num_features = None

    @staticmethod
    def _compute_components_loglikelihood(x, means, covariances, seed):

        llhds = [
            MultiVariateGaussian(
                mean=means[i], covariance=covariances[i], seed=seed
            ).loglikelihood(x)
            for i in range(len(means))
        ]

        return np.array(llhds)

    @staticmethod
    def _compute_components_likelihood(x, means, covariances, seed):
        lhds = [
            MultiVariateGaussian(
                mean=means[i], covariance=covariances[i], seed=seed
            ).likelihood(x)
            for i in range(len(means))
        ]

        return np.array(lhds)

    @staticmethod
    def _compute_data_likelihood(
        X,
        means,
        covariances,
        mixtures,
        reduction: [str, NoneType] = "sum",
        seed=None,
    ):

        lhd_matrix = np.vstack(
            [
                GMM._compute_components_likelihood(x, means, covariances, seed=seed)
                * mixtures
                for x in X
            ]
        )

        if reduction == "sum":
            lhd_matrix = lhd_matrix.sum(axis=1)
        elif reduction == "probs":
            lhd_matrix /= lhd_matrix.sum(axis=1, keepdims=True)

        return lhd_matrix

    def _expectation(self, X, mixtures, means, covariances):

        prob_matrix = self._compute_data_likelihood(
            X,
            means,
            covariances,
            mixtures,
            reduction="probs",
        )

        return prob_matrix

    def _maximization(self, X, prob_matrix):

        mixtures = prob_matrix.mean(axis=0)
        means = prob_matrix.T @ X / prob_matrix.sum(axis=0)

        covariances = np.stack(
            [
                GMM._compute_component_covariance(X, means[k], prob_matrix[:, k])
                for k in range(self.n_components)
            ]
        )

        return means, covariances, mixtures

    @staticmethod
    def _compute_component_covariance(X, means, prob_vector):

        X -= means
        return (prob_vector[:, np.newaxis] * X).T @ X / prob_vector.sum(axis=0)

    def fit(self, X, y=None):

        X = check_array(X)
        num_samples, self.num_features = X.shape

        means, covariances, mixtures = self._initialize_parameters()

        for i in range(self.max_iter):

            exp_out = self._expectation(X, mixtures, means, covariances)

            old_mixtures = mixtures

            means, covariances, mixtures = self._maximization()

            if self.is_converged(mixtures, old_mixtures):
                _logger.info("Converged")
                break

        self._means, self._covariances, self._mixtures = means, covariances, mixtures

    def _initialize_parameters(self):

        np.random.seed(self.random_state)
        means = np.random.randn(self.n_components, self.num_features)
        covariances = np.stack(
            [np.eye(self.num_features) for _ in range(self.n_components)]
        )
        mixtures = np.ones(self.n_components) / self.n_components

        return means, covariances, mixtures

    def is_converged(self, mixtures, old_mixtures):

        return sum(abs(mixtures - old_mixtures)) < self.tol

    def predict_proba(self, X):

        X = check_array(X)

        self.is_fitted()

        probs = self._expectation(X, self._mixtures, self._means, self._covariances)
        return probs

    def predict(self, X):

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)

    def is_fitted(self):

        if self._mixtures is None:
            raise ValueError(f"Model is not fitted!")
