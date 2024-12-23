"""

"""
from typing import Union, Dict, Callable
import numpy as np
from numpy.typing import NDArray, ArrayLike

from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y, check_consistent_length, check_array

from src.utils.mlutils import euclidean_distance


class KNN(ClassifierMixin):

    def __init__(self, k:int=3, class_weight:Union[str, Dict]=None, f_dist:Callable=None):
        """
        K-Nearest Neighbors Classifier
        :param k: number of nearest neighbors
        """
        self.k = k
        self.class_weight = class_weight

        if f_dist is None:
            f_dist = euclidean_distance
        elif not isinstance(f_dist, Callable):
            raise TypeError("f_dist must be callable")
        self.f_dist = f_dist

    def fit(self, X:ArrayLike, y:ArrayLike, sample_weight:ArrayLike) -> None:
        """
        Fit K-Nearest Neighbors Classifier
        :param X: a nxp matrix of features
        :param y: a length n array of labels
        :param sample_weight: a length n array of sample weights
        :return: None
        """
        X, y = check_X_y(X, y)
        check_consistent_length(y, sample_weight)

        self.X_train=X
        self.y_train=y
        self.sample_weight=sample_weight


    def predict(self, X:ArrayLike):
        """
        Predict labels for X
        :param X: An nxp matrix of features
        :return:
        """

        X_checked = check_array(X)



