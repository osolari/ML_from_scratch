import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_consistent_length


def euclidean_distance(point1:ArrayLike, point2:ArrayLike)-> float:
    """
    Computes the Euclidean distance between two points.
    :param point1: a length n vector
    :param point2: a length n vector
    :return: float
    """

    point1 = np.array(point1)
    point2 = np.array(point2)

    check_consistent_length(point1, point2)

    return np.sqrt(sum((point1 - point2)**2))

def matrix_euclidean_matrix(X1:ArrayLike, X2:ArrayLike) -> NDArray:
    """
    Computes the Euclidean distance matrix between two matrices.
    :param X1: n1xp matrix
    :param X2: n2xp matrix
    :return: a n1xn2 matrix
    """

    X1 = np.array(X1)
    X2 = np.array(X2)

    if X1.shape[1] != X2.shape[1]:
        raise ValueError('X1 and X2 must have the same number of columns')

    return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=-1))

