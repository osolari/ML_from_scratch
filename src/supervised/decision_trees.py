"""
Base Decision Tree Implementation
---------------------------------

This module implements a basic decision tree for classification or regression tasks. The decision tree splits
data recursively based on a feature and cutoff value that maximize information gain or minimize loss.

Theory:
-------
A decision tree is a hierarchical structure where:
- Internal nodes represent decisions based on feature values.
- Leaf nodes represent output values or classes.

Key Parameters:
- **max_depth**: Limits the depth of the tree to prevent overfitting.
- **min_sample_split**: Minimum number of samples required to split a node.
- **min_purity**: A threshold for stopping splits when node purity is sufficiently high.

Key Methods:
- **fit**: Builds the decision tree using the training data.
- **predict**: Traverses the tree to make predictions for new data.

Advantages:
-----------
- **Interpretability**: Decision trees are easy to visualize and understand.
- **Non-parametric**: No assumptions about data distribution.
- **Feature Importance**: Highlights which features are most important for predictions.

Limitations:
------------
- **Overfitting**: Can create overly complex models.
- **Bias Toward Multi-Valued Features**: May favor features with more unique values.
- **Instability**: Small changes in data can cause major tree structure changes.

Use Cases:
----------
- Classification tasks (e.g., predicting categories).
- Regression tasks (e.g., predicting continuous values).
- Feature selection and data exploration.
"""

from typing import Optional, Dict, Union, Callable
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from collections import Counter

from src.utils.datautils import bisect_array_on_feature
from src.utils.mlutils import (
    compute_information_gain,
    compute_entropy,
    compute_majority_class,
    compute_variance_reduction,
)


class DecisionNode:
    """
    Represents a single node in the decision tree.

    Attributes
    ----------
    feature_idx : Optional[int]
        Index of the feature used for splitting. None for leaf nodes.
    cutoff : Optional[float]
        The cutoff value for the feature used for splitting. None for leaf nodes.
    val : Optional[Union[int, float, str]]
        Value stored in leaf nodes.
    left : Optional[DecisionNode]
        Left child node.
    right : Optional[DecisionNode]
        Right child node.
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        cutoff: Optional[float] = None,
        val: Optional[Union[int, float, str]] = None,
        left: Optional["DecisionNode"] = None,
        right: Optional["DecisionNode"] = None,
    ):
        self.feature_idx = feature_idx
        self.cutoff = cutoff
        self.val = val
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        """
        Checks if the current node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf, False otherwise.
        """
        return self.left is None and self.right is None


from typing import Optional, Callable, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from numpy.typing import NDArray


class DecisionNode:
    """
    Represents a node in the decision tree.

    Attributes
    ----------
    feature_idx : Optional[int]
        The index of the feature used to split at this node. None if it's a leaf node.
    cutoff : Optional[Union[int, float, str]]
        The cutoff value for the split. None if it's a leaf node.
    left : Optional[DecisionNode]
        The left child of the node. None if it's a leaf node.
    right : Optional[DecisionNode]
        The right child of the node. None if it's a leaf node.
    val : Optional[Union[int, float, str]]
        The value at the leaf node (if this node is a leaf).
    """

    def __init__(self, feature_idx=None, cutoff=None, left=None, right=None, val=None):
        self.feature_idx = feature_idx
        self.cutoff = cutoff
        self.left = left
        self.right = right
        self.val = val

    def is_leaf(self) -> bool:
        return self.val is not None


class _BaseDecisionTree(BaseEstimator):
    """
    Base class for decision tree implementations.

    Attributes
    ----------
    max_depth : float
        Maximum depth of the tree. Default is infinity.
    min_sample_split : int
        Minimum number of samples required to split an internal node.
    min_purity : float
        Minimum impurity gain required to make a split.
    loss : Callable, optional
        Loss function used to evaluate splits.
    root : Optional[DecisionNode]
        The root node of the decision tree.
    _compute_impurity : Optional[Callable]
        Function to compute the impurity gain.
    _compute_leaf_value : Optional[Callable]
        Function to compute the value at leaf nodes.
    """

    def __init__(
        self,
        max_depth: float = float("inf"),
        min_sample_split: int = 2,
        min_purity: float = 1e-7,
        loss: Optional[Callable] = None,
    ):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_purity = min_purity
        self.loss = loss

        self.root: Optional[DecisionNode] = None
        self._compute_impurity: Optional[Callable] = None
        self._compute_leaf_value: Optional[Callable] = None

    def fit(self, X: NDArray, y: NDArray) -> "_BaseDecisionTree":
        """
        Fit the decision tree to the given data.

        Parameters
        ----------
        X : NDArray
            Input features, shape (n_samples, n_features).
        y : NDArray
            Target values, shape (n_samples,).

        Returns
        -------
        _BaseDecisionTree
            The fitted decision tree.
        """
        X, y = check_X_y(X, y)

        # Build the tree
        self.root = self._fit(X, y)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict the target values for the input data.

        Parameters
        ----------
        X : NDArray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        NDArray
            Predicted target values, shape (n_samples,).
        """
        X = check_array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _fit(self, X: NDArray, y: NDArray, depth: int = 0) -> DecisionNode:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : NDArray
            Input features, shape (n_samples, n_features).
        y : NDArray
            Target values, shape (n_samples,).
        depth : int, optional
            Current depth of the tree, by default 0.

        Returns
        -------
        DecisionNode
            The root node of the (sub)tree.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        max_score = 0
        decision_rule = None

        # Stopping conditions
        if (
            n_samples < self.min_sample_split
            or depth >= self.max_depth
            or n_classes == 1
        ):
            val = self._compute_leaf_value(y)
            return DecisionNode(val=val)

        else:
            for feat_idx in range(n_features):
                Xi = X[:, feat_idx]
                cutoffs = np.unique(Xi)
                for cutoff in cutoffs:
                    mask = bisect_array_on_feature(
                        X=X, feature_idx=feat_idx, cutoff=cutoff, return_mask_only=True
                    )
                    yl, yr = y[mask], y[~mask]

                    if len(yl) and len(yr):
                        score = self._compute_impurity(y, yl, yr)
                        if score > max_score:
                            max_score = score
                            decision_rule = dict(
                                feature_idx=feat_idx, cutoff=cutoff, mask=mask
                            )

            if max_score > self.min_purity and decision_rule is not None:
                left_subtree = self._fit(
                    X[decision_rule["mask"]], y[decision_rule["mask"]], depth + 1
                )
                right_subtree = self._fit(
                    X[~decision_rule["mask"]], y[~decision_rule["mask"]], depth + 1
                )
                return DecisionNode(
                    feature_idx=decision_rule["feature_idx"],
                    cutoff=decision_rule["cutoff"],
                    left=left_subtree,
                    right=right_subtree,
                )

    @staticmethod
    def _traverse_tree(x: NDArray, node: DecisionNode) -> Union[int, float, str]:
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters
        ----------
        x : NDArray
            Input sample, shape (n_features,).
        node : DecisionNode
            The current node in the tree.

        Returns
        -------
        Union[int, float, str]
            Predicted value for the input sample.
        """
        if node.is_leaf():
            return node.val

        feature_val = x[node.feature_idx]
        subtree = node.right
        if isinstance(feature_val, (int, float)):
            if feature_val <= node.cutoff:
                subtree = node.left
        elif feature_val == node.cutoff:
            subtree = node.right

        return _BaseDecisionTree._traverse_tree(x, subtree)


class DecisionTreeClassifier(_BaseDecisionTree):
    """
    Decision Tree Classifier that extends the base decision tree implementation.

    This class implements a classification tree by using information gain as the impurity metric
    and the majority class of the target values as the leaf value.

    Methods
    -------
    fit(X, y)
        Fit the classification tree to the input data.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Fit the classification tree to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples,).

        Returns
        -------
        DecisionTreeClassifier
            The fitted classification tree.
        """
        # Assign the impurity computation and leaf value computation methods
        self._compute_impurity = compute_information_gain
        self._compute_leaf_value = compute_majority_class

        # Fit the tree using the base implementation
        super(DecisionTreeClassifier, self).fit(X, y)
        return self


class DecisionTreeRegressor(_BaseDecisionTree):
    """
    Decision Tree Regressor that extends the base decision tree implementation.

    This class implements a regression tree by using the mean squared error (MSE) as the impurity metric
    and the mean of the target values as the leaf value.

    Methods
    -------
    fit(X, y)
        Fit the regression tree to the input data.

    _compute_mean_y(y)
        Compute the mean of the target values for leaf nodes.
    """

    @staticmethod
    def _compute_mean_y(y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the mean of the target values along the first axis and return a scalar if the result is one-dimensional.

        This method calculates the mean of the input array and handles the case where the result is an array with a single
        element by converting it to a scalar.

        Parameters
        ----------
        y : np.ndarray
            The input dataset of target values. It should be a 1D or 2D array.

        Returns
        -------
        Union[float, np.ndarray]
            The mean of the target values. If the result is one-dimensional, a scalar value is returned; otherwise,
            the mean array is returned.

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array([1, 2, 3])
        >>> DecisionTreeRegressor._compute_mean_y(y)
        2.0

        >>> y = np.array([[1, 2], [3, 4], [5, 6]])
        >>> DecisionTreeRegressor._compute_mean_y(y)
        array([3., 4.])

        Notes
        -----
        - If `y` is a 2D array, the mean is computed along `axis=0` (row-wise).
        - If the computed mean is an array with one element, it is returned as a scalar.
        """
        # Compute the mean of the input array along the first axis
        _y_mean = y.mean(axis=0)

        # Return the mean as a scalar if it has a single element
        return _y_mean if np.ndim(_y_mean) > 0 else float(_y_mean)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """
        Fit the regression tree to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples,).

        Returns
        -------
        DecisionTreeRegressor
            The fitted regression tree.
        """
        # Assign the impurity computation and leaf value computation methods
        self._compute_impurity = compute_variance_reduction
        self._compute_leaf_value = DecisionTreeRegressor._compute_mean_y

        # Fit the tree using the base implementation
        super(DecisionTreeRegressor, self).fit(X, y)
        return self


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# def plot_classification_results(X, y_true, y_pred, title="Classification Results"):
#     """
#     Plot the classification results for 2D data.
#
#     Parameters
#     ----------
#     X : ndarray
#         Feature matrix of shape (n_samples, 2).
#     y_true : ndarray
#         True labels of shape (n_samples,).
#     y_pred : ndarray
#         Predicted labels of shape (n_samples,).
#     title : str, optional
#         Title for the plot, by default "Classification Results".
#     """
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         X[:, 0],
#         X[:, 1],
#         c=y_true,
#         cmap="coolwarm",
#         label="True Labels",
#         alpha=0.6,
#         edgecolor="k",
#     )
#     plt.scatter(
#         X[:, 0], X[:, 1], c=y_pred, cmap="cool", marker="x", label="Predicted Labels"
#     )
#     plt.title(title)
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.legend()
#     plt.show()

#
# def test_decision_tree():
#     """
#     Test the DecisionTree class on various datasets and plot results.
#
#     Tests
#     -----
#     - Fits a DecisionTree classifier on the breast cancer dataset.
#     - Calculates accuracy on the test set.
#     - Validates model performance and functionality.
#     - Plots classification results for synthetic 2D datasets.
#
#     Additional Tests
#     ----------------
#     - Tests edge cases with small datasets.
#     - Verifies behavior with single-class targets.
#     - Confirms handling of large datasets.
#     """
#     # Load dataset
#     data = datasets.load_breast_cancer()
#     X, y = data.data, data.target
#
#     # Split dataset into train and test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#
#     # Initialize and train the decision tree
#     clf = DecisionTree(max_depth=10)
#     clf.fit(X_train, y_train)
#
#     # Predict and calculate accuracy
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy on test set: {acc:.4f}")
#
#     # Additional test case: small dataset
#     X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
#     y_small = np.array([0, 1, 0, 1])
#     clf_small = DecisionTree(max_depth=2)
#     clf_small.fit(X_small, y_small)
#     y_small_pred = clf_small.predict(X_small)
#     small_acc = accuracy_score(y_small, y_small_pred)
#     print(f"Accuracy on small dataset: {small_acc:.4f}")
#
#     # Additional test case: single-class target
#     X_single = np.array([[1, 2], [3, 4], [5, 6]])
#     y_single = np.array([1, 1, 1])
#     clf_single = DecisionTree(max_depth=2)
#     clf_single.fit(X_single, y_single)
#     y_single_pred = clf_single.predict(X_single)
#     print(f"Predictions for single-class dataset: {y_single_pred}")
#
#     # Additional test case: large dataset
#     X_large = np.random.rand(1000, 10)
#     y_large = np.random.randint(0, 2, 1000)
#     X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
#         X_large, y_large, test_size=0.3, random_state=42
#     )
#     clf_large = DecisionTree(max_depth=15)
#     clf_large.fit(X_train_large, y_train_large)
#     y_large_pred = clf_large.predict(X_test_large)
#     large_acc = accuracy_score(y_test_large, y_large_pred)
#     print(f"Accuracy on large dataset: {large_acc:.4f}")
#
#     # Additional test case: 2D synthetic dataset
#     X_synthetic, y_synthetic = datasets.make_classification(
#         n_samples=200,
#         n_features=2,
#         n_informative=2,
#         n_redundant=0,
#         n_clusters_per_class=1,
#         random_state=42,
#     )
#     X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = (
#         train_test_split(X_synthetic, y_synthetic, test_size=0.3, random_state=42)
#     )
#     clf_synthetic = DecisionTree(max_depth=5)
#     clf_synthetic.fit(X_train_synthetic, y_train_synthetic)
#     y_pred_synthetic = clf_synthetic.predict(X_test_synthetic)
#     synthetic_acc = accuracy_score(y_test_synthetic, y_pred_synthetic)
#     print(f"Accuracy on synthetic 2D dataset: {synthetic_acc:.4f}")
#
#     # Plot classification results
#     plot_classification_results(X_test_synthetic, y_test_synthetic, y_pred_synthetic)
#
#
# if __name__ == "__main__":
#     test_decision_tree()