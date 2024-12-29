import numpy as np
from typing import Tuple, Union, Optional


def bisect_array_on_feature(
    X: np.ndarray,
    feature_idx: int,
    cutoff: Union[float, int, str],
    return_mask_only: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Split a NumPy array into two subsets based on a condition applied to a specific feature column.

    The function divides the input array `X` into two subsets using a condition defined by the `cutoff` value
    applied to the feature at index `feature_idx`. Optionally, it can return only the mask indicating the split.

    Parameters
    ----------
    X : np.ndarray
        The input 2D array to split. Each row corresponds to an observation, and each column corresponds to a feature.
    feature_idx : int
        The index of the feature column to apply the cutoff condition.
    cutoff : Union[float, int, str]
        The cutoff value used to split the array. For numeric types (float or int), the condition is "greater than
        or equal to". For non-numeric types (e.g., strings), the condition is equality.
    return_mask_only : bool, optional
        If `True`, only the mask indicating which rows satisfy the condition is returned. Default is `False`.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
        If `return_mask_only` is `True`, returns a 1D boolean array (mask) indicating which rows satisfy the condition.
        Otherwise, returns a tuple containing:
            - The subset of rows satisfying the condition.
            - The subset of rows not satisfying the condition.
            - `None` (as a placeholder for future extension).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> bisect_array_on_feature(X, feature_idx=0, cutoff=3)
    (array([[3, 4],
           [5, 6]]), array([[1, 2]]), None)

    >>> bisect_array_on_feature(X, feature_idx=1, cutoff=4, return_mask_only=True)
    array([False,  True,  True])

    >>> X = np.array([['apple', 'red'], ['banana', 'yellow'], ['apple', 'green']])
    >>> bisect_array_on_feature(X, feature_idx=0, cutoff='apple')
    (array([['apple', 'red'],
           ['apple', 'green']], dtype='<U6'),
     array([['banana', 'yellow']], dtype='<U6'), None)

    Notes
    -----
    - The input array `X` is assumed to have consistent data types along each column.
    - The function supports both numeric and string cutoffs. Ensure the data in the feature column matches the cutoff type.
    """

    # Determine the condition function based on the cutoff type
    if isinstance(cutoff, (float, int)):
        fn = lambda x: x >= cutoff  # Numeric cutoff: greater than or equal
    else:
        fn = lambda x: x == cutoff  # Non-numeric cutoff: equality

    # Apply the condition to split the array
    mask = fn(X[:, feature_idx])

    # Return the two subsets and optionally the mask
    if return_mask_only:
        return mask
    else:
        return X[mask], X[~mask], None
