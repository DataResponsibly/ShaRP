from itertools import product, combinations
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from sharp.utils import scores_to_ordering
import pandas as pd


# Not reviewed
# Returns neighbors that are either close or far ranking wise
# AND subselects the top n neighbors in terms of feature similarity
def _find_neighbors(
    original_data, rankings, contributions, row_idx, n_neighbors, similar_outcome=True
):
    row_data = np.array(original_data)[row_idx]
    row_rank = np.array(rankings)[row_idx]
    min_ranking = max(0, row_rank - n_neighbors)
    max_ranking = min(row_rank + n_neighbors, max(rankings))

    # Select neighbors that are close ranking-wise
    if similar_outcome:
        mask = (
            (rankings >= min_ranking)
            & (rankings <= max_ranking)
            & (rankings != row_rank)
        )
    else:  # Select neighbors that are far ranking wise
        mask = (rankings < min_ranking) | (rankings > max_ranking)
    data_neighbors = np.array(original_data)[mask]
    cont_neighbors = np.array(contributions)[mask]

    # Select neighbors that are close distance-wise
    distances = np.apply_along_axis(
        lambda row: euclidean(row, row_data), 1, data_neighbors
    )
    neighbors_idx = np.argpartition(distances, -n_neighbors)[-n_neighbors:]
    data_neighbors = data_neighbors[neighbors_idx]
    cont_neighbors = cont_neighbors[neighbors_idx]
    return data_neighbors, cont_neighbors


# Not reviewed
# Returns all neighbors that are similar feature wise
# The Euclidean distance between items has to be under the threshold
def _find_all_neighbors(
    original_data, rankings, contributions, row_idx, threshold=None
):
    row_data = np.array(original_data)[row_idx]

    data_neighbors = np.array(original_data)
    cont_neighbors = np.array(contributions)
    rank_neighbors = np.array(rankings)

    # Select neighbors that are close distance-wise
    distances = np.apply_along_axis(
        lambda row: euclidean(row, row_data), 1, data_neighbors
    )
    # Apply threshold
    if threshold is not None:
        neighbors_idx = np.where(distances <= threshold)[0]
        data_neighbors = data_neighbors[neighbors_idx]
        cont_neighbors = cont_neighbors[neighbors_idx]
        rank_neighbors = rank_neighbors[neighbors_idx]
        return (
            data_neighbors,
            cont_neighbors,
            rank_neighbors,
            distances[distances <= threshold],
        )
    # Or return distances from all items
    return (
        data_neighbors,
        cont_neighbors,
        rank_neighbors,
        distances,
    )


# Reviewed
def _get_importance_mask(row_cont, threshold):
    if threshold >= 1:
        # Calculate order of absolute contributions
        row_abs = np.abs(row_cont)
        # Find n=threshold largest items
        res = sorted(row_abs.index.values, key=lambda sub: row_abs[sub])[-threshold:]
        # Set mask
        mask = pd.Series(
            data=[True if i in res else False for i in row_cont.index.values],
            index=row_cont.index.values,
        )
    else:
        # Calculate cumulative absolute contribution order
        total_contribution = np.sum(np.abs(row_cont))
        order = np.argsort(np.abs(row_cont))
        cumulative_cont = np.cumsum(np.abs(row_cont)[order]) / total_contribution
        # Find elements withe the smallest contribution
        # (to meet the threshold it's easier to do the reverse operation)
        mask = (cumulative_cont < 1 - threshold)[order]
        # Reverse array
        mask = ~mask

    # Check whether ties exist
    possible_configs = [mask.copy()]
    tie_values = [
        (idx_old, cont)
        for idx_old, cont in row_cont[mask].items()
        if cont in row_cont[~mask].values
    ]
    # Make all possible sets of ties
    for idx_old, tie_val in tie_values:
        idx_new = row_cont[row_cont == tie_val].index.values
        # Exclude all selected indexes that have the same value
        idx_new = list(set(idx_new).difference(mask[mask].index.values))
        for idx in idx_new:
            new_mask = mask.copy()
            new_mask[idx_old] = False
            new_mask[idx] = True
            possible_configs.append(new_mask)

    return possible_configs


# Reviewed
def jaccard_similarity(a, b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(set(a)) + len(set(b))) - intersection
    return float(intersection) / union


# Reviewed
def kendall_similarity(a, b):
    normalizer = (len(a) * (len(a) - 1)) / 2
    idx_pair = list(combinations(range(len(a)), 2))
    val_pair_a = [(a[i], a[j]) for i, j in idx_pair if a[i] != a[j]]
    val_pair_b = [(b[i], b[j]) for i, j in idx_pair if b[i] != b[j]]
    inversions = 0
    for (val11, val12), (val21, val22) in zip(val_pair_a, val_pair_b):
        if ((val11 > val12) and (val21 < val22)) or (
            (val11 < val12) and (val21 > val22)
        ):
            inversions = inversions + 1
    kt = 1 - (2 * inversions) / normalizer
    return (kt + 1) / 2


# Reviewed
def row_wise_kendall(results1, results2):
    """
    Calculate the row-wise Kendall's similarity between two sets of
    contributions.

    Parameters
    ----------
    results1 : array-like
        The first set of contributions.
    results2 : array-like
        The second set of contributions.

    Returns
    -------
    float
        The row-wise Kendall's similarity.

    Notes
    -----
    The row-wise Kendall's similarity measures the similarity
    between two sets of rankings. It takes into account ties and is robust to
    outliers.

    """
    results = [results1, results2]

    # Check for ties
    ranks = []
    for result in results:
        rank = scores_to_ordering(result, direction=1)
        # Correct rank values to reflect ties
        for val in result:
            mask = result == val
            if mask.sum() > 1:
                rank[mask] = rank[mask].max()
        ranks.append(rank)

    row_sensitivity = kendall_similarity(*ranks)
    return row_sensitivity


# Reviewed
def row_wise_jaccard(results1, results2, n_features):
    """
    Calculate the row-wise Jaccard similarity between two sets of results.

    Parameters
    ----------
    results1 : numpy.ndarray
        The first set of results. It should be a 2-dimensional array with shape
        (n_samples, n_features).
    results2 : numpy.ndarray
        The second set of results. It should be a 2-dimensional array with shape
        (n_samples, n_features).
    n_features : int, float or None, default=0.8
        The number of top features to consider. If None, all features are
        considered. If an integer value is provided, only the top n_features
        features are considered. If n_features < 1, the most
        important features are determined based on their contribution to the
        total score (as a percentage of the total contribution in absolute
        values).

    Returns
    -------
    float
        The row-wise Jaccard similarity between the two sets of results.

    Notes
    -----
    The row-wise Jaccard similarity is calculated by first converting the
    results into rankings using the `scores_to_ordering` function. Then, the top
    n_features features are selected based on the rankings. Finally, the Jaccard
    similarity is calculated between the selected features for each row.

    If n_features is less than 1, the most important features are determined
    based on their contribution to the total score.  The cumulative contribution
    of each feature is calculated and the features are selected until the
    cumulative contribution exceeds 1 - n_features.

    Examples
    --------
    >>> results1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> results2 = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    >>> n_features = 2
    >>> row_wise_jaccard(results1, results2, n_features)
    """

    if n_features is None:
        n_features = results1.shape[1]

    masks1 = _get_importance_mask(results1, n_features)
    masks2 = _get_importance_mask(results2, n_features)

    row_similarities = []
    for mask1, mask2 in product(masks1, masks2):
        top_idx1 = mask1[mask1].index.values
        top_idx2 = mask2[mask2].index.values
        row_similarity = jaccard_similarity(top_idx1, top_idx2)
        row_similarities.append(row_similarity)

    return max(row_similarities)


# Reviewed
def row_wise_euclidean(results1, results2, normalization=True):
    if normalization:
        # Make vectors into unit vectors
        v1 = normalize([results1])[0]
        v2 = normalize([results2])[0]
        return euclidean(v1, v2) / 2
    else:
        return euclidean(results1, results2)


# Reviewed
def euclidean_agreement(results1, results2, normalization):
    """
    Calculate the Euclidean agreement between two sets of contributions across a
    dataset. Results are normalized, 0 means most dis-similar and 1 means most
    similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of contributions results.
    results2 : pandas.DataFrame
        The second set of contributions results.

    Returns
    -------
    pandas.Series
        A pandas Series containing the Euclidean agreement values for each pair
        of contributions vectors in `results1` and `results2`. The values are
        normalized between 0 and 1, where 0 means most similar and 1 means most
        dissimilar.

    Notes
    -----
    The Euclidean agreement is calculated by comparing each pair of contributions
    vectors in `results1` and `results2` using the Euclidean distance.
    """
    return results1.reset_index(drop=True).apply(
        lambda row: 1 - row_wise_euclidean(row, results2.iloc[row.name], normalization),
        axis=1,
    )


# Reviewed
def kendall_agreement(results1, results2):
    """
    Calculate the Kendall agreement between two sets of contributions across a
    dataset. Results are normalized, 0 means most dis-similar and 1 means most
    similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of contributions results.
    results2 : pandas.DataFrame
        The second set of contributions results.

    Returns
    -------
    pandas.Series
        A pandas Series containing the Kendall agreement values for each pair
        of contributions vectors in `results1` and `results2`. The values are
        normalized between 0 and 1, where 0 means most dis-similar and 1 means most
        similar.

    Notes
    -----
    The Kendall agreement is calculated by comparing each pair of contributions
    vectors in `results1` and `results2` using the Kendall's tau correlation
    coefficient. The agreement is then averaged across all pairs of rankings.
    """
    return results1.reset_index(drop=True).apply(
        lambda row: row_wise_kendall(row, results2.iloc[row.name]), axis=1
    )


# Reviewed
def jaccard_agreement(results1, results2, n_features=0.8):
    """
    Calculate the Jaccard similarity between two sets of results. Results are
    normalized, 0 means most dis-similar and 1 means most similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of results.
    results2 : pandas.DataFrame
        The second set of results.
    n_features : int, float or None, default=0.8
        The number of top features to consider. If None, all features are
        considered. If an integer value is provided, only the top n_features
        features are considered. If n_features < 1, the most
        important features are determined based on their contribution to the
        total score (as a percentage of the total contribution in absolute
        values).

    Returns
    -------
    pandas.Series
        The Jaccard agreement between each pair of results.

    Notes
    -----
    The Jaccard agreement is a measure of similarity between two sets of
    results. It is calculated as the average Jaccard similarity coefficient
    between each pair of results.
    """
    if n_features is None:
        n_features = results1.shape[1]

    return results1.reset_index(drop=True).apply(
        lambda row: row_wise_jaccard(row, results2.iloc[row.name], n_features),
        axis=1,
    )


_ROW_WISE_MEASURES = {
    "euclidean": row_wise_euclidean,
    "jaccard": row_wise_jaccard,
    "kendall": row_wise_kendall,
}

_MEASURES = {
    "euclidean": euclidean_agreement,
    "jaccard": jaccard_agreement,
    "kendall": kendall_agreement,
}
