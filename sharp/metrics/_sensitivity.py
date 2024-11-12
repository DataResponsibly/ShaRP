import numpy as np
from sklearn.utils import check_random_state
from sharp.utils import scores_to_ordering
from ._base import (
    _find_neighbors,
    _find_all_neighbors,
    _get_importance_mask,
    _ROW_WISE_MEASURES,
)


def _pairwise_outcome_sensitivity(
    row_data1,
    row_data2,
    row_cont1,
    row_cont2,
    score_func,
    original_scores,
    threshold,
    n_tests,
    stds,
    rng,
):
    # Find the most important features
    masks = [
        _get_importance_mask(row_cont, threshold) for row_cont in [row_cont1, row_cont2]
    ]
    mask1, mask2 = masks

    # Apply perturbation to the most important features
    perturbations = [rng.normal(loc=0, scale=stds) for _ in range(n_tests)]
    rows_pert1 = np.array([row_data1 + pert * mask1 for pert in perturbations])
    rows_pert2 = np.array([row_data2 + pert * mask2 for pert in perturbations])

    # Compute the prediction gap fidelity
    pert_ranks = []
    for rows_pert in [rows_pert1, rows_pert2]:
        rows_pert_score = score_func(rows_pert)
        rows_pert_rank = np.array(
            [
                scores_to_ordering(np.append(original_scores, score))[-1]
                for score in rows_pert_score
            ]
        )
        pert_ranks.append(rows_pert_rank)

    return np.abs(pert_ranks[0] - pert_ranks[1]).mean()


def row_based_outcome_sensitivity(
    original_data,
    rankings,
    original_scores,
    score_func,
    contributions,
    row_idx,
    threshold=0.8,
    n_neighbors=10,
    n_tests=10,
    std_multiplier=0.2,
    random_state=None,
):
    rng = check_random_state(random_state)

    # Select close neighbors
    data_neighbors, cont_neighbors = _find_neighbors(
        original_data, rankings, contributions, row_idx, n_neighbors
    )
    stds = np.std(original_data, axis=0) * std_multiplier

    # Compute distance between the target point and its neighbors
    scores = []
    for i in range(len(data_neighbors)):
        row_cont1 = np.array(contributions)[row_idx]
        row_cont2 = cont_neighbors[i]
        score = _pairwise_outcome_sensitivity(
            np.array(original_data)[row_idx],
            data_neighbors[i],
            row_cont1,
            row_cont2,
            score_func,
            original_scores,
            threshold,
            n_tests,
            stds,
            rng,
        )
        scores.append(score)

    return np.mean(scores)


def outcome_sensitivity(
    original_data,
    score_func,
    contributions,
    threshold=0.8,
    n_neighbors=10,
    n_tests=10,
    std_multiplier=0.2,
    aggregate_results=False,
    random_state=None,
):
    original_scores = score_func(original_data)
    rankings = scores_to_ordering(original_scores)

    sensitivities = np.vectorize(
        lambda row_idx: row_based_outcome_sensitivity(
            original_data,
            rankings,
            original_scores,
            score_func,
            contributions,
            row_idx,
            threshold,
            n_neighbors,
            n_tests,
            std_multiplier,
            random_state,
        )
    )(np.arange(len(original_data)))
    if aggregate_results:
        return (
            np.mean(sensitivities),
            np.std(sensitivities) / np.sqrt(sensitivities.size),
        )
    else:
        return sensitivities


def row_wise_explanation_sensitivity(
    original_data,
    contributions,
    row_idx,
    rankings,
    n_neighbors=10,
    agg_type="mean",
    measure="kendall",
    similar_outcome=True,
    **kwargs,
):
    row_cont = np.array(contributions)[row_idx]

    # Select close neighbors
    data_neighbors, cont_neighbors = _find_neighbors(
        original_data, rankings, contributions, row_idx, n_neighbors, similar_outcome
    )

    # Compute Kendall tau distance between the target point and its neighbors
    distances = np.apply_along_axis(
        lambda row: _ROW_WISE_MEASURES[measure](row, row_cont, **kwargs),
        1,
        cont_neighbors,
    )

    if agg_type == "max":
        return np.max(distances)
    elif agg_type == "mean":
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


def row_wise_explanation_sensitivity_all_neighbors(
    original_data,
    contributions,
    row_idx,
    rankings,
    threshold=0.1,
    measure="kendall",
    **kwargs,
):
    row_cont = np.array(contributions)[row_idx]
    row_rank = np.array(rankings)[row_idx]

    # Select all neighbors that are under the threshold
    data_neighbors, cont_neighbors, rank_neighbors, feature_distances = (
        _find_all_neighbors(original_data, rankings, contributions, row_idx, threshold)
    )

    # Compute the measure (e.g. Kendall tau) distance between the target point and its
    # neighbors
    measure_distances = np.apply_along_axis(
        lambda row: _ROW_WISE_MEASURES[measure](row, row_cont, **kwargs),
        1,
        cont_neighbors,
    )

    return measure_distances, row_rank - rank_neighbors, feature_distances


# Calculates the explanation sensitivity of every row of original data and its
# closest neighbors,
def explanation_sensitivity(
    original_data,
    contributions,
    rankings,
    n_neighbors=10,
    agg_type="mean",
    measure="kendall",
    similar_outcome=True,
    **kwargs,
):
    sensitivities = np.vectorize(
        lambda row_idx: row_wise_explanation_sensitivity(
            original_data,
            contributions,
            row_idx,
            rankings,
            n_neighbors,
            agg_type,
            measure,
            similar_outcome,
            **kwargs,
        )
    )(np.arange(len(original_data)))
    return np.mean(sensitivities), np.std(sensitivities) / np.sqrt(sensitivities.size)


def explanation_sensitivity_all_neighbors(
    original_data, contributions, rankings, measure="kendall", threshold=0.1, **kwargs
):
    return lambda row_idx: row_wise_explanation_sensitivity_all_neighbors(
        original_data, contributions, row_idx, rankings, threshold, measure, **kwargs
    )
