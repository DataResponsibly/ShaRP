import numpy as np
from sharp.metrics import outcome_fidelity


def test_outcome_fidelity_no_target_pairs_rank_true():
    contributions = np.array([[0.1, 0.2], [0.3, 0.4]])
    target = np.array([0.5, 0.6])
    avg_target = 0.55
    target_max = 1
    result = outcome_fidelity(contributions, target, avg_target, target_max, rank=True)
    expected = (
        1
        - np.mean(np.abs(target - (avg_target - contributions.sum(axis=1))))
        / target_max
    )
    assert np.isclose(result, expected)


def test_outcome_fidelity_no_target_pairs_rank_false():
    contributions = np.array([[0.1, 0.2], [0.3, 0.4]])
    target = np.array([0.5, 0.6])
    avg_target = 0.55
    target_max = 1
    result = outcome_fidelity(contributions, target, avg_target, target_max, rank=False)
    expected = np.mean(
        1 - np.abs(target - (avg_target + contributions.sum(axis=1))) / target_max
    )
    assert np.isclose(result, expected)


def test_outcome_fidelity_with_target_pairs_rank_true():
    contributions = np.array([[0.1, 0.2], [0.3, 0.4]])
    target = np.array([0.5, 0.6])
    avg_target = 0.55
    target_max = 1
    target_pairs = np.array([0.4, 0.7])
    result = outcome_fidelity(
        contributions, target, avg_target, target_max, target_pairs, rank=True
    )
    better_than = target < target_pairs
    est_better_than = contributions.sum(axis=1) > 0
    expected = (better_than == est_better_than).mean()
    assert np.isclose(result, expected)


def test_outcome_fidelity_with_target_pairs_rank_false():
    contributions = np.array([[0.1, 0.2], [0.3, 0.4]])
    target = np.array([0.5, 0.6])
    avg_target = 0.55
    target_max = 1
    target_pairs = np.array([0.4, 0.7])
    result = outcome_fidelity(
        contributions, target, avg_target, target_max, target_pairs, rank=False
    )
    better_than = target > target_pairs
    est_better_than = contributions.sum(axis=1) > 0
    expected = (better_than == est_better_than).mean()
    assert np.isclose(result, expected)
