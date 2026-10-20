"""Decoding with a fitted Gaussian model: sequence policy, emissions, Viterbi."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from .config import GAUSSIAN_STATES, GaussianConfig, GaussianModel
from .features import FEATURE_COLUMNS


def decoding_matrix(model: GaussianModel, config: GaussianConfig) -> np.ndarray:
    """The fitted transition matrix blended with the configured sequence policy.

    Persistence moves towards ``exp(-step / dwell)``: for transition and climb only
    upwards, for search in either direction. Exits are blended towards the next phase
    of the cycle, or evenly from transition when search may be skipped.
    """
    policy = config.policy
    fitted = model.transition
    component = {name: i for i, name in enumerate(model.state_names)}
    out = np.zeros_like(fitted)
    for k, name in enumerate(GAUSSIAN_STATES):
        i = component[name]
        forward = component[GAUSSIAN_STATES[(k + 1) % len(GAUSSIAN_STATES)]]
        exits = [j for j in range(len(fitted)) if j != i]
        weights = fitted[i, exits]
        weights = weights / weights.sum() if weights.sum() else np.full(2, 0.5)
        if name == "transition" and policy.allow_search_skip:
            target = np.full(2, 0.5)
        else:
            p = policy.forward_probability
            target = np.array([p if j == forward else 1.0 - p for j in exits])
        weights = (1.0 - policy.weight) * weights + policy.weight * target
        stay = fitted[i, i]
        if name == "search":
            floor = np.exp(-config.decision_step_s / policy.search_mean_dwell_s)
            stay += policy.weight * (floor - stay)
        else:
            floor = np.exp(-config.decision_step_s / policy.mean_dwell_s)
            stay += policy.weight * max(0.0, floor - stay)
        # An absorbing fitted state keeps a possible exit.
        stay = min(stay, 1.0 - np.finfo(float).eps)
        out[i, i] = stay
        out[i, exits] = (1.0 - stay) * weights
    return out


def log_emissions(
    model: GaussianModel, config: GaussianConfig, features: np.ndarray
) -> np.ndarray:
    """Log density of each standardized feature row under each component, (n, k).

    With ``marginalize_turn_coherence`` the Gaussians are marginalized over coherence:
    its mean entry and covariance row and column are dropped.
    """
    keep = [
        i
        for i, name in enumerate(FEATURE_COLUMNS)
        if not (config.marginalize_turn_coherence and name == "turn_coherence")
    ]
    return np.column_stack(
        [
            multivariate_normal(mean[keep], cov[np.ix_(keep, keep)]).logpdf(
                features[:, keep]
            )
            for mean, cov in zip(model.means, model.covariances, strict=True)
        ]
    )


def viterbi(
    log_b: np.ndarray, log_initial: np.ndarray, log_transition: np.ndarray
) -> np.ndarray:
    """The most probable component path of one contiguous block of decisions."""
    n, k = log_b.shape
    score = log_initial + log_b[0]
    previous = np.zeros((n, k), dtype=np.int64)
    for step in range(1, n):
        candidates = score[:, None] + log_transition
        previous[step] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_b[step]
    path = np.empty(n, dtype=np.int64)
    path[-1] = score.argmax()
    for step in range(n - 1, 0, -1):
        path[step - 1] = previous[step, path[step]]
    return path
