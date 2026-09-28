"""Split posterior rows into disjoint placement-training and scoring sets."""

from __future__ import annotations

import numpy as np

STATIC_POSTERIOR_ARRAYS = {"zones"}


def _take_posterior_rows(
    post: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Take `rows` from posterior arrays and preserve static metadata arrays."""
    return {k: v if k in STATIC_POSTERIOR_ARRAYS else v[rows] for k, v in post.items()}


def split_greedy_and_score_draws(
    post: dict[str, np.ndarray],
    requested_score: int,
    requested_pre: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int, np.ndarray, np.ndarray]:
    """Return randomized, disjoint draws for placement training and scenario scoring."""
    available = int(post["R0"].size)
    if requested_score <= 0:
        raise ValueError(f"requested_score must be positive, got {requested_score}")
    if requested_pre <= 0:
        raise ValueError(f"requested_pre must be positive, got {requested_pre}")

    required = requested_score + requested_pre
    if available < required:
        if requested_score == 1 and available >= 2:
            n_pre = available - 1
        else:
            raise ValueError(
                "scenario subsample is too small for disjoint placement training and "
                f"scoring: have {available} rows, need at least {required} "
                f"({requested_pre} greedy pre-pass + {requested_score} scoring). "
                f"Rerun extract_subsample.py with --n {required}."
            )
    else:
        n_pre = requested_pre

    n_score = min(requested_score, available - n_pre)
    rows = np.random.default_rng(seed).permutation(available)
    train_rows = rows[:n_pre]
    score_rows = rows[n_pre : n_pre + n_score]
    train = _take_posterior_rows(post, train_rows)
    score = _take_posterior_rows(post, score_rows)
    return train, score, n_pre, n_score, train_rows, score_rows
