"""
Utilities for reproducible dataset splitting in the ECG pipeline.

This module defines GroupKFold helpers used to create train/validation folds
on development records (`a/b/c`) while keeping the final `x` cohort untouched.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold


def _to_group_array(groups: Sequence[str] | np.ndarray) -> np.ndarray:
    """Normalize incoming group labels to a 1D NumPy array of strings."""
    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(f"`groups` must be 1D, got shape={arr.shape}")
    return arr.astype(str)


def iter_group_kfold_indices(
    groups: Sequence[str] | np.ndarray,
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) folds grouped by record id.

    Notes:
    - Groups must be record-level identifiers (e.g., 'a01', 'b03', 'c10').
    - Fold split is deterministic for a fixed ordering of `groups`.
    - GroupKFold guarantees no group overlap between train and validation.
    """
    if n_splits < 2:
        raise ValueError("`n_splits` must be >= 2")

    g = _to_group_array(groups)
    n_unique = len(np.unique(g))
    if n_unique < n_splits:
        raise ValueError(
            f"Not enough unique groups for n_splits={n_splits}. "
            f"Found unique groups={n_unique}."
        )

    # Dummy features/labels are unused by GroupKFold; only length and groups matter.
    dummy = np.zeros((len(g), 1), dtype=np.float32)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(dummy, dummy, groups=g):
        yield train_idx, val_idx


def summarize_fold_groups(
    groups: Sequence[str] | np.ndarray,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
) -> Tuple[List[str], List[str]]:
    """
    Return sorted unique group names for train and validation splits.

    Useful for logging and for validating no record leakage across folds.
    """
    g = _to_group_array(groups)
    train_groups = sorted(set(g[list(train_idx)].tolist()))
    val_groups = sorted(set(g[list(val_idx)].tolist()))
    return train_groups, val_groups
