"""Cluster bootstrap over tasks (protocol §4.5, §10: never report a metric
without a cluster bootstrap CI over tasks; steps are not independent).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_bootstrap_ci(
    task_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Resample whole tasks with replacement, recompute metric_fn on the
    pooled steps of the resampled tasks. Returns point estimate + CI.
    """
    rng = np.random.default_rng(seed)
    unique_tasks = np.unique(task_ids)
    n_tasks = len(unique_tasks)

    task_to_idx: dict = {}
    for t in unique_tasks:
        task_to_idx[t] = np.where(task_ids == t)[0]

    point = metric_fn(y_true, y_score)

    boot_stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sampled_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        idx = np.concatenate([task_to_idx[t] for t in sampled_tasks])
        try:
            boot_stats[b] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            # degenerate resample (e.g. only one class present)
            boot_stats[b] = np.nan

    valid = boot_stats[~np.isnan(boot_stats)]
    lo = float(np.percentile(valid, 100 * (alpha / 2)))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return {
        "point": float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_resamples_valid": int(len(valid)),
        "n_tasks": int(n_tasks),
    }


def cis_non_overlapping(ci_a: dict, ci_b: dict) -> bool:
    return ci_a["ci_hi"] < ci_b["ci_lo"] or ci_b["ci_hi"] < ci_a["ci_lo"]
