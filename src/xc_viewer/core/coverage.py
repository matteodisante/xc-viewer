"""Explicit denominators and reasons for segmentation coverage."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def decision_reasons(points: pd.DataFrame) -> np.ndarray:
    """Partition decisions into classified, feature edge, quality mask, or other."""
    reasons = np.where(
        points["phase"].eq("unclassified"), "unavailable_features", "classified"
    )
    missing = points["phase"].eq("unclassified").to_numpy()
    for flag in ("quality_masked", "feature_edge"):
        if flag in points:
            reasons[missing & points[flag].fillna(False).to_numpy(dtype=bool)] = flag
    return reasons


def native_coverage_summary(fixes: pd.DataFrame) -> dict:
    """Count displayed cleaned fixes, and left-labelled edges within physical runs.

    Counts refer to fixes, not flights. The duration denominator
    excludes acquisition gaps; each native edge takes its starting fix's colour.
    """
    counts = fixes["phase_reason"].value_counts().to_dict()
    seconds: defaultdict[str, float] = defaultdict(float)
    for _, run in fixes.groupby("track_run", sort=False):
        dt = np.diff(run["t"].to_numpy(dtype=float))
        reasons = run["phase_reason"].to_numpy()[:-1]
        for reason in np.unique(reasons):
            seconds[str(reason)] += float(dt[reasons == reason].sum())
    n = len(fixes)
    total_s = sum(seconds.values())
    return {
        "n_cleaned_fixes": n,
        "unclassified_fix_percent": 100 * (n - counts.get("classified", 0)) / n
        if n
        else None,
        "fixes_by_reason": {str(k): int(v) for k, v in counts.items()},
        "cleaned_duration_s": total_s,
        "unclassified_duration_percent": 100
        * (total_s - seconds["classified"])
        / total_s
        if total_s
        else None,
        "seconds_by_reason": dict(seconds),
    }
