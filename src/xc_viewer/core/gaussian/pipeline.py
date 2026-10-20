"""Labelling one cleaned flight on the decision grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GaussianConfig
from .features import FEATURE_COLUMNS, segment_features
from .model import decoding_matrix, log_emissions, viterbi

UNCLASSIFIED = "unclassified"
POINT_COLUMNS = [
    "segment_id",
    "t",
    "feature_edge",
    "quality_masked",
    *FEATURE_COLUMNS,
    "phase",
]


def segment_flight(
    fixes: pd.DataFrame, config: GaussianConfig, *, discipline: str
) -> pd.DataFrame:
    """Decision points of one cleaned flight, labelled by the discipline's model.

    Each preprocessing segment is decoded on its own, and each run of consecutive
    decisions with features is an independent Viterbi sequence: no transition crosses
    a segment boundary or an unavailable decision.

    Args:
        fixes: The retained fixes of one flight, with ``segment_id``.
        config: The segmenter settings and fitted models.
        discipline: The ``Discipline.name`` whose model applies.

    Returns:
        :data:`POINT_COLUMNS`, one row per decision time. Decisions without features
        are ``unclassified``; an ineligible segment contributes no rows.
    """
    model = config.model_for(discipline)
    log_transition = np.log(decoding_matrix(model, config))
    with np.errstate(divide="ignore"):
        log_initial = np.log(model.initial)
    names = np.asarray(model.state_names, dtype=object)
    frames = []
    for segment_id, segment in fixes.groupby("segment_id", sort=True):
        frame = segment_features(segment, config)
        if frame is None:
            continue
        frame.insert(0, "segment_id", segment_id)
        phase = np.full(len(frame), UNCLASSIFIED, dtype=object)
        valid = np.flatnonzero(~(frame["feature_edge"] | frame["quality_masked"]))
        for block in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
            if block.size == 0:
                continue
            raw = frame[FEATURE_COLUMNS].to_numpy(dtype=float)[block]
            standardized = (raw - model.scaler_mean) / model.scaler_scale
            log_b = log_emissions(model, config, standardized)
            phase[block] = names[viterbi(log_b, log_initial, log_transition)]
        frame["phase"] = phase
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=POINT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[POINT_COLUMNS]
