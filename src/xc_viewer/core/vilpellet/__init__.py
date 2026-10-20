"""Vilpellet's binary-feature flight-phase segmenter, ported from the authors' code.

This package is a port, not a second attempt at the problem.  It reproduces the
segmentation of Vilpellet et al. (2026) as their code performs it: three binary
features per fix, a three-state hidden Markov model whose emissions are a categorical
law over the eight possible feature triples, Viterbi decoding under parameters fitted
once on 2019 Alpine paraglider flights, and a rolling majority vote over the decoded
path.  No parameter is re-estimated here.

Typical use::

    from xc_viewer.core.vilpellet import (
        load_vilpellet_config, segment_flight,
    )

    config = load_vilpellet_config()
    track = segment_flight(result.fixes, config, discipline="paragliders")
    climb = track.fixes.loc[track.fixes["phase"] == "climb"]
"""

from .config import (
    FEATURE_VALUES,
    VILPELLET_STATES,
    HMMParameters,
    VilpelletConfig,
    load_vilpellet_config,
    with_recommendations,
)
from .features import (
    OBSERVATION_COLUMNS,
    build_observation_frame,
    observation_matrix,
)
from .model import emission_probabilities, rolling_majority, viterbi
from .pipeline import (
    PHASE_SEGMENT_COLUMNS,
    UNCLASSIFIED,
    VilpelletTrack,
    phase_runs,
    segment_eligibility,
    segment_flight,
    segment_track,
)

__all__ = [
    "FEATURE_VALUES",
    "OBSERVATION_COLUMNS",
    "PHASE_SEGMENT_COLUMNS",
    "UNCLASSIFIED",
    "VILPELLET_STATES",
    "HMMParameters",
    "VilpelletConfig",
    "VilpelletTrack",
    "build_observation_frame",
    "emission_probabilities",
    "load_vilpellet_config",
    "observation_matrix",
    "phase_runs",
    "rolling_majority",
    "segment_eligibility",
    "segment_flight",
    "segment_track",
    "viterbi",
    "with_recommendations",
]
