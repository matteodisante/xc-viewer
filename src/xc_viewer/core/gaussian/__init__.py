"""The Gaussian flight-phase segmenter of the thesis experiments.

Four continuous features (mean vertical speed, mean horizontal speed, mean absolute
turn rate and turn coherence over 30-s windows) on a 10-s decision grid; a three-state
model with full-covariance Gaussian emissions, fitted per discipline on the cleaned
FFVL archive; Viterbi decoding of the marginal over the first three features, with a
soft sequence preference. The fitted parameters live in
``configs/segmentation_gaussian.yaml``; nothing is refitted here.
"""

from .config import GAUSSIAN_STATES, GaussianConfig, load_gaussian_config
from .pipeline import UNCLASSIFIED, segment_flight

__all__ = [
    "GAUSSIAN_STATES",
    "UNCLASSIFIED",
    "GaussianConfig",
    "load_gaussian_config",
    "segment_flight",
]
