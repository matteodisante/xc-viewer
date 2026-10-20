"""Settings and fitted parameters of the Gaussian phase segmenter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_GAUSSIAN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "segmentation_gaussian.yaml"
)

# The phases in the order of the sequence policy's cycle.
GAUSSIAN_STATES = ("transition", "search", "climb")


@dataclass(frozen=True)
class SequencePolicy:
    """Soft persistence and cyclic exit preference applied when decoding."""

    weight: float
    mean_dwell_s: float
    search_mean_dwell_s: float
    forward_probability: float
    allow_search_skip: bool

    def __post_init__(self) -> None:
        """Reject values that would make the decoding matrix ill-defined."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("sequence policy weight must lie in [0, 1]")
        if self.mean_dwell_s <= 0.0 or self.search_mean_dwell_s <= 0.0:
            raise ValueError("sequence policy dwell times must be positive")
        if not 0.5 < self.forward_probability < 1.0:
            raise ValueError("forward probability must lie in (0.5, 1)")


@dataclass(frozen=True)
class GaussianModel:
    """One discipline's fitted model; component ``i`` is ``state_names[i]``."""

    state_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and the stochasticity the decoder relies on."""
        k, d = self.means.shape
        if sorted(self.state_names) != sorted(GAUSSIAN_STATES) or k != 3:
            raise ValueError("state_names must name transition, search and climb")
        if self.covariances.shape != (k, d, d) or self.transition.shape != (k, k):
            raise ValueError("Gaussian model parameters are misshaped")
        if self.initial.shape != (k,) or self.scaler_mean.shape != (d,):
            raise ValueError("Gaussian model parameters are misshaped")
        if np.any(self.scaler_scale <= 0.0):
            raise ValueError("scaler_scale must be positive")
        for block in (self.transition, self.initial.reshape(1, -1)):
            if np.any(block < 0.0) or not np.allclose(block.sum(axis=1), 1.0):
                raise ValueError("initial and transition rows must be probabilities")


@dataclass(frozen=True)
class GaussianConfig:
    """Feature windows, decoding choices and one fitted model per discipline."""

    decision_step_s: float
    window_s: float
    max_native_dt_s: float
    min_horizontal_speed_mps: float
    marginalize_turn_coherence: bool
    policy: SequencePolicy
    models: dict[str, GaussianModel]

    def model_for(self, discipline: str) -> GaussianModel:
        """The model fitted for one discipline name (``Discipline.name``)."""
        try:
            return self.models[discipline]
        except KeyError as error:
            raise KeyError(f"no Gaussian model fitted for {discipline!r}") from error


def load_gaussian_config(path: str | Path | None = None) -> GaussianConfig:
    """Load and validate ``configs/segmentation_gaussian.yaml``, or ``path``."""
    import yaml

    source = Path(path) if path is not None else DEFAULT_GAUSSIAN_CONFIG_PATH
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    features, decoding = raw["features"], raw["decoding"]
    models = {
        str(name): GaussianModel(
            state_names=tuple(str(s) for s in block["state_names"]),
            scaler_mean=np.array(block["scaler_mean"], dtype=float),
            scaler_scale=np.array(block["scaler_scale"], dtype=float),
            initial=np.array(block["initial"], dtype=float),
            transition=np.array(block["transition_matrix"], dtype=float),
            means=np.array(block["means"], dtype=float),
            covariances=np.array(block["covariances"], dtype=float),
        )
        for name, block in raw["models"].items()
    }
    return GaussianConfig(
        decision_step_s=float(features["decision_step_s"]),
        window_s=float(features["window_s"]),
        max_native_dt_s=float(features["max_native_dt_s"]),
        min_horizontal_speed_mps=float(features["min_horizontal_speed_mps"]),
        marginalize_turn_coherence=bool(decoding["marginalize_turn_coherence"]),
        policy=SequencePolicy(**decoding["sequence_policy"]),
        models=models,
    )
