"""Typed configuration and fitted parameters for the Vilpellet phase segmenter.

Everything the decoder needs is data: the feature windows, the straightness threshold,
the eligibility gate and the three fitted parameter blocks all come from
``configs/segmentation_vilpellet.yaml``.  Nothing is refitted here, so the loader's job
is to validate that the file still describes a usable model rather than to choose
anything on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import numpy as np

DEFAULT_VILPELLET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "segmentation_vilpellet.yaml"
)

# The behavioural vocabulary, in the order the Chapter 4 segmenter also uses.  It is a
# set, not an ordering of HMM components: `VilpelletConfig.state_names` carries that.
VILPELLET_STATES = ("transition", "search", "climb")

# The eight values the three binary features can take, enumerated exactly as the
# reference implementation enumerates them.  Row k of this array is the observation
# whose probability is entry k of each emission vector.
FEATURE_VALUES = np.array(
    [[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=np.int64
)

# The state names of the reference implementation's own `regime_map` line, which
# transposes the first and third components relative to every other artefact of that
# package.  Retained so the transposition can be reproduced deliberately.
LITERAL_MINIMAL_SCRIPT_STATE_NAMES = ("transition", "search", "climb")


@dataclass(frozen=True)
class FeatureParams:
    """Window sizes of the binary feature construction, counted in fixes."""

    position_smoothing_fixes: int
    turn_smoothing_fixes: int
    persistence_fixes: int
    beta_persistence: float

    def __post_init__(self) -> None:
        """Reject windows that cannot produce the three indicators."""
        windows = (
            self.position_smoothing_fixes,
            self.turn_smoothing_fixes,
            self.persistence_fixes,
        )
        if any(int(value) != value or value < 2 for value in windows):
            raise ValueError("Vilpellet feature windows must be integers of at least 2")
        beta = self.beta_persistence
        if not isfinite(beta) or not 0.0 < beta <= 1.0:
            raise ValueError("beta_persistence must lie in (0, 1]")


@dataclass(frozen=True)
class Eligibility:
    """The guards the reference scripts apply before decoding a track."""

    max_mean_dt_s: float
    min_fixes: int
    require_unique_times: bool = True
    require_increasing_times: bool = True

    def __post_init__(self) -> None:
        """Reject a gate that would admit cadences the sample windows cannot serve."""
        if not isfinite(self.max_mean_dt_s) or self.max_mean_dt_s <= 0.0:
            raise ValueError("max_mean_dt_s must be finite and positive")
        if self.min_fixes < 3:
            raise ValueError("min_fixes must leave at least one interior fix")


@dataclass(frozen=True)
class Recommendations:
    """The author's own usage advice, expressed as optional post-processing.

    Each is inactive by default.  They select which decoded runs an analysis keeps and
    are therefore kept separate from the segmentation itself, which always labels every
    eligible fix.
    """

    drop_tail_s: float = 0.0
    min_run_s: float = 0.0
    search_requires_following_climb: bool = False

    def __post_init__(self) -> None:
        """Reject negative durations, which would silently extend a track."""
        if not isfinite(self.drop_tail_s) or self.drop_tail_s < 0.0:
            raise ValueError("drop_tail_s must be finite and non-negative")
        if not isfinite(self.min_run_s) or self.min_run_s < 0.0:
            raise ValueError("min_run_s must be finite and non-negative")

    @property
    def active(self) -> bool:
        """Whether any recommendation would change the retained runs."""
        return bool(
            self.drop_tail_s > 0.0
            or self.min_run_s > 0.0
            or self.search_requires_following_climb
        )


@dataclass(frozen=True)
class InputPolicy:
    """Which trajectory the features are built from, and how it is pre-filtered."""

    source: str = "cleaned"
    pre_smooth: bool = True

    def __post_init__(self) -> None:
        """Reject an input source this package has no adapter for."""
        if self.source not in {"cleaned", "raw_gnss"}:
            raise ValueError("Vilpellet input source must be 'cleaned' or 'raw_gnss'")


@dataclass(frozen=True)
class HMMParameters:
    """The fitted three-state model, exactly as published in the reference code.

    Attributes:
        emission: Shape ``(3, 8)``.  Row ``i`` is the categorical law of component
            ``i`` over :data:`FEATURE_VALUES`.
        transition: Shape ``(3, 3)``, row-stochastic.
        initial: Shape ``(3,)``.
    """

    emission: np.ndarray
    transition: np.ndarray
    initial: np.ndarray

    def __post_init__(self) -> None:
        """Check the shapes and the stochasticity the decoder relies on."""
        if self.emission.shape != (3, len(FEATURE_VALUES)):
            raise ValueError("Vilpellet emissions must be a 3x8 array")
        if self.transition.shape != (3, 3) or self.initial.shape != (3,):
            raise ValueError("Vilpellet transition/initial parameters are misshaped")
        for name, block in (
            ("emission", self.emission),
            ("transition", self.transition),
            ("initial", self.initial.reshape(1, -1)),
        ):
            if not np.isfinite(block).all() or (block < 0).any():
                raise ValueError(f"Vilpellet {name} parameters must be finite and >= 0")
            if not np.allclose(block.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"Vilpellet {name} rows must sum to one")


@dataclass(frozen=True)
class VilpelletConfig:
    """All fixed choices of the transcribed segmenter.

    Attributes:
        features: Window sizes of the binary feature construction.
        alpha_straight_rad: Straightness threshold per discipline name.
        majority_fixes: Width of the rolling majority vote applied after Viterbi.
        eligibility: The cadence and length gate.
        state_names: Behavioural name of HMM component ``i`` at position ``i``.
        parameters: The fitted model.
        recommendations: Optional run selection advised by the author.
        input_policy: Which trajectory the features are built from.
    """

    features: FeatureParams
    alpha_straight_rad: dict[str, float]
    majority_fixes: int
    eligibility: Eligibility
    state_names: tuple[str, ...]
    parameters: HMMParameters
    recommendations: Recommendations = Recommendations()
    input_policy: InputPolicy = InputPolicy()

    def __post_init__(self) -> None:
        """Check the naming and the threshold table the decoder indexes into."""
        if self.majority_fixes < 1:
            raise ValueError("the majority window must cover at least one fix")
        if len(self.state_names) != 3 or set(self.state_names) != set(VILPELLET_STATES):
            raise ValueError(
                "state_names must name transition, search and climb exactly once"
            )
        if not self.alpha_straight_rad:
            raise ValueError("at least one straightness threshold must be configured")
        for discipline, alpha in self.alpha_straight_rad.items():
            if not isfinite(alpha) or alpha <= 0.0:
                raise ValueError(f"{discipline}: alpha_straight_rad must be positive")

    def alpha_for(self, discipline: str) -> float:
        """The straightness threshold fitted for one discipline.

        Args:
            discipline: A key of ``alpha_straight_rad``, i.e. a
                :attr:`xc_viewer.core.disciplines.Discipline.name`.

        Returns:
            The threshold in radians per fix.

        Raises:
            KeyError: If no threshold was inferred for that discipline.
        """
        try:
            return float(self.alpha_straight_rad[discipline])
        except KeyError as error:
            known = ", ".join(sorted(self.alpha_straight_rad))
            raise KeyError(
                f"no Vilpellet straightness threshold for {discipline!r}; have {known}"
            ) from error

    def state_name(self, component: int) -> str:
        """The behavioural name of one HMM component."""
        return self.state_names[int(component)]


def load_vilpellet_config(path: str | Path | None = None) -> VilpelletConfig:
    """Load the transcribed segmenter's protocol and fitted parameters from YAML.

    Args:
        path: Optional replacement configuration file.

    Returns:
        A validated immutable configuration.

    Raises:
        ValueError: If the file describes a model the decoder cannot run.
    """
    import yaml

    source = Path(path) if path is not None else DEFAULT_VILPELLET_CONFIG_PATH
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))

    emissions = raw["parameters"]["emission"]
    names = tuple(str(name) for name in raw["state_names"])
    if bool(raw.get("literal_minimal_script", False)):
        names = LITERAL_MINIMAL_SCRIPT_STATE_NAMES
    missing = set(VILPELLET_STATES).difference(emissions)
    if missing:
        raise ValueError(f"missing Vilpellet emission vectors: {sorted(missing)}")
    parameters = HMMParameters(
        # Row i must be the emission law of component i, and `state_names` is what says
        # which behaviour component i carries.  Indexing the YAML block by name through
        # that list is what keeps the two blocks from drifting apart.
        emission=np.array([emissions[name] for name in names], dtype=float),
        transition=np.array(raw["parameters"]["transition_matrix"], dtype=float),
        initial=np.array(raw["parameters"]["initial"], dtype=float),
    )
    return VilpelletConfig(
        features=FeatureParams(**raw["features"]),
        alpha_straight_rad={
            str(key): float(value) for key, value in raw["alpha_straight_rad"].items()
        },
        majority_fixes=int(raw["decoding"]["majority_fixes"]),
        eligibility=Eligibility(**raw["eligibility"]),
        state_names=names,
        parameters=parameters,
        recommendations=Recommendations(**raw.get("recommendations", {})),
        input_policy=InputPolicy(**raw.get("input", {})),
    )


def with_recommendations(
    config: VilpelletConfig,
    *,
    drop_tail_s: float | None = None,
    min_run_s: float | None = None,
    search_requires_following_climb: bool | None = None,
) -> VilpelletConfig:
    """Return ``config`` with some of the author's usage advice switched on.

    Args:
        config: The loaded configuration.
        drop_tail_s: Seconds to discard from the end of each flight.
        min_run_s: Shortest decoded run an analysis keeps.
        search_requires_following_climb: Keep a search run only when a climb follows.

    Returns:
        A new configuration; the original is unchanged.
    """
    current = config.recommendations
    advice = Recommendations(
        drop_tail_s=current.drop_tail_s if drop_tail_s is None else drop_tail_s,
        min_run_s=current.min_run_s if min_run_s is None else min_run_s,
        search_requires_following_climb=(
            current.search_requires_following_climb
            if search_requires_following_climb is None
            else search_requires_following_climb
        ),
    )
    return replace(config, recommendations=advice)
