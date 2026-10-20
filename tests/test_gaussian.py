"""Tests for the Gaussian flight-phase segmenter (xc_viewer.core.gaussian)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xc_viewer.core.gaussian import load_gaussian_config, segment_flight
from xc_viewer.core.gaussian.features import window_integrals
from xc_viewer.core.gaussian.model import decoding_matrix

CONFIG = load_gaussian_config()

# Labels of the original implementation (hmmlearn and the saved model, with the same
# decoder settings) on `_synthetic_flight()`, as (phase, number of decisions) runs.
EXPECTED_RUNS = {
    "paragliders": [
        ("unclassified", 2),
        ("transition", 58),
        ("climb", 61),
        ("search", 29),
        ("transition", 59),
        ("unclassified", 1),
    ],
    "hang gliders": [
        ("unclassified", 2),
        ("transition", 57),
        ("search", 1),
        ("climb", 61),
        ("search", 30),
        ("transition", 58),
        ("unclassified", 1),
    ],
}


def _synthetic_flight() -> pd.DataFrame:
    """A 1-Hz glide, thermal, S-turn search and glide; then a 20-s segment."""
    legs = [  # duration [s], horizontal speed, vertical speed, turn rate [rad/s]
        (600, 11.0, -1.0, lambda t: 0.02 * np.sin(t / 40.0)),
        (600, 10.5, 1.8, lambda t: 0.3 + 0.0 * t),
        (300, 8.0, -0.3, lambda t: 0.12 * np.sign(np.sin(np.pi * t / 15.0))),
        (600, 11.0, -1.0, lambda t: 0.02 * np.cos(t / 50.0)),
    ]
    t = np.arange(sum(leg[0] for leg in legs), dtype=float)
    local = np.concatenate([np.arange(leg[0], dtype=float) for leg in legs])
    which = np.repeat(np.arange(len(legs)), [leg[0] for leg in legs])
    v_h = np.array([legs[i][1] for i in which])
    v_z = np.array([legs[i][2] for i in which])
    omega = np.array([legs[i][3](s) for i, s in zip(which, local, strict=True)])
    heading = np.cumsum(omega)
    v_e, v_n = v_h * np.cos(heading), v_h * np.sin(heading)
    flight = pd.DataFrame(
        {
            "segment_id": 0,
            "t": t,
            "E": np.cumsum(v_e),
            "N": np.cumsum(v_n),
            "z": 1500.0 + np.cumsum(v_z),
            "v_E": v_e,
            "v_N": v_n,
            "v_z": v_z,
            "a_E": -v_h * omega * np.sin(heading),
            "a_N": v_h * omega * np.cos(heading),
            "z_reconstructed": False,
            "z_derivative_reconstructed": False,
            "edge": False,
        }
    )
    slow = flight.iloc[:40].assign(segment_id=1, t=t[-1] + 20.0 * np.arange(1, 41))
    return pd.concat([flight, slow], ignore_index=True)


def _runs(phases: pd.Series) -> list[tuple[str, int]]:
    starts = phases.ne(phases.shift()).cumsum()
    return [(group.iloc[0], len(group)) for _, group in phases.groupby(starts)]


@pytest.mark.parametrize("discipline", ["paragliders", "hang gliders"])
def test_labels_match_the_original_implementation(discipline):
    points = segment_flight(_synthetic_flight(), CONFIG, discipline=discipline)
    assert _runs(points["phase"]) == EXPECTED_RUNS[discipline]


def test_slow_segments_have_no_decisions_and_masked_windows_no_label():
    flight = _synthetic_flight()
    flight.loc[700, "z_reconstructed"] = True
    points = segment_flight(flight, CONFIG, discipline="paragliders")
    assert set(points["segment_id"]) == {0}
    masked = points.loc[points["quality_masked"]]
    # Windows of 30 s centred every 10 s, plus the fixes bracketing their ends.
    assert masked["t"].tolist() == [690.0, 700.0, 710.0]
    assert masked["phase"].eq("unclassified").all()


def test_window_integrals_are_exact_for_piecewise_linear_values():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    values = np.array([0.0, 2.0, 2.0, 6.0])
    got = window_integrals(t, values, np.array([0.5, 0.0]), np.array([3.0, 4.0]))
    assert got == pytest.approx([0.75 + 2.0 + 3.0, 1.0 + 2.0 + 8.0])


def test_decoding_matrix_is_stochastic_and_zero_weight_keeps_the_fit():
    for model in CONFIG.models.values():
        matrix = decoding_matrix(model, CONFIG)
        assert matrix.sum(axis=1) == pytest.approx(np.ones(3))
        unchanged = replace(CONFIG, policy=replace(CONFIG.policy, weight=0.0))
        assert decoding_matrix(model, unchanged) == pytest.approx(model.transition)
