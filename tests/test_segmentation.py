"""The saved segmentations: the refit loader, the agreement count and the switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xc_viewer import data, segmentation
from xc_viewer.core.disciplines import DATA_FOLDER_ENV, PARAGLIDERS


@pytest.fixture(autouse=True)
def _vilpellet_default():
    segmentation.use(segmentation.DEFAULT)
    yield
    segmentation.use(segmentation.DEFAULT)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A data folder with refit runs of two flights, one of them in two segments."""
    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path))
    monkeypatch.delenv(PARAGLIDERS.env, raising=False)
    folder = tmp_path / "paragliders/derived/segmentation/refit"
    (folder / "model").mkdir(parents=True)
    runs = pd.DataFrame(
        {
            "source": "paraglider",
            "flight_id": ["10", "10", "10", "10", "11"],
            # The saved segment_id is xc-dynamics's: it differs from the viewer's.
            "segment_id": [7, 7, 7, 8, 0],
            "phase": ["search", "climb", "transition", "climb", "climb"],
            "t_start": [-2.0, 10.0, 20.0, 40.0, 0.0],
            "t_end": [9.0, 19.0, 25.0, 49.0, 5.0],
        }
    )
    runs.to_parquet(folder / "phase_segments.parquet")
    pd.DataFrame(
        {
            "flight_id": ["10", "11", "12"],
            "status": ["classified", "classified", "cadence_above_gate"],
        }
    ).to_parquet(folder / "phase_coverage.parquet")
    return tmp_path


def _fixes(n=50, segments=(0, 1), start=0):
    """One fix per second; the viewer's own segment numbers (0 and 1)."""
    t = np.arange(start, start + n, dtype=float)
    return pd.DataFrame(
        {
            "segment_id": np.where(t < 30, segments[0], segments[1]),
            "t": t,
            "E": t,
            "N": t * 0.0,
            "z": t * 0.0,
        }
    )


def test_label_from_runs_joins_on_time_and_ignores_the_saved_segment_id():
    runs = pd.DataFrame(
        {
            "segment_id": [7, 7, 8],
            "phase": ["search", "climb", "transition"],
            "t_start": [-2.0, 10.0, 40.0],
            "t_end": [9.0, 19.0, 49.0],
        }
    )
    out = data.label_from_runs(_fixes(), runs)
    phase = out.set_index("t")["phase"]
    assert phase.loc[0:9].eq("search").all()
    assert phase.loc[10:19].eq("climb").all()
    assert phase.loc[20:39].eq("unclassified").all()
    assert phase.loc[40:49].eq("transition").all()
    assert out.loc[out.t == 15, "phase_reason"].item() == "classified"
    assert out.loc[out.t == 25, "phase_reason"].item() == "outside_saved_runs"
    # Line grouping: a new run at every change of phase and of cleaning segment.
    assert out["phase_run"].nunique() == 5


def test_label_from_runs_keeps_the_viewers_rows_and_order():
    fixes = _fixes().sample(frac=1.0, random_state=0)
    out = data.label_from_runs(
        fixes, pd.DataFrame(columns=["phase", "t_start", "t_end"])
    )
    assert len(out) == len(fixes)
    assert out["t"].is_monotonic_increasing
    assert out["phase"].eq("unclassified").all()


def test_load_refit_phases_reads_one_flight_and_leaves_the_rest_unclassified(archive):
    track = data.load_refit_phases(_fixes(), PARAGLIDERS, "10")
    assert track.mapping_method == "refit-2026"
    phase = track.fixes.set_index("t")["phase"]
    assert phase.loc[0:9].eq("search").all()
    assert phase.loc[26:39].eq("unclassified").all()
    assert track.coverage["fixes_by_reason"] == {
        "classified": 10 + 10 + 6 + 10,
        "outside_saved_runs": 14,
    }
    # Another flight's runs never reach this one.
    other = data.load_refit_phases(_fixes(), PARAGLIDERS, "11")
    assert other.fixes["phase"].eq("climb").sum() == 6


def test_load_refit_phases_names_the_status_of_a_flight_without_runs(archive):
    track = data.load_refit_phases(_fixes(), PARAGLIDERS, "12")
    assert track.fixes["phase"].eq("unclassified").all()
    assert track.coverage["fixes_by_reason"] == {"cadence_above_gate": 50}
    unknown = data.load_refit_phases(_fixes(), PARAGLIDERS, "99")
    assert unknown.coverage["fixes_by_reason"] == {"not_in_saved_runs": 50}


def test_load_refit_phases_is_none_without_the_saved_files(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path))
    monkeypatch.delenv(PARAGLIDERS.env, raising=False)
    (tmp_path / "paragliders/derived").mkdir(parents=True)
    assert data.load_refit_phases(_fixes(), PARAGLIDERS, "10") is None


def _track(phases):
    fixes = pd.DataFrame({"phase": phases})
    return data.PhaseTrack(fixes=fixes, mapping_method="x")


def test_agreement_counts_overall_and_per_phase():
    a = _track(["climb", "climb", "search", "search", "transition", "unclassified"])
    b = _track(
        ["climb", "search", "search", "transition", "unclassified", "unclassified"]
    )
    result = data.agreement(a, b)
    assert result["n_fixes"] == 6
    assert result["both_classified"] == 4
    assert result["agree"] == 2
    assert result["share"] == 0.5
    assert (result["only_first"], result["only_second"]) == (1, 0)
    assert result["by_phase"]["climb"] == {"both": 1, "either": 2, "share": 0.5}
    assert result["by_phase"]["search"] == {"both": 1, "either": 3, "share": 1 / 3}
    assert result["by_phase"]["transition"]["both"] == 0


def test_agreement_without_a_common_label_has_no_share():
    result = data.agreement(_track(["unclassified"]), _track(["unclassified"]))
    assert result["share"] is None
    assert result["by_phase"] == {}


def test_agreement_refuses_tracks_of_different_length():
    with pytest.raises(ValueError):
        data.agreement(_track(["climb"]), _track(["climb", "climb"]))


def test_the_default_is_vilpellet_and_use_switches_it():
    assert segmentation.current().key == "vilpellet"
    assert segmentation.use("refit").mapping_method == "refit-2026"
    assert segmentation.current().key == "refit"
    with pytest.raises(ValueError):
        segmentation.use("nope")
    assert segmentation.current().key == "refit"


def test_load_phases_follows_the_choice(archive):
    fixes = _fixes()
    assert data.load_phases(fixes, PARAGLIDERS, "10", "refit").mapping_method == (
        "refit-2026"
    )
    segmentation.use("refit")
    assert data.load_phases(fixes, PARAGLIDERS, "10").mapping_method == "refit-2026"
    segmentation.use("vilpellet")
    # Vilpellet decodes the fixes itself and reads no saved file.
    assert data.load_phases(fixes, PARAGLIDERS, "10").mapping_method == (
        "vilpellet-fitted-2019"
    )


def test_missing_names_the_absent_files_and_skips_absent_disciplines(archive, tmp_path):
    assert segmentation.missing(key="refit") == []
    absent = segmentation.missing(key="vilpellet")
    assert "paragliders/derived/segmentation/vilpellet/phase_segments.parquet" in absent
    assert not any(name.startswith("hang_gliders/") for name in absent)
    text = segmentation.needs_text("vilpellet")
    assert text.startswith("The Vilpellet segmentation needs ")
    assert segmentation.needs_text("refit") == ""
