"""Per-flight visibility must survive styling and agree with endpoint markers."""

import numpy as np
import pandas as pd
import pytest

from xc_viewer.route_scene import RouteScene


@pytest.fixture(params=["routes", "groups"])
def viewer(request, qapp):
    from xc_viewer.widgets.group_scene import GroupScene
    from xc_viewer.widgets.route_comparison import RouteComparison

    groups = request.param == "groups"
    view = GroupScene() if groups else RouteComparison()
    population = pd.DataFrame(
        {
            "rank": np.arange(1, 13),
            "discipline": "para",
            "flight_id": [str(i) for i in range(12)],
            "duration_s": 1000.0,
            "speed_group": ["fast"] * 5 + ["other"] * 2 + ["slow"] * 5,
            "departure_utc": 1700000000 + np.arange(12) * 60,
            "arrival_utc": 1700001000 + np.arange(12) * 60,
            "departure_delay_s": np.arange(12) * 60,
            "path_km": 80.0,
            "net_km": 100.0,
            "gap_s": 10.0,
            "clean_segments": 2,
        }
    )
    scene = RouteScene(
        population,
        12,
        (180, 1300) if groups else (90, 650, 100, 650),
        [
            [
                np.array([[a, i * 100, 1000], [b, i * 100, 2000]], np.float32)
                for a, b in [(-50000, -10000), (10000, 50000)]
            ]
            for i in range(12)
        ],
        (900000, 6500000, 1010000, 6510000),
        np.array([955000, 6505000]),
        cell_m=5000 if groups else 10000,
    )
    view.set_scene(scene)
    qapp.processEvents()  # Complete the group's initial deferred camera fit.
    yield view
    view.shutdown()
    view.close()


def assert_visible(view, expected):
    for lines, show in zip(view._lines, expected, strict=True):
        assert all(line.visible() == show for line in lines)
    np.testing.assert_array_equal(view._annotations.mask, expected)
    assert view._visible_count.text() == (
        f"{sum(expected)} / {len(expected)} flights visible"
    )


def test_checkboxes_persist_across_styles_and_modes_and_reset_for_new_scene(
    viewer, qapp, monkeypatch
):
    from PyQt6.QtCore import Qt

    scene = viewer._scene
    population = scene.selected.copy(deep=True)
    camera = viewer._view.cameraParams()
    positions = [[line.pos.copy() for line in lines] for lines in viewer._lines]
    with monkeypatch.context() as patch:
        patch.setattr(viewer, "set_scene", lambda *_: pytest.fail("Geometry reloaded"))
        item = viewer._table.item(0, 0)
        item.setCheckState(Qt.CheckState.Unchecked)
        expected = [False] + [True] * 11
        assert_visible(viewer, expected)
        assert not viewer._table.isRowHidden(0)  # Can tick it again.
        viewer._table.selectRow(0)  # Highlighting never resurrects a hidden flight.
        for change in (
            lambda: viewer._line_width.setValue(0.3),
            lambda: viewer._surface_mode.setCurrentIndex(2),
            lambda: viewer._terrain.setChecked(False),
            lambda: viewer._vertical.setValue(3),
        ):
            change()
            assert_visible(viewer, expected)
            assert viewer._view.cameraParams() == camera
        if viewer._mode.findData("fast") >= 0:
            viewer._mode.setCurrentIndex(viewer._mode.findData("fast"))
            assert_visible(viewer, [False] + [True] * 4 + [False] * 7)
            assert not viewer._table.isRowHidden(0)
        else:
            viewer._mode.setCurrentIndex(viewer._mode.findData("selected"))
            assert_visible(viewer, [False] * 12)
            viewer._table.selectRow(1)
            assert_visible(viewer, [False, True] + [False] * 10)
        viewer._mode.setCurrentIndex(viewer._mode.findData("all"))
        assert_visible(viewer, expected)
        viewer._hide_all.click()
        assert_visible(viewer, [False] * 12)
        assert viewer._table.rowCount() == 12
        viewer._table.item(3, 0).setCheckState(Qt.CheckState.Checked)
        assert_visible(viewer, [i == 3 for i in range(12)])
        viewer._mode.setCurrentIndex(1)
        viewer._show_all.click()
        assert viewer._mode.currentData() == "all"
        assert_visible(viewer, [True] * 12)
        assert viewer._view.cameraParams() == camera
        pd.testing.assert_frame_equal(scene.selected, population)
        for lines, original in zip(viewer._lines, positions, strict=True):
            for line, xyz in zip(lines, original, strict=True):
                np.testing.assert_array_equal(line.pos, xyz)
    viewer._hide_all.click()
    viewer.set_scene(scene)
    assert_visible(viewer, [True] * 12)
    viewer._clear_scene()
    assert viewer._visible_count.text() == "0 / 0 flights visible"
    assert not viewer._show_all.isEnabled() and not viewer._hide_all.isEnabled()


def test_one_click_on_checkbox_hides_then_restores_flight(viewer, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

    # Show the table independently; the click test does not require a GPU context.
    table = viewer._table
    table.setParent(None)
    table.resize(1000, 350)
    table.show()
    qapp.processEvents()
    try:
        item = table.item(0, 0)
        option = QStyleOptionViewItem()
        option.initFrom(table)
        option.rect = table.visualItemRect(item)
        option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
        option.checkState = Qt.CheckState.Checked
        checkbox = table.style().subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, table
        )
        assert checkbox.isValid()
        for checked in (False, True):
            QTest.mouseClick(
                table.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center()
            )
            assert (item.checkState() == Qt.CheckState.Checked) == checked
            assert_visible(viewer, [checked] + [True] * 11)
            assert not table.isRowHidden(0)
    finally:
        table.close()
        table.setParent(viewer)
