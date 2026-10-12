"""Trackpad navigation moves through the scene without changing the 3D data."""

import numpy as np
import pytest

pytest.importorskip("pyqtgraph.opengl")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import (
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QNativeGestureEvent,
    QPointingDevice,
    QVector3D,
    QWheelEvent,
)

from xc_viewer.widgets.thermal_3d import TerrainView


@pytest.fixture
def view(qapp):
    widget = TerrainView()
    widget.resize(800, 600)
    widget.setCameraPosition(
        pos=QVector3D(0, 0, 1000), distance=100, elevation=35, azimuth=-55
    )
    yield widget
    widget.stop_movement()
    widget.close()


def xyz(vector):
    return np.array([vector.x(), vector.y(), vector.z()])


def wheel(
    view,
    *,
    pixels=None,
    angles=None,
    phase=Qt.ScrollPhase.NoScrollPhase,
    modifiers=Qt.KeyboardModifier.NoModifier,
):
    event = QWheelEvent(
        QPointF(200, 200),
        QPointF(200, 200),
        QPoint() if pixels is None else pixels,
        QPoint() if angles is None else angles,
        Qt.MouseButton.NoButton,
        modifiers,
        phase,
        False,
    )
    view.wheelEvent(event)
    assert event.isAccepted()


@pytest.mark.parametrize("delta", [-120, 120])
def test_two_finger_scroll_keeps_original_zoom_sensitivity_and_target(view, delta):
    center = xyz(view.opts["center"])
    wheel(
        view,
        pixels=QPoint(40, -25),
        angles=QPoint(0, delta),
        phase=Qt.ScrollPhase.ScrollUpdate,
    )
    np.testing.assert_array_equal(xyz(view.opts["center"]), center)
    assert view.opts["distance"] == pytest.approx(100 * 0.999**delta)
    assert view.opts["fov"] == 60
    assert view.opts["azimuth"] == -55 and view.opts["elevation"] == 35


def test_mouse_wheel_and_ctrl_scroll_zoom_without_distorting_fov(view):
    center = xyz(view.opts["center"])
    wheel(view, angles=QPoint(0, 120))
    assert 1 < view.opts["distance"] < 100
    distance = view.opts["distance"]
    wheel(
        view,
        pixels=QPoint(0, 40),
        phase=Qt.ScrollPhase.ScrollUpdate,
        modifiers=Qt.KeyboardModifier.ControlModifier,
    )
    assert 1 < view.opts["distance"] < distance
    np.testing.assert_array_equal(xyz(view.opts["center"]), center)
    assert view.opts["fov"] == 60


def test_native_mac_pinch_zooms_and_can_reverse(view, qapp):
    center = xyz(view.opts["center"])
    for amount in (0.5, -0.5):
        event = QNativeGestureEvent(
            Qt.NativeGestureType.ZoomNativeGesture,
            QPointingDevice.primaryPointingDevice(),
            2,
            QPointF(200, 200),
            QPointF(200, 200),
            QPointF(200, 200),
            amount,
            QPointF(),
        )
        qapp.sendEvent(view, event)
        assert event.isAccepted()
        if amount > 0:
            assert view.opts["distance"] < 100
    assert view.opts["distance"] == pytest.approx(100)
    assert view.opts["fov"] == 60
    np.testing.assert_array_equal(xyz(view.opts["center"]), center)


@pytest.mark.parametrize(
    "modifier", [Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier]
)
def test_one_finger_drag_can_pan_or_look_from_fixed_camera(view, modifier):
    center, camera = xyz(view.opts["center"]), xyz(view.cameraPosition())
    for kind, point in (
        (QEvent.Type.MouseButtonPress, QPointF(100, 100)),
        (QEvent.Type.MouseMove, QPointF(170, 140)),
    ):
        event = QMouseEvent(
            kind,
            point,
            point,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            modifier,
        )
        if kind == QEvent.Type.MouseButtonPress:
            view.mousePressEvent(event)
        else:
            view.mouseMoveEvent(event)
    assert not np.allclose(xyz(view.opts["center"]), center)
    if modifier == Qt.KeyboardModifier.AltModifier:
        np.testing.assert_allclose(xyz(view.cameraPosition()), camera, atol=1e-4)
        assert view.opts["azimuth"] != -55
    else:
        assert view.opts["azimuth"] == -55 and view.opts["elevation"] == 35
    assert view.opts["distance"] == 100


@pytest.mark.parametrize(
    "key, direction",
    [
        (Qt.Key.Key_W, [0, 1, 0]),
        (Qt.Key.Key_S, [0, -1, 0]),
        (Qt.Key.Key_A, [-1, 0, 0]),
        (Qt.Key.Key_D, [1, 0, 0]),
        (Qt.Key.Key_Q, [0, 0, -1]),
        (Qt.Key.Key_E, [0, 0, 1]),
    ],
)
def test_keys_translate_along_view_or_vertical_and_release_stops(view, key, direction):
    view.setCameraPosition(elevation=0, azimuth=-90)
    center, camera = xyz(view.opts["center"]), xyz(view.cameraPosition())
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )
    shift = xyz(view.cameraPosition()) - camera
    assert np.linalg.norm(shift) > 0
    np.testing.assert_allclose(shift / np.linalg.norm(shift), direction, atol=1e-4)
    np.testing.assert_allclose(xyz(view.opts["center"]) - center, shift, atol=1e-4)
    assert view.opts["distance"] == 100
    assert view.keyTimer.isActive()
    view.keyReleaseEvent(
        QKeyEvent(QEvent.Type.KeyRelease, key, Qt.KeyboardModifier.NoModifier)
    )
    assert not view.keyTimer.isActive() and not view.keysPressed


def test_losing_focus_stops_movement_and_escape_can_reach_dialog(view):
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_W, Qt.KeyboardModifier.NoModifier)
    )
    view.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    center = xyz(view.opts["center"])
    view.evalKeyState()
    np.testing.assert_array_equal(xyz(view.opts["center"]), center)
    assert not view.keyTimer.isActive() and not view.keysPressed
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    view.keyPressEvent(event)
    assert not event.isAccepted()


@pytest.mark.parametrize("distance", [1, 20000])
def test_close_zoom_keeps_nearby_and_distant_terrain_inside_clip_planes(view, distance):
    view.setCameraPosition(distance=distance)
    viewport = (0, 0, 800, 600)
    projection = view.projectionMatrix(viewport, viewport)
    # At close zoom the far side of a 10 km area must remain visible; moving
    # forward with a distant orbit target must not hide nearby objects either.
    for depth in (1.1, 10000):
        projected = projection.map(QVector3D(0, 0, -depth))
        assert -1 < projected.z() < 1
