"""Finer IGN relief and aerial photo in a square around the centre of the view."""

from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..route_scene import DETAIL_SIDE_M, detail_window, load_detail

#: Milliseconds between two looks at the camera.
POLL_MS = 500
#: The square loads when the camera is nearer than this many sides to the view centre.
NEAR = 3
#: Scenes narrower than this many sides keep the overview alone.
WIDER = 1.5


class TerrainDetail(QObject):
    """Patch the terrain of a scene with finer data where the camera looks.

    The square follows the centre of the view, loads in the background and replaces
    the overview inside it; its responses are cached on disk with the overview's.
    """

    #: The TerrainPatch just drawn.
    patched = pyqtSignal(object)

    def __init__(self, parent=None, loader=load_detail):
        """Nothing loads until `attach`; `loader` is load_detail or a stand-in."""
        super().__init__(parent)
        self._loader = loader
        self._pool = ThreadPoolExecutor(1)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._scene = self._view = self._surface = None
        self._future = self._pending = self.patch = None
        self._failed = set()

    def attach(self, scene, view, surface):
        """Follow `view` over the terrain `surface` of a scene wider than WIDER."""
        self.clear()
        if scene.terrain is None or scene.terrain_folder is None:
            return
        west, south, east, north = scene.bounds
        if max(east - west, north - south) < WIDER * DETAIL_SIDE_M:
            return
        self._scene, self._view, self._surface = scene, view, surface
        self._timer.start()

    def clear(self):
        """Stop following; a load still running is dropped when it ends."""
        self._timer.stop()
        if self._future is not None:
            self._future.cancel()
        self._scene = self._view = self._surface = None
        self._future = self._pending = self.patch = None
        self._failed.clear()

    def _poll(self):
        """Draw a finished load, then ask for the square under the camera."""
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            try:
                patch = future.result()
            except (OSError, ValueError):
                self._failed.add(self._pending)
            else:
                self._draw(patch)
        if (
            self._future is not None
            or self._view.opts["distance"] > NEAR * DETAIL_SIDE_M
        ):
            return
        centre, origin = self._view.opts["center"], self._scene.origin
        window = detail_window(
            self._scene.terrain, (centre.x() + origin[0], centre.y() + origin[1])
        )
        current = None if self.patch is None else self.patch.window
        if window is None or window == current or window in self._failed:
            return
        self._pending = window
        self._future = self._pool.submit(
            self._loader,
            self._scene.terrain,
            window,
            self._scene.origin,
            self._scene.terrain_folder,
        )

    def _draw(self, patch):
        """Replace the mesh and the detail photo of the surface."""
        if self._view.isValid():
            self._view.makeCurrent()
            try:
                self._surface.release_detail()
            finally:
                self._view.doneCurrent()
        self._surface.setMeshData(
            vertexes=patch.vertices, faces=patch.faces, vertexColors=patch.colours
        )
        self._surface.set_detail(patch.photo, patch.bounds)
        self.patch = patch
        self.patched.emit(patch)
