"""Debounced background requests with a single worker and stale-result rejection."""

from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..density_maps import fetch_background


class DensityBackgrounds(QObject):
    """Keep only the newest requested viewport per panel; never block Qt on HTTP."""

    ready = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, folder, parent=None):
        """Start no network work until viewports are requested."""
        super().__init__(parent)
        self.folder = folder
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="density-map")
        self._wanted = {}
        self._done = {}
        self._active = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def request(self, wanted):
        """Replace pending work after a zoom, resize, or source change."""
        self._wanted = dict(wanted)
        self._done = {k: v for k, v in self._done.items() if wanted.get(k) == v}
        if wanted or self._active:
            self._timer.start()

    def reset(self):
        """New axes need their own artists even if the viewport is unchanged."""
        self._wanted.clear()
        self._done.clear()
        # Invalidate an in-flight result without preventing its cache write.
        if self._active:
            future, _, request = self._active
            self._active = future, None, request

    def close(self):
        """Cancel queued work and stop callbacks; an active HTTP request may finish."""
        self._timer.stop()
        self._wanted.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _poll(self):
        if self._active:
            future, key, request = self._active
            if not future.done():
                return
            self._active = None
            current = self._wanted.get(key) == request
            try:
                result = future.result()
            except Exception as exc:
                if current:
                    self._done[key] = request
                    self.failed.emit(
                        key, f"Background unavailable: {exc}. Zoom again to retry."
                    )
            else:
                if current:
                    self._done[key] = request
                    self.ready.emit(key, result)
        for key, request in self._wanted.items():
            if self._done.get(key) != request:
                self._active = (
                    self._pool.submit(fetch_background, self.folder, request),
                    key,
                    request,
                )
                return
        self._timer.stop()
