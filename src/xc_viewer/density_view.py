"""A density image that re-bins itself to the visible window when the user zooms.

matplotlib only, no Qt (see the package docstring). The toolbar's zoom and pan change
the axes limits; this listens for that and asks a ``source`` for a fresh histogram of
just the visible window, so zooming in reveals finer bins instead of larger pixels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

#: ``source(x0, x1, y0, y1) -> (counts[y, x], x_edges, y_edges, bin_label)``.
Source = Callable[[float, float, float, float], tuple]


def histogram_source(x, y, x_range, y_range, min_bin, bins=150) -> Source:
    """A source that bins scattered points, about ``bins`` bins across the view.

    Bin sizes are multiples of ``min_bin`` (in the axes' own unit), so a deeper zoom
    stops refining at the data's own resolution.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def source(x0, x1, y0, y1):
        x0, x1 = max(x0, x_range[0]), min(x1, x_range[1])
        y0, y1 = max(y0, y_range[0]), min(y1, y_range[1])
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1)), np.array([x0, x0 + 1]), np.array([y0, y0 + 1]), ""
        size = max(np.ceil((x1 - x0) / bins / min_bin), 1) * min_bin
        xe = np.arange(np.floor(x0 / size), np.ceil(x1 / size) + 1) * size
        ye = np.arange(np.floor(y0 / size), np.ceil(y1 / size) + 1) * size
        counts, _, _ = np.histogram2d(x, y, bins=(xe, ye))
        return counts.T, xe, ye, f"{size:g}"

    return source


class AdaptiveDensity:
    """One image (and colorbar) on ``ax``, kept matched to the visible window."""

    def __init__(
        self,
        ax,
        source: Source,
        *,
        label="Count per bin",
        unit="",
        cmap="magma_r",
        alpha=1.0,
        norm=None,
    ):
        """Draw for the current limits and follow later zoom/pan on ``ax``."""
        self.ax, self.source, self.label, self.unit = ax, source, label, unit
        self.image = None
        self.colorbar = None
        self._size = ""
        self._alpha = alpha
        self._last = None
        self._cids = [
            ax.callbacks.connect(name, self._update)
            for name in ("xlim_changed", "ylim_changed")
        ]
        self._cmap = cmap
        self._norm = norm

    def add_colorbar(self, figure):
        """Attach a colorbar once an image exists."""
        if self.image is not None and self.colorbar is None:
            self.colorbar = figure.colorbar(
                ScalarMappable(norm=self.image.norm, cmap=self._cmap),
                ax=self.ax,
                shrink=0.85,
                label=self._caption(),
            )

    def set_alpha(self, alpha):
        """Opacity of the density colours, from 0 (invisible) to 1."""
        self._alpha = alpha
        if self.image is not None:
            self.image.set_alpha(alpha)

    def _caption(self):
        return (
            f"{self.label} · {self._size} {self.unit}".strip()
            if self._size
            else self.label
        )

    def refresh(self):
        """Redraw for the current limits, even if they have not changed."""
        self._last = None
        self._update()

    def close(self):
        """Stop following the axes and drop the colorbar."""
        for cid in self._cids:
            self.ax.callbacks.disconnect(cid)
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

    def _update(self, _=None):
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        if (x0, x1, y0, y1) == self._last:
            return
        self._last = (x0, x1, y0, y1)
        counts, xe, ye, self._size = self.source(x0, x1, y0, y1)
        shown = np.ma.masked_less_equal(counts, 0)
        top = max(float(counts.max()) if counts.size else 1.0, 2.0)
        # Exact pixel edges also handle partial coarse pixels at the map boundary.
        # Replacing the mesh keeps zoomed arrays bounded without resampling values.
        if self.image is not None:
            self.image.remove()
        self.image = self.ax.pcolormesh(
            xe,
            ye,
            shown,
            shading="flat",
            cmap=self._cmap,
            norm=self._norm or LogNorm(vmin=1, vmax=top),
            alpha=self._alpha,
            zorder=3,
            rasterized=True,
        )
        if self.colorbar is not None:
            self.colorbar.set_label(self._caption())
            self.colorbar.mappable.set_norm(self.image.norm)
