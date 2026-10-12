"""Interactive viewer of cross-country soaring flights and their thermals.

Only :mod:`xc_viewer.app`, :mod:`xc_viewer.main_window` and the
``widgets`` it composes import Qt. Modules such as ``data.py``, ``catalog_index.py``
and ``plotting.py`` import neither Qt nor matplotlib's Qt backend, so they stay usable
-- and testable -- without a display.
"""

from __future__ import annotations
