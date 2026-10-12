"""The shared definitions of the regional boxes and their padded map frames.

No Qt import (see the package docstring).
"""

from __future__ import annotations

from .core.regions import REGIONAL_BOXES

# (west, east, south, north). REGIONAL_BOXES lacks the Massif Central box, which the
# regional maps also draw.
MASSIF_CENTRAL = (1.8, 4.6, 43.6, 46.2)
REGIONS = {**REGIONAL_BOXES, "Massif Central": MASSIF_CENTRAL}

PAD_DEG = 0.3


def extent(box) -> tuple[float, float, float, float]:
    """``(lon_min, lat_min, lon_max, lat_max)`` of the padded map frame for ``box``."""
    west, east, south, north = box
    return west - PAD_DEG, south - PAD_DEG, east + PAD_DEG, north + PAD_DEG
