"""Descriptive geographic boxes shared by the thesis map and regional analyses."""

# West, east, south, north in decimal degrees; boundaries are inclusive.
REGIONAL_BOXES = {
    "Alps": (5.4, 10.0, 43.8, 46.6),
    "Pyrenees": (-1.9, 3.3, 42.0, 43.5),
    "Channel Coast": (-1.8, 2.0, 48.3, 51.2),
    "Champagne-Lorraine": (2.1, 6.2, 48.0, 50.6),
}


def region_box_masks(frame):
    """Select initial positions within each box, without altitude restrictions."""
    return {
        name: (frame.lon0.between(west, east) & frame.lat0.between(south, north))
        .fillna(False)
        .to_numpy(dtype=bool)
        for name, (west, east, south, north) in REGIONAL_BOXES.items()
    }
