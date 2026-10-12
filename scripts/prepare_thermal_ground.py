#!/usr/bin/env python3
"""Rebase saved climb intersections on each cell's lowest IGN terrain elevation."""

import argparse
from pathlib import Path

from xc_viewer.locking import exclusive
from xc_viewer.thermal_ground import upgrade_terrain_store
from xc_viewer.thermal_store import find_store_path


def main():
    """Upgrade atomically using saved edges and DEMs, without decoding flights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--terrain-directory", type=Path)
    args = parser.parse_args()
    path = args.store or find_store_path()
    if path is None:
        parser.error("Connect the data folder or pass --store thermal-planes.sqlite3")
    with exclusive(
        path.with_name(".prepare"), "Another offline preparation is already running."
    ):
        upgrade_terrain_store(
            path,
            folder=args.terrain_directory,
            progress=lambda message: print(message, flush=True),
        )


if __name__ == "__main__":
    main()
