#!/usr/bin/env python3
"""Save the 3 x 3 neighbourhood of every published cell into the viewer file.

Zoom - in the thermal-plane viewer reads only these products. The first run
labels every flight crossing the neighbouring squares from the processed
archives (Vilpellet segmentation) and downloads their IGN backgrounds; later runs
resume where the previous one stopped.
"""

import argparse
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xc_viewer.locking import BusyError, exclusive  # noqa: E402
from xc_viewer.thermal_neighbours import (  # noqa: E402
    prepare_neighbour_imagery,
    prepare_neighbour_points,
    prepare_neighbour_terrain,
)
from xc_viewer.thermal_store import load_store  # noqa: E402

BUSY = "Another offline preparation is already running."


def main() -> int:
    """Prepare the neighbourhoods under the shared offline-writer lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--image-px",
        type=int,
        default=4000,
        help="Width of a neighbour background in pixels (published cells: 4000)",
    )
    parser.add_argument("--skip-imagery", action="store_true")
    parser.add_argument(
        "--terrain-only",
        action="store_true",
        help="Only complete the IGN elevation cache for the 10 km 3D view",
    )
    args = parser.parse_args()
    store = args.store or load_store().path
    last = 0.0

    def progress(message: str) -> None:
        """Report progress without one line per flight."""
        nonlocal last
        now = monotonic()
        if now - last >= 2:
            print(message, flush=True)
            last = now

    try:
        with exclusive(store.with_name(".prepare"), BUSY):
            if not args.terrain_only:
                prepare_neighbour_points(store, workers=args.workers, progress=progress)
                if not args.skip_imagery:
                    prepare_neighbour_imagery(
                        store,
                        size=args.image_px,
                        progress=lambda s: print(s, flush=True),
                    )
            prepare_neighbour_terrain(store, progress=lambda s: print(s, flush=True))
    except BusyError as exc:
        print(exc)
        return 1
    except KeyboardInterrupt:
        print("Stopped. Completed work is kept; run again to resume.")
        return 130
    print(f"Ready: {store} ({store.stat().st_size / 1e9:.3f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
