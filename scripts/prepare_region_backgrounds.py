#!/usr/bin/env python3
"""Download the static IGN and hillshade backgrounds of the regional maps.

For the hours/km² maps themselves, use prepare_thermal_density.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xc_viewer.thermal_imagery import prepare_region_backgrounds  # noqa: E402
from xc_viewer.thermal_store import load_store  # noqa: E402


def main() -> int:
    """Write the regions' terrain images beside the store (needs network)."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731
    store = load_store()
    if store is None:
        print("thermal-planes.sqlite3 not found: skipping terrain images")
        return 1
    prepare_region_backgrounds(store.path, progress=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
