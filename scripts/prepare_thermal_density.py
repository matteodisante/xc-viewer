#!/usr/bin/env python3
"""Prepare conservative thermal residence-time maps in hours/km² in the data folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xc_viewer.thermal_time_prepare import prepare  # noqa: E402


def main():
    """Build all regions and cells, or benchmark a bounded separate output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["regions", "cells", "routes"])
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.only == "routes":
        from xc_viewer.route_density import prepare_density

        if args.limit_batches is not None:
            parser.error("--only routes requires a complete archive scan")
        prepare_density(
            path=args.output, progress=lambda message: print(message, flush=True)
        )
        return
    prepare(
        regions=args.only != "cells",
        cells=args.only != "regions",
        limit_batches=args.limit_batches,
        output=args.output,
        progress=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
