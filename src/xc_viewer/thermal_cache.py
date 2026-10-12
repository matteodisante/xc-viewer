"""Persistent, resumable climb products shared by viewer sessions in the data folder."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .core.vilpellet.config import DEFAULT_VILPELLET_CONFIG_PATH
from .fingerprint import file_identity

if TYPE_CHECKING:
    from .thermal_index import ThermalIndex

EDGE_COLUMNS = [f"{axis}{end}" for end in (0, 1) for axis in ("x", "y", "z", "utc")]


#: Bump when the decoding or climb extraction code changes (data.py, geodesy.py,
#: thermal_geometry.py, thermal_index.py or core/vilpellet), so that saved climbs
#: from the previous code are no longer reused.
CLIMB_VERSION = 1


def segmentation_signature(index: ThermalIndex, source: str) -> str:
    """Identify climb products by census, code version and segmentation config.

    Portable (see :mod:`xc_viewer.fingerprint`): the same data folder gives
    the same key on any computer.
    """
    # A test/custom index without an archive signature is identified by its file.
    census = index.signature if index.signature else file_identity(index.path)
    config = Path(DEFAULT_VILPELLET_CONFIG_PATH)
    settings = file_identity(config) if config.is_file() else "missing"
    return hashlib.sha256(
        json.dumps([CLIMB_VERSION, census, source, settings]).encode()
    ).hexdigest()


class ClimbCache:
    """One compressed edge array per flight/cell/method, committed independently."""

    def __init__(self, index: ThermalIndex, source: str):
        """Open persistent products next to the archive's saved cell index."""
        self.path = index.path.with_name("thermal-climbs.sqlite3")
        self.key = segmentation_signature(index, source)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS climbs (
                cache_key TEXT, discipline TEXT, flight_id TEXT,
                ix INTEGER, iy INTEGER, status TEXT, edges BLOB,
                PRIMARY KEY (cache_key, discipline, flight_id, ix, iy)
            )
        """)
        self.db.commit()

    def get(self, discipline, flight_id, cell, *, geometry=True):
        """Read a completed result, or None; optionally skip decompressing geometry."""
        payload = "edges" if geometry else "NULL"
        record = self.db.execute(
            f"SELECT status, {payload} FROM climbs WHERE cache_key=? AND discipline=? "
            "AND flight_id=? AND ix=? AND iy=?",
            (self.key, discipline, flight_id, cell.ix, cell.iy),
        ).fetchone()
        if record is None:
            return None
        status, blob = record
        if not geometry or blob is None:
            edges = pd.DataFrame(columns=EDGE_COLUMNS)
        else:
            with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                edges = pd.DataFrame(saved["edges"], columns=EDGE_COLUMNS)
        return status, edges

    def put(self, discipline, flight_id, cell, status, edges):
        """Commit a whole-flight result; cancellation never leaves a partial flight."""
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, edges=edges.reindex(columns=EDGE_COLUMNS).to_numpy(dtype=float)
        )
        self.db.execute(
            "INSERT OR REPLACE INTO climbs VALUES (?,?,?,?,?,?,?)",
            (
                self.key,
                discipline,
                flight_id,
                cell.ix,
                cell.iy,
                status,
                buffer.getvalue(),
            ),
        )
        self.db.commit()

    def close(self):
        """Release the SQLite handle on the worker that opened it."""
        self.db.close()

    def __enter__(self):
        """Use a bounded handle around one load or preparation request."""
        return self

    def __exit__(self, *_):
        """Close even when a decode fails or the user cancels."""
        self.close()
