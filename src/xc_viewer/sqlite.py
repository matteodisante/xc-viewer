"""SQLite connections that are closed when their ``with`` block ends.

``with sqlite3.connect(path) as db`` only commits or rolls back: the connection stays
open until garbage collection. On Windows an open database cannot be renamed or
deleted, and every saved product is published by renaming a finished temporary file.

No Qt import (see the package docstring).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def connect(*args, **kwargs) -> Iterator[sqlite3.Connection]:
    """``sqlite3.connect`` for a ``with`` block: commit or roll back, then close.

    Args:
        *args: Passed to :func:`sqlite3.connect`.
        **kwargs: Passed to :func:`sqlite3.connect`.

    Yields:
        The open connection.
    """
    db = sqlite3.connect(*args, **kwargs)
    try:
        with db:
            yield db
    finally:
        db.close()
