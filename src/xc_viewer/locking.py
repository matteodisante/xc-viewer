"""One writer at a time for each saved product, on every operating system.

No Qt import (see the package docstring).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class BusyError(RuntimeError):
    """Another process (or viewer window) is already preparing the product."""


@contextmanager
def exclusive(product: str | Path, busy: str) -> Iterator[None]:
    """Hold the lock file beside ``product``, or fail at once if another holds it.

    Args:
        product: The file being written; the lock is ``<product>.lock``.
        busy: The message of the error raised when the lock is already held.

    Raises:
        BusyError: Another process (or viewer window) is preparing ``product``.
    """
    lock = FileLock(Path(product).with_suffix(".lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise BusyError(busy) from exc
    try:
        yield
    finally:
        lock.release()
