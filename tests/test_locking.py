"""A second writer of the same product is refused at once, on every system."""

import pytest

from xc_viewer.locking import exclusive


def test_a_held_product_lock_refuses_a_second_writer(tmp_path):
    product = tmp_path / "route-cells.sqlite3"
    with exclusive(product, "busy"):  # noqa: SIM117 (the outer lock must be held)
        with pytest.raises(RuntimeError, match="busy"), exclusive(product, "busy"):
            pass
    with exclusive(product, "busy"):
        pass
