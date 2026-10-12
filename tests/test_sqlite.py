"""A finished database can be renamed at once, as Windows requires."""

import sqlite3

import pytest

from xc_viewer.sqlite import connect


def test_the_connection_is_closed_and_committed_after_the_block(tmp_path):
    temporary = tmp_path / "product.building.sqlite3"
    with connect(temporary) as db:
        db.execute("CREATE TABLE t (x)")
        db.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")
    temporary.replace(tmp_path / "product.sqlite3")
    with connect(tmp_path / "product.sqlite3") as db:
        assert db.execute("SELECT x FROM t").fetchall() == [(1,)]


def test_a_failed_block_rolls_back(tmp_path):
    path = tmp_path / "product.sqlite3"
    with connect(path) as db:
        db.execute("CREATE TABLE t (x)")
    with pytest.raises(RuntimeError), connect(path) as db:
        db.execute("INSERT INTO t VALUES (1)")
        raise RuntimeError
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
