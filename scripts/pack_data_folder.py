#!/usr/bin/env python3
r"""Build a portable data folder from the thesis archive.

The thesis archive keeps each discipline under its own root (``raw/igc``,
``catalog``, ``derived``). This script copies, for each root, only what the viewer
reads into ``<destination>/paragliders`` and ``<destination>/hang_gliders``:

* the ``.igc`` tracks, the catalog and the cleaned tables (``fixes.parquet``,
  ``flights_meta.parquet``);
* the saved Vilpellet runs, coverage and parameters;
* every saved viewer product under ``derived/viewer`` (census, route index, thermal
  planes, density rasters, terrain and imagery caches).

The climb cache keeps only its current Vilpellet rows. Saved products then receive
the portable fingerprints of this repository (see ``xc_viewer.fingerprint``),
so the destination opens on any computer without preparing anything again.

A product is re-stamped only if the thesis viewer still considers it current. The
script asks the thesis repository's own code (``--thesis-repo``), so a stale product
is copied as is and the viewer will offer to prepare it again. Without the thesis
repository, every product is trusted as current.

The copy is resumable: a file already copied with the same size and time is skipped.
The thesis archive is only read, never modified.

Example::

    uv run python scripts/pack_data_folder.py \
        --para /Volumes/SSD/paragliders/ffvl_cfd_igc \
        --hang /Volumes/SSD/hang_gliders/delta_cfd_igc \
        --to /Volumes/Other/xc-data
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from xc_viewer import datafolder
from xc_viewer.core.disciplines import DISCIPLINES, HANG_GLIDERS, PARAGLIDERS
from xc_viewer.sqlite import connect

#: Files every discipline root contributes, relative to the root.
INPUTS = (
    "catalog/catalog.csv",
    "derived/fixes.parquet",
    "derived/flights_meta.parquet",
    "derived/segmentation/vilpellet/phase_segments.parquet",
    "derived/segmentation/vilpellet/phase_coverage.parquet",
    "derived/segmentation/vilpellet/model/parameters.json",
)
VIEWER = "derived/viewer"
#: Where the saved products sit inside the first discipline's root.
PRODUCTS = "derived/viewer/thermal-planes"
CLIMBS = "thermal-climbs.sqlite3"
#: Saved products the viewer no longer reads: older store versions and the Gaussian
#: HMM rasters.
UNUSED = ("archive", "route-thermal-duration.npz", "thermal-regions.npz")
#: SQLite tables whose text holds product fingerprints.
STAMPED_TABLES = {
    "thermal-cells.sqlite3": ("metadata", "build_info"),
    "route-cells.sqlite3": ("metadata",),
    "thermal-planes.sqlite3": ("metadata", "neighbour_flights"),
    "thermal-vilpellet-activity.sqlite3": ("metadata",),
}

#: Run inside the thesis repository: the fingerprints its viewer expects right now.
THESIS_PROBE = r"""
import json
from soaring.reporting.disciplines import DISCIPLINES
from soaring.viewer import group_flights, route_density, route_index, thermal_index
from soaring.viewer.thermal_activity import _validated_products
from soaring.viewer.thermal_cache import segmentation_signature


def attempt(compute):
    try:
        return compute()
    except Exception:
        return None


discs = [d for d in DISCIPLINES.values() if d.derived_dir() is not None]
census = thermal_index.load_saved_index(discs)
routes = route_index.load_saved_index()
archives = route_index.available_archives()
probe = {
    "census": attempt(lambda: thermal_index.archive_signature(discs)),
    "climbs": attempt(lambda: segmentation_signature(census, "vilpellet")),
    "activity": attempt(lambda: _validated_products(census)[1]),
    "routes": attempt(lambda: route_index.archive_signature(archives)),
    "density": attempt(lambda: route_density.source_signature(archives)),
    "group": attempt(lambda: group_flights._signature(routes)),
}
print("PROBE" + json.dumps(probe))
"""


def _plan(source: Path, target: Path) -> list[tuple[Path, Path]]:
    """Every (source, destination) file pair of one discipline, climbs excluded."""
    pairs = [(source / name, target / name) for name in INPUTS]
    pairs += [
        (path, target / path.relative_to(source))
        for path in sorted((source / "raw/igc").rglob("*.igc"))
    ]
    viewer = source / VIEWER
    if viewer.is_dir():
        for path in sorted(viewer.rglob("*")):
            parts = path.relative_to(viewer).parts
            if (
                path.is_file()
                and path.name != CLIMBS
                and not any(part in UNUSED for part in parts)
                and not any(part.startswith(".") for part in parts)
                and path.suffix not in (".lock", ".copying")
                and not path.name.endswith(("-journal", "-wal", "-shm"))
                and ".building" not in path.name
            ):
                pairs.append((path, target / path.relative_to(source)))
    return [(a, b) for a, b in pairs if a.is_file()]


def _reusable(source: Path, candidate: Path | None) -> bool:
    """Whether ``candidate`` is an identical copy of ``source`` to link from.

    Tracks are compared by size, other files byte for byte.
    """
    if candidate is None or not candidate.is_file():
        return False
    if candidate.stat().st_size != source.stat().st_size:
        return False
    return source.suffix == ".igc" or filecmp.cmp(source, candidate, shallow=False)


def _present(source: Path, target: Path, candidate: Path | None) -> bool:
    """Whether ``target`` already holds ``source``.

    Same size, and either the same time, the same linked file, or a track (tracks
    never change once downloaded).
    """
    if not target.is_file():
        return False
    current, stat = target.stat(), source.stat()
    if current.st_size != stat.st_size:
        return False
    return (
        current.st_mtime_ns == stat.st_mtime_ns
        or target.suffix == ".igc"
        or (
            candidate is not None and candidate.is_file() and candidate.samefile(target)
        )
    )


def _origin(source, candidate, *, link: bool) -> Path | None:
    """The file to hard-link ``source``'s destination from, or ``None`` to copy."""
    if _reusable(source, candidate):
        return candidate
    if link and source.suffix in (".igc", ".parquet", ".csv"):
        return source
    return None


def _copy(
    pairs: list[tuple[Path, Path]],
    *,
    link: bool,
    root: Path,
    reuse: Path | None = None,
) -> None:
    """Copy (or hard-link) each file unless an identical copy is already there.

    Args:
        pairs: (source, destination) files of one discipline.
        link: Hard-link tracks and tables from ``root`` (same disk only).
        root: The discipline's archive root the sources sit in.
        reuse: An identical copy of ``root`` on the destination disk; its files are
            hard-linked instead of copied.
    """
    total = sum(a.stat().st_size for a, _ in pairs)
    done = linked = 0
    for i, (source, target) in enumerate(pairs, 1):
        stat = source.stat()
        done += stat.st_size
        candidate = reuse / source.relative_to(root) if reuse else None
        if _present(source, target, candidate):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.copying")
        temporary.unlink(missing_ok=True)
        origin = _origin(source, candidate, link=link)
        if origin is None:
            shutil.copy2(source, temporary)
        else:
            # Never fall back to a copy: the space check counted this file as free.
            os.link(origin, temporary)
            linked += 1
        temporary.replace(target)
        if i % 5000 == 0 or stat.st_size > 2**30:
            print(f"  {done / 1e9:,.1f} / {total / 1e9:,.1f} GB", flush=True)
    if linked:
        print(f"  {linked:,} files hard-linked instead of copied")


def _links_work(folder: Path) -> bool:
    """Whether the disk holding ``folder`` supports hard links (exFAT does not)."""
    folder.mkdir(parents=True, exist_ok=True)
    first, second = folder / ".link-probe", folder / ".link-probe-2"
    try:
        first.write_bytes(b"")
        os.link(first, second)
        return True
    except OSError:
        return False
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def _space_needed(pairs, *, link: bool, root: Path, reuse: Path | None, block: int):
    """Bytes the copies will occupy, rounded up to the destination's block size."""
    needed = 0
    for source, target in pairs:
        candidate = reuse / source.relative_to(root) if reuse else None
        if _present(source, target, candidate) or _origin(source, candidate, link=link):
            continue
        needed += -(-source.stat().st_size // block) * block
    return needed


def _probe(thesis: Path, roots: dict[str, Path]) -> dict:
    """The thesis viewer's current fingerprints for the given archive roots."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOARING_")}
    # A discipline left out must stay out of the thesis' fingerprints too.
    absent = str(Path(__file__).with_name("no-such-archive"))
    env["SOARING_PARA_DATA_ROOT"] = str(roots.get("paragliders", absent))
    env["SOARING_DELTA_DATA_ROOT"] = str(roots.get("hang gliders", absent))
    command = ["uv", "run", "--frozen", "--project", str(thesis)]
    command += ["--group", "viewer", "python", "-c", THESIS_PROBE]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    lines = [x for x in result.stdout.splitlines() if x.startswith("PROBE")]
    if result.returncode or not lines:
        raise SystemExit(f"The thesis probe failed:\n{result.stderr}")
    return json.loads(lines[-1][len("PROBE") :])


def _stored(products: Path) -> dict:
    """The fingerprints the saved products record, for the trusted mode."""
    stored = {}

    def metadata(name):
        path = products / name
        if not path.is_file():
            return {}
        with connect(path) as db:
            rows = db.execute("SELECT * FROM metadata").fetchall()
        return dict(rows) if rows and len(rows[0]) == 2 else {"signature": rows[0][0]}

    stored["census"] = metadata("thermal-cells.sqlite3").get("signature")
    stored["routes"] = metadata("route-cells.sqlite3").get("signature")
    stored["activity"] = metadata("thermal-vilpellet-activity.sqlite3").get("signature")
    keys = metadata("thermal-planes.sqlite3").get("segmentation_signatures")
    stored["climbs"] = json.loads(keys).get("vilpellet") if keys else None
    density = products / "route-thermal-duration-vilpellet.npz"
    if density.is_file():
        with np.load(density, allow_pickle=False) as saved:
            stored["density"] = json.loads(str(saved["metadata"])).get("signature")
    group = products / "group-flight-catalog.parquet"
    if group.is_file():
        meta = pq.read_schema(group).metadata or {}
        stored["group"] = meta.get(b"group_signature", b"").decode() or None
    return stored


def _new_fingerprints(disciplines) -> dict:
    """This repository's fingerprints of the copied archive."""
    from xc_viewer import (
        group_flights,
        route_density,
        route_index,
        thermal_index,
    )
    from xc_viewer.thermal_activity import _validated_products
    from xc_viewer.thermal_cache import segmentation_signature

    new = {}
    census = thermal_index.cache_path(disciplines)
    new["census"] = thermal_index.archive_signature(disciplines)
    index = thermal_index.ThermalIndex(
        census, tuple(d.name for d in disciplines), new["census"]
    )
    new["climbs"] = segmentation_signature(index, "vilpellet")
    try:
        new["activity"] = _validated_products(index)[1] if census.is_file() else None
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError):
        new["activity"] = None
    archives = route_index.available_archives()
    new["routes"] = route_index.archive_signature(archives)
    new["density"] = route_density.source_signature(archives)
    routes = route_index.RouteIndex(
        route_index.route_cache_path(archives), tuple(archives), new["routes"]
    )
    new["group"] = group_flights._signature(routes)
    return new


def _restamp_sqlite(path: Path, tables, mapping: dict) -> None:
    """Replace old fingerprints by new ones in the text columns of ``tables``."""
    with connect(path) as db:
        present = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
        for table in tables:
            if table not in present:
                continue
            columns = [
                r[1]
                for r in db.execute(f"PRAGMA table_info({table})")
                if r[2].upper() in ("TEXT", "")
            ]
            for column in columns:
                for old, new in mapping.items():
                    db.execute(
                        f"UPDATE {table} SET {column}=replace({column},?,?) "
                        f"WHERE instr({column},?)>0",
                        (old, new, old),
                    )
        if path.name == "thermal-planes.sqlite3":
            # The Gaussian HMM is gone: keep only the Vilpellet climb key.
            row = db.execute(
                "SELECT value FROM metadata WHERE key='segmentation_signatures'"
            ).fetchone()
            if row:
                keys = {"vilpellet": json.loads(row[0]).get("vilpellet")}
                db.execute(
                    "UPDATE metadata SET value=? WHERE key='segmentation_signatures'",
                    (json.dumps(keys),),
                )


def _restamp_npz(path: Path, mapping: dict) -> None:
    """Rewrite the JSON metadata of a saved raster with the new fingerprint."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    text = str(arrays["metadata"])
    for old, new in mapping.items():
        text = text.replace(old, new)
    arrays["metadata"] = np.array(text)
    temporary = path.with_name(f".{path.name}.copying")
    with temporary.open("wb") as out:
        np.savez(out, **arrays)
    temporary.replace(path)


def _restamp_parquet(path: Path, mapping: dict) -> None:
    """Rewrite a parquet file's schema metadata with the new fingerprint."""
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    value = metadata.get(b"group_signature", b"").decode()
    metadata[b"group_signature"] = mapping.get(value, value).encode()
    temporary = path.with_name(f".{path.name}.copying")
    pq.write_table(
        table.replace_schema_metadata(metadata), temporary, compression="zstd"
    )
    temporary.replace(path)


def _copy_climbs(source: Path, target: Path, old: str, new: str) -> int:
    """Copy only the current Vilpellet climbs, re-keyed; returns the row count."""
    temporary = target.with_name(f".{target.name}.copying")
    temporary.unlink(missing_ok=True)
    with connect(temporary.resolve().as_uri(), uri=True) as db:
        db.execute("""
            CREATE TABLE climbs (
                cache_key TEXT, discipline TEXT, flight_id TEXT,
                ix INTEGER, iy INTEGER, status TEXT, edges BLOB,
                PRIMARY KEY (cache_key, discipline, flight_id, ix, iy)
            )
        """)
        db.execute(
            "ATTACH DATABASE ? AS old", (source.resolve().as_uri() + "?mode=ro",)
        )
        db.execute(
            "INSERT INTO climbs SELECT ?,discipline,flight_id,ix,iy,status,edges "
            "FROM old.climbs WHERE cache_key=?",
            (new, old),
        )
        count = db.execute("SELECT COUNT(*) FROM climbs").fetchone()[0]
        db.commit()
        db.execute("DETACH DATABASE old")
    temporary.replace(target)
    return count


def main() -> int:
    """Copy, verify and re-stamp; print what each tab will find."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--para", type=Path, help="Paraglider archive root")
    parser.add_argument("--hang", type=Path, help="Hang-glider archive root")
    parser.add_argument("--to", type=Path, required=True, help="Data folder to fill")
    parser.add_argument(
        "--thesis-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "soaring-anomalous-transport",
        help="Thesis repository used to check that products are current",
    )
    parser.add_argument(
        "--para-copy",
        type=Path,
        help="Identical copy of the paraglider root on the destination disk, to link",
    )
    parser.add_argument(
        "--hang-copy",
        type=Path,
        help="Identical copy of the hang-glider root on the destination disk, to link",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Hard-link tracks and tables instead of copying (same disk only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only list what would be copied"
    )
    args = parser.parse_args()
    roots = {
        d.name: root
        for d, root in ((PARAGLIDERS, args.para), (HANG_GLIDERS, args.hang))
        if root is not None
    }
    if not roots:
        parser.error("Give --para and/or --hang")
    destination = args.to.expanduser().resolve()
    if any(destination.is_relative_to(r.resolve()) for r in roots.values()):
        parser.error("The data folder must not be inside an archive root")

    plans = {}
    for name, root in roots.items():
        target = destination / DISCIPLINES[name].folder
        plans[name] = _plan(root, target)
        size = sum(a.stat().st_size for a, _ in plans[name])
        climbs = root / PRODUCTS / CLIMBS
        extra = climbs.stat().st_size if climbs.is_file() else 0
        print(
            f"{name}: {len(plans[name]):,} files, {size / 1e9:,.1f} GB "
            f"(+ at most {extra / 1e9:,.1f} GB of climbs)"
        )
    copies = {PARAGLIDERS.name: args.para_copy, HANG_GLIDERS.name: args.hang_copy}
    # The destination may not exist yet: measure the disk it will be created on.
    disk = next(p for p in (destination, *destination.parents) if p.exists())
    wants_links = args.link or any(copies[name] for name in roots)
    if wants_links and not _links_work(destination):
        parser.error(
            f"{destination} is on a disk without hard links (exFAT?): move the "
            "archive copy into place instead of --para-copy/--hang-copy/--link"
        )
    for name in roots:
        for linked in (copies[name], roots[name] if args.link else None):
            if linked and linked.stat().st_dev != disk.stat().st_dev:
                parser.error(f"{linked} is not on the destination disk: no links")
    # Windows has no statvfs; 4 KiB is the usual NTFS cluster.
    block = os.statvfs(disk).f_frsize if hasattr(os, "statvfs") else 4096
    needed = sum(
        _space_needed(
            pairs, link=args.link, root=roots[name], reuse=copies[name], block=block
        )
        for name, pairs in plans.items()
    )
    needed += sum(
        climbs.stat().st_size
        for root in roots.values()
        if (climbs := root / PRODUCTS / CLIMBS).is_file()
    )
    free = shutil.disk_usage(disk).free
    print(f"To write: {needed / 1e9:,.1f} GB, free: {free / 1e9:,.1f} GB")
    if needed > 0.98 * free:
        raise SystemExit("Not enough free space on the destination disk.")
    if args.dry_run:
        return 0

    if args.thesis_repo.is_dir():
        print(f"Checking products with {args.thesis_repo}")
        probe = _probe(args.thesis_repo, roots)
    else:
        print("No thesis repository: every saved product is trusted as current.")
        probe = None

    for name, pairs in plans.items():
        print(f"Copying {name}…", flush=True)
        _copy(pairs, link=args.link, root=roots[name], reuse=copies[name])

    datafolder.use(destination)
    disciplines = [d for d in DISCIPLINES.values() if d.name in roots]
    products = destination / disciplines[0].folder / PRODUCTS
    stored = _stored(products)
    if probe is None:
        current = {k: v for k, v in stored.items() if v}
    else:
        current = {k: v for k, v in stored.items() if v and v == probe.get(k)}
        # The climb cache holds many keys; the thesis names the current one.
        if probe.get("climbs"):
            current["climbs"] = probe["climbs"]
    new = _new_fingerprints(disciplines)
    mapping = {current[k]: new[k] for k in current if new.get(k)}

    climbs = roots[disciplines[0].name] / PRODUCTS / CLIMBS
    if climbs.is_file() and "climbs" in current:
        count = _copy_climbs(
            climbs, products / CLIMBS, current["climbs"], new["climbs"]
        )
        print(f"Climb cache: {count:,} current Vilpellet products")
    for name, tables in STAMPED_TABLES.items():
        if (products / name).is_file():
            _restamp_sqlite(products / name, tables, mapping)
    density = products / "route-thermal-duration-vilpellet.npz"
    if density.is_file():
        _restamp_npz(density, mapping)
    group = products / "group-flight-catalog.parquet"
    if group.is_file():
        _restamp_parquet(group, mapping)

    report = {
        k: "ready" if k in current else ("stale" if stored.get(k) else "absent")
        for k in new
    }
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {name: str(root) for name, root in roots.items()},
        "checked_against_thesis": probe is not None,
        "products": report,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for key, state in report.items():
        print(f"  {key}: {state}")
    print(f"Ready: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
