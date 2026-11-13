"""Save a PNG of one group episode in the 3D scene of the Group flights tab, without a click.

    uv run python scripts/episode_snapshot.py 9052 out.png --at 1500 --azimuth -40 --elevation 30

The scene is the viewer's own, drawn in a window that stays open for a few seconds: the
OpenGL view needs a screen. Terrain and imagery come from IGN and are cached with the
route index; with --cache they go to another folder, so that the data folder is not
written (the route index is copied there once).
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SURFACES = {"elevation": 0, "aerial": 1, "relief": 2}


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode", type=int, help="episode_id of the groups/ tables")
    p.add_argument("output", type=Path, help="PNG to write")
    p.add_argument("--root", type=Path, help="data folder (default: the environment's)")
    p.add_argument("--cache", type=Path, help="folder for the index copy and the terrain")
    p.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"),
                   help="UTC seconds: shorten the episode to them (the replay adds 10 minutes each side)")
    p.add_argument("--at", type=float, default=0, help="seconds after the episode starts")
    p.add_argument("--utc", type=float, help="UTC second of the instant, instead of --at")
    p.add_argument("--azimuth", type=float)
    p.add_argument("--elevation", type=float)
    p.add_argument("--distance", type=float, help="camera distance in m")
    p.add_argument("--zoom", type=float, help="multiply the camera distance by this")
    p.add_argument("--centre", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="view centre in m: east, north from the scene centre, altitude")
    p.add_argument("--follow", action="store_true", help="centre on the members")
    p.add_argument("--extent", choices=("whole", "flown", "trail"), default="whole",
                   help="tracks whole, up to the instant, or the bright trail only")
    p.add_argument("--trail", type=int, help="minutes of bright trail")
    p.add_argument("--colour-by", choices=("group", "pilot", "episode"), default="pilot")
    p.add_argument("--track-by", choices=("pilot", "episode"), default="pilot")
    p.add_argument("--numbers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--width", type=float, help="track width in px")
    p.add_argument("--vertical", type=float, help="vertical exaggeration")
    p.add_argument("--surface", choices=SURFACES, default="relief")
    p.add_argument("--no-terrain", action="store_true")
    p.add_argument("--no-markers", action="store_true", help="hide the start and end marks")
    p.add_argument("--aerial-side", type=int, help="pixels per side of the photo of the whole scene")
    p.add_argument("--detail", type=float, nargs=3, metavar=("SIDE", "SNAP", "PIXELS"),
                   help="finer photo near the view centre: side in m, step of its centre in m, pixels (at most 5000)")
    p.add_argument("--mute", type=float, help="keep this share of the colour of the photo")
    p.add_argument("--outline", type=float, help="dark outline of the tracks in px")
    p.add_argument("--arrows", type=int, nargs=2, metavar=("FLIGHT", "EVERY_M"),
                   help="arrowheads along the track of a member, one per EVERY_M of path")
    p.add_argument("--arrow-px", type=int, default=14)
    p.add_argument("--background", default="#ffffff", help="colour of the view")
    p.add_argument("--hide", type=int, nargs="*", default=[], metavar="FLIGHT",
                   help="members not to draw, by flight number")
    p.add_argument("--hide-panels", action="store_true", help="show the 3D view alone")
    p.add_argument("--size", type=int, nargs=2, default=(1600, 1000), metavar=("W", "H"),
                   help="pixels of the PNG")
    p.add_argument("--settle", type=float, default=3, help="seconds before the image")
    return p.parse_args()


def spin(seconds):
    """Let Qt work for `seconds`."""
    from PyQt6.QtWidgets import QApplication

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def mute(photo, share):
    """The RGB uint8 `photo` with `share` of its colour."""
    grey = photo.mean(axis=2, keepdims=True)
    return np.clip(grey + share * (photo - grey), 0, 255).astype(np.uint8)


def choose(combo, data):
    combo.setCurrentIndex(combo.findData(data))


def main():
    args = parse()
    from xc_viewer import datafolder
    from xc_viewer.app import configure_graphics, use_system_certificates

    if args.root is not None:
        datafolder.use(args.root)
    root = datafolder.current()
    if root is None:
        sys.exit("Give --root, the data folder.")
    if args.cache is not None:
        from xc_viewer.route_index import route_cache_path

        saved, args.cache = route_cache_path(), args.cache.resolve()
        args.cache.mkdir(parents=True, exist_ok=True)
        if not (args.cache / saved.name).exists():
            shutil.copy(saved, args.cache / saved.name)
        os.environ["XC_VIEWER_CACHE_DIR"] = str(args.cache)
    use_system_certificates()
    configure_graphics()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QVector3D
    from PyQt6.QtWidgets import QApplication

    from xc_viewer.group_episodes import load_episode_scene, load_tables
    from xc_viewer.route_index import load_saved_index
    from xc_viewer.widgets.english import use_english
    from xc_viewer.widgets.group_scene import EpisodeScene
    from xc_viewer.widgets.theme import apply_theme

    app = QApplication(sys.argv[:1])
    use_english()
    apply_theme(app)
    index = load_saved_index()
    if index is None:
        sys.exit("No route index: open the Group flights tab once and prepare it.")
    tables = load_tables(root / "groups")
    if args.clip is not None:
        episodes = tables.episodes
        episodes.loc[episodes.episode_id == args.episode, ["t_start", "t_end"]] = args.clip
    data = load_episode_scene(
        index, tables, args.episode, progress=lambda m: print(m, file=sys.stderr),
        aerial_side=args.aerial_side,
    )
    if args.mute is not None and data[0].aerial is not None:
        data[0].aerial = mute(data[0].aerial, args.mute)
    scene = EpisodeScene()
    scene.resize(1500, 950)
    scene.show()
    scene.set_episode(*data)
    ep = data[1]
    spin(1)  # the scene fits its camera once its widgets have their size

    if args.detail is not None:
        from xc_viewer.route_scene import load_detail

        detail = scene._detail
        detail.side, detail.snap = args.detail[:2]
        pixels = int(args.detail[2])

        def loader(*a):
            patch = load_detail(*a, pixels=pixels)
            if args.mute is not None:  # the finer photo is muted like the overview
                patch.photo = mute(patch.photo, args.mute)
            return patch

        detail._loader = loader
        detail.attach(data[0], scene._view, scene._surface)
    if args.hide_panels:
        scene.hide_panels()
    view = scene._view
    view.setBackgroundColor(args.background)
    scene._terrain.setChecked(not args.no_terrain)
    scene._annotations.setVisible(not args.no_markers)
    scene._surface_mode.setCurrentIndex(SURFACES[args.surface])
    choose(scene._colour_by, args.colour_by)
    choose(scene._track_by, args.track_by)
    choose(scene._extent, args.extent)
    scene._numbers.setChecked(args.numbers)
    scene._follow.setChecked(args.follow)
    if args.trail is not None:
        scene._trail.setValue(args.trail)
    if args.width is not None:
        scene._line_width.setValue(args.width)
    if args.vertical is not None:
        scene._vertical.setValue(args.vertical)
    for k, flight in enumerate(ep.members.flight):
        if int(flight) in args.hide:
            scene._table.item(k, 0).setCheckState(Qt.CheckState.Unchecked)
    if args.width is not None:
        for line in scene._trails:  # the trail has a width of its own
            line.width = args.width
    if args.outline is not None:
        for line in (*scene._trails, *scene._flown, *(l for ls in scene._lines for l in ls)):
            line.outline = args.outline
    if args.arrows is not None:
        scene.show_arrows(args.arrows[0], args.arrows[1], args.arrow_px)
    scene.seek(args.utc if args.utc is not None else ep.window[0] + args.at)

    camera = {}
    if args.centre is not None:
        x, y, z = args.centre
        camera["pos"] = QVector3D(x, y, z * scene._vertical.value())
    for key in ("azimuth", "elevation", "distance"):
        if getattr(args, key) is not None:
            camera[key] = getattr(args, key)
    if args.zoom is not None:
        camera["distance"] = camera.get("distance", view.opts["distance"]) * args.zoom
    view.setCameraPosition(**camera)
    spin(args.settle)
    detail = scene._detail  # the finer terrain loads where the camera looks
    end = time.monotonic() + 90
    while detail._future is not None and time.monotonic() < end:
        spin(0.5)
    spin(1)
    ratio = scene.devicePixelRatio()  # the PNG has this many pixels per point
    scene.save_image(args.output, [round(v / ratio) for v in args.size])
    inside = np.vstack([p for t in ep.tracks.values() for p in t.pieces])
    lo, hi = inside[:, 1:].min(axis=0), inside[:, 1:].max(axis=0)  # camera hints
    first = np.mean([t.pieces[0][0, 1:3] for t in ep.tracks.values()], axis=0)
    last = np.mean([t.pieces[-1][-1, 1:3] for t in ep.tracks.values()], axis=0)
    heading = np.degrees(np.arctan2(*(last - first)[::-1]))
    print(f"mean heading {heading:.0f} deg: azimuth {heading - 90:.0f} sees the flight go to the right", file=sys.stderr)
    print(f"tracks span x {lo[0]:.0f} {hi[0]:.0f}, y {lo[1]:.0f} {hi[1]:.0f}, z {lo[2]:.0f} {hi[2]:.0f} m", file=sys.stderr)
    print("members (number: flight):", ", ".join(f"{k + 1}: {f}" for k, f in enumerate(ep.members.flight)), file=sys.stderr)
    if scene._located:
        c = np.mean(list(scene._located.values()), axis=0)
        print(f"members at the instant: centre {c[0]:.0f} {c[1]:.0f} {c[2]:.0f} m", file=sys.stderr)
    print(f"{args.output}: {args.size[0]} x {args.size[1]} px (screen ratio {ratio:g})")
    scene.shutdown()
    scene.close()


if __name__ == "__main__":
    main()
