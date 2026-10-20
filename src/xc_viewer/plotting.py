"""Drawing a trajectory: matplotlib only, no Qt import (see the package docstring).

Kept separate from the Qt widgets: a change to how a line is drawn cannot touch how a
flight is loaded, and this module is importable -- and testable -- without Qt or a
display at all (``matplotlib.use("Agg")``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from xc_viewer.core.style import DISCIPLINE_COLORS, UNCLASSIFIED_COLOR
from xc_viewer.core.style import PHASE_COLORS as _MANUSCRIPT_PHASE_COLORS

from .data import format_dms

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Deterministic PDF metadata, the same convention as scripts/**/generate_*.py:
# committing (or diffing) an exported figure produces a clean diff.
PDF_METADATA = {
    "Creator": "xc_viewer",
    "Producer": "xc_viewer",
    "CreationDate": None,
}

_AXIS_UNITS = {
    "alt": "altitude [m]",
    "E": "East [m]",
    "N": "North [m]",
    "z": "altitude [m]",
}

# The manuscript's phase set, so a label looks the same in the tool a person marks
# it in and in the figure it ends up as, plus the tool-only "unclassified" grey.
PHASE_COLORS = {**_MANUSCRIPT_PHASE_COLORS, "unclassified": UNCLASSIFIED_COLOR}


def make_axes(fig: Figure, *, is_3d: bool) -> Axes:
    """Fresh (2D or 3D) axes on ``fig``, replacing whatever was there.

    Matplotlib cannot switch an existing ``Axes`` between 2D and 3D in place, so every
    redraw that might change dimensionality clears the figure and starts over.

    Args:
        fig: The figure to draw on; it is cleared first.
        is_3d: Whether the axes carry a third dimension.

    Returns:
        The new axes.
    """
    fig.clf()
    return fig.add_subplot(1, 1, 1, projection="3d" if is_3d else None)


def center_message(ax: Axes, text: str, *, is_3d: bool) -> None:
    """A status message centred on ``ax`` (e.g. "pick a flight"), instead of a plot.

    Not a plain ``ax.text(0.5, 0.5, text, transform=ax.transAxes)`` throughout: on an
    ``Axes3D`` that call raises (``Axes3D.text()`` takes ``x, y, z, s`` -- a 3D
    position -- not the 2D ``x, y, s`` a plain ``Axes`` does, so the same call
    silently reinterprets ``text`` as a z-coordinate and then raises on the missing
    ``s``). ``Axes3D.text2D`` is the one already meant for axes-fraction placement
    regardless of the 3D projection, so this dispatches to it there.

    Args:
        ax: The axes to write on.
        text: The message.
        is_3d: Whether the axes carry a third dimension.
    """
    if is_3d:
        ax.text2D(  # type: ignore[attr-defined]
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )
    else:
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )


def plot_trajectory(
    ax: Axes,
    *,
    raw: pd.DataFrame | None = None,
    cleaned: pd.DataFrame | None = None,
    x: str,
    y: str,
    z: str | None = None,
    dms: bool = False,
    raw_color: str = "0.45",
    cleaned_color: str = DISCIPLINE_COLORS["paragliders"],
    color_by: str | None = "segment_id",
    group_by: str | None = None,
    color_map: Mapping[str, str] | None = None,
    raw_label: str = "raw",
    cleaned_label: str = "cleaned",
    mark_raw_endpoints: bool = True,
    cleaned_band: bool = False,
    legend_prefix: str = "",
) -> None:
    """Draw the raw and/or cleaned trajectory on ``ax``, overlaid.

    ``x``/``y``/``z`` name columns of ``raw``/``cleaned`` directly -- ``"lat"``,
    ``"lon"``, ``"alt"`` for the geographic frame, ``"E"``, ``"N"``, ``"z"`` for the
    local ENU one (:mod:`xc_viewer.data`) -- so the caller is free to pick any
    two (2D) or three (3D, when ``z`` is given) of them; passing ``z=None`` draws a 2D
    plot on a plain ``Axes``, passing it draws a 3D one on an ``Axes3D``
    (:func:`make_axes`).

    ``cleaned`` is drawn as one line per distinct value of ``group_by`` (or
    ``color_by`` when ``group_by`` is omitted): a cleaned trajectory can be split
    into several segments
    (``xc_viewer.core.preproc.resample``), and connecting across a segment
    boundary
    would draw a gap the pipeline deliberately left unbridged as if it were measured
    motion. ``color_by`` is a plain column name rather than a fixed segment concept so
    that coloring by flight phase is data-driven rather than embedded in the drawing
    code.  Passing ``color_by="phase"``, ``group_by="phase_run"`` and
    :data:`PHASE_COLORS` colors the semantic phases while keeping non-contiguous runs
    separate.  Pass both grouping arguments as ``None`` to draw one undivided line.
    ``raw`` has no such split: it is drawn as
    a single line, in a lighter, dashed, neutral style so it reads as "underneath" the
    cleaned trajectory rather than a second competing series.

    Args:
        ax: The target axes, from :func:`make_axes` (2D or 3D, matching whether ``z``
            is given).
        raw: The raw track's coordinate table, or ``None`` to omit it.
        cleaned: The cleaned trajectory's coordinate table, or ``None`` to omit it.
        x: Column to plot on the x-axis.
        y: Column to plot on the y-axis.
        z: Column to plot on the z-axis (3D), or ``None`` for a 2D plot.
        dms: If ``True``, an axis literally named ``"lat"`` or ``"lon"`` gets
            degrees-minutes-seconds tick labels
            (:func:`xc_viewer.data.format_dms`)
            instead of decimal degrees.
        raw_color: Line color for the raw trajectory.
        cleaned_color: Line color for the cleaned trajectory (pass a
            ``Discipline.color`` to match the rest of the repo's figures).
        color_by: Column whose values select ``color_map`` entries; also the default
            line-group column when ``group_by`` is omitted.
        group_by: Explicit line-group column.  The phase viewer passes ``phase_run``
            so repeated occurrences of one phase are never bridged.
        color_map: Optional mapping from ``color_by`` values to line colours.  Its
            values become legend labels once each.
        raw_label: Legend label for the raw line.
        cleaned_label: Legend label for the cleaned line(s).
        mark_raw_endpoints: If ``True`` and ``raw`` is drawn, mark its very first and
            very last fix with distinct markers -- so the raw line reads as running
            the whole recorded track end to end, not a subset it's easy to mistake
            for the full one at a glance.
        cleaned_band: Draw the cleaned trajectory as a wide translucent band without
            fix markers, so a thin line drawn over it stays readable: the viewer
            overlays two segmentations of one flight this way.
        legend_prefix: Text put before each ``color_map`` legend label.
    """
    if raw is not None and len(raw):
        _draw(
            ax,
            raw,
            x,
            y,
            z,
            color=raw_color,
            ls="--",
            lw=1.0,
            alpha=0.85,
            label=raw_label,
        )
        if mark_raw_endpoints:
            _mark_endpoint(
                ax,
                raw,
                x,
                y,
                z,
                row=0,
                marker="^",
                color="#2a9d3f",
                label="raw start",
            )
            _mark_endpoint(
                ax,
                raw,
                x,
                y,
                z,
                row=-1,
                marker="s",
                color="#c1272d",
                label="raw end",
            )
    if cleaned is not None and len(cleaned):
        _draw_grouped(
            ax,
            cleaned,
            x,
            y,
            z,
            color=cleaned_color,
            ls="-",
            lw=7.0 if cleaned_band else 1.4,
            alpha=0.35 if cleaned_band else 1.0,
            label=cleaned_label,
            color_by=color_by,
            group_by=color_by if group_by is None else group_by,
            color_map=color_map,
            marker="." if color_map is not None and not cleaned_band else None,
            legend_prefix=legend_prefix,
        )
    ax.set_xlabel(_axis_label(x, dms))
    ax.set_ylabel(_axis_label(y, dms))
    _maybe_dms_ticks(ax.xaxis, x, dms)
    _maybe_dms_ticks(ax.yaxis, y, dms)
    if z is not None:
        ax.set_zlabel(_axis_label(z, dms))  # type: ignore[attr-defined]
        _maybe_dms_ticks(ax.zaxis, z, dms)  # type: ignore[attr-defined]
    if (raw is not None and len(raw)) or (cleaned is not None and len(cleaned)):
        ax.legend(fontsize=8)


def save_pdf(fig: Figure, path: str | Path) -> None:
    """Save ``fig`` as a vector PDF, with the repo's deterministic metadata."""
    fig.savefig(path, metadata=PDF_METADATA, bbox_inches="tight")


def _axis_label(col: str, dms: bool) -> str:
    """The axis label for one plotted column, honouring the DMS toggle."""
    if col == "lat":
        return "latitude" if dms else "latitude [deg]"
    if col == "lon":
        return "longitude" if dms else "longitude [deg]"
    return _AXIS_UNITS.get(col, col)


def _maybe_dms_ticks(axis, col: str, dms: bool) -> None:
    """Switch one matplotlib axis to degrees-minutes-seconds tick labels, if asked."""
    if not dms or col not in ("lat", "lon"):
        return
    from matplotlib.ticker import FuncFormatter

    kind: Literal["lat", "lon"] = "lat" if col == "lat" else "lon"
    axis.set_major_formatter(FuncFormatter(lambda value, _pos: format_dms(value, kind)))
    axis.set_tick_params(labelrotation=30)


def _draw_grouped(
    ax,
    table,
    x,
    y,
    z,
    *,
    color,
    ls,
    lw,
    alpha,
    label,
    color_by,
    group_by,
    color_map,
    marker,
    legend_prefix,
):
    """Draw ``table`` as one line per ``group_by`` value, coloured by ``color_by``.

    Falls back to a single undivided line when ``group_by`` is absent or missing from
    ``table``; see :func:`plot_trajectory` for what each grouping is used for.
    """
    if group_by is None or group_by not in table.columns:
        _draw(ax, table, x, y, z, color=color, ls=ls, lw=lw, alpha=alpha, label=label)
        return
    first = True
    labelled_values: set[str] = set()
    for _, segment in table.groupby(group_by, sort=False):
        value = (
            str(segment[color_by].iloc[0])
            if color_by is not None and color_by in segment.columns
            else None
        )
        mapped_color = color_map.get(value, color) if color_map is not None else color
        if color_map is not None and value is not None:
            line_label = (
                legend_prefix + value if value not in labelled_values else "_nolegend_"
            )
            labelled_values.add(value)
        else:
            line_label = label if first else "_nolegend_"
        # Close each colour run at the next native vertex of the same physical
        # track. This colours every edge, including phase changes, without joining
        # disconnected segments or nonconsecutive occurrences of the same phase.
        if group_by == "phase_run" and "track_run" in table.columns:
            last_position = table.index.get_loc(segment.index[-1])
            if last_position + 1 < len(table):
                following = table.iloc[[last_position + 1]]
                if following["track_run"].iloc[0] == segment["track_run"].iloc[-1]:
                    import pandas as pd

                    segment = pd.concat([segment, following])
        _draw(
            ax,
            segment,
            x,
            y,
            z,
            color=mapped_color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            label=line_label,
            marker=marker,
        )
        first = False


def _mark_endpoint(ax, table, x, y, z, *, row, marker, color, label):
    """Scatter a single labelled marker at one row (``0`` or ``-1``) of ``table``."""
    point = table.iloc[[row]]
    kwargs = {
        "marker": marker,
        "color": color,
        "s": 45,
        "zorder": 5,
        "label": label,
        "edgecolors": "white",
        "linewidths": 0.6,
    }
    if z is None:
        ax.scatter(point[x].to_numpy(), point[y].to_numpy(), **kwargs)
    else:
        zs = point[z].to_numpy()
        ax.scatter(point[x].to_numpy(), point[y].to_numpy(), zs, **kwargs)


def _draw(ax, table, x, y, z, *, color, ls, lw, alpha, label, marker=None):
    """Plot one unbroken line of ``table``, in 2D or 3D depending on ``z``."""
    xs, ys = table[x].to_numpy(), table[y].to_numpy()
    if z is None:
        ax.plot(
            xs,
            ys,
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            label=label,
            marker=marker,
            markersize=2.5,
        )
    else:
        zs = table[z].to_numpy()
        ax.plot(
            xs,
            ys,
            zs,
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            label=label,
            marker=marker,
            markersize=2.5,
        )
