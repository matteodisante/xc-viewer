"""The Sources & methods tab: where the data come from, and how every number is made.

This page holds the exact definitions. Each screen's Info window says what the screen
shows and links here for the details; its How to use window gives the steps (see
:mod:`.help_text`). Numbers are read from the configuration and the code's constants
when the tab is first opened, so the page states what the viewer actually does.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..geography import REGIONS as LAUNCH_REGIONS
from ..route_index import MAX_FLIGHTS, ROUTE_CELL_M
from ..thermal_geometry import CELL_M
from ..thermal_imagery import LAYERS
from ..thermal_imagery import SERVICE as IGN_WMS
from ..thermal_regions import PAD_DEG
from ..thermal_regions import REGIONS as DENSITY_REGIONS
from ..thermal_ridges import DATASET_URL as RGE_ALTI_URL
from ..thermal_ridges import LAYER as DEM_LAYER
from ..thermal_ridges import PARAMETERS as DEM_WINDOW
from ..thermal_time import BASE_M
from .info_browser import info_browser
from .source_notes import (
    BD_ORTHO,
    CONTOURS,
    FLIGHT_SOURCE_HTML,
    HILLSHADE_URL,
    LICENCE_OUVERTE,
    NATURAL_EARTH,
    NATURAL_EARTH_WORLD,
    ORTHO_DATES,
    PLAN_IGN,
    VILPELLET_PAPER,
)

SECTIONS = [
    ("sources", "Flight data"),
    ("maps", "Maps and terrain"),
    ("processing", "Processing a flight"),
    ("phases", "Flight phases"),
    ("time", "Dates, times and altitudes"),
    ("launches", "Launch map"),
    ("cells", "Cells and ranking"),
    ("crossings", "Plane crossings"),
    ("hours", "Climbing hours per km²"),
    ("routes", "Routes"),
    ("groups", "Launch groups"),
]
_TITLES = dict(SECTIONS)


def _heading(anchor: str) -> str:
    return f'<a name="{anchor}"></a><h3>{_TITLES[anchor]}</h3>'


def _back() -> str:
    return '<p><a href="#contents">Back to the contents</a></p>'


def _lon(value: float) -> str:
    return f"{abs(value):g}&deg;{'W' if value < 0 else 'E'}"


def _contents() -> str:
    links = " &nbsp;&middot;&nbsp; ".join(
        f'<a href="#{anchor}">{title.replace(" ", "&nbsp;")}</a>'
        for anchor, title in SECTIONS
    )
    return f"""
<a name="contents"></a><h2>Sources &amp; methods</h2>
<p class="role">Where the data come from, and exactly how each number in the viewer
is computed. Every screen also has two buttons: <b>Info</b> says what that screen
shows and how to read it, and <b>How to use</b> lists the steps to operate it.</p>
<p>{links}</p>
"""


def _sources() -> str:
    return f"""
{_heading("sources")}
<p>The flights are those declared to the FFVL's cross-country contest, the Coupe
F&eacute;d&eacute;rale de Distance (CFD): {FLIGHT_SOURCE_HTML}.</p>
<ul>
<li><b>IGC files</b>, written by each pilot's flight recorder: a header with the date,
then one line per fix with the UTC time of day, latitude, longitude, GNSS altitude and
pressure altitude. Every position, time and altitude in the viewer comes from
them.</li>
<li><b>The catalogue</b>, one entry per declared flight: pilot, date, take-off and
landing sites, department, flight type, wing model and class, club, declared distance
and duration. It serves to find and describe flights. It comes from the pilots'
declarations and can contain mistakes.</li>
<li><b>Prepared results</b>, computed once from the whole archive and stored in the
data folder: cleaned tracks and their phase labels, the twelve cells with their
crossings, the climbing-time grids, the route index and the launch times. With them
the tabs open quickly and work offline.</li>
</ul>
{_back()}
"""


def _maps() -> str:
    grid = int(DEM_WINDOW["grid_m"])
    side = int(CELL_M // grid)
    licence = f'<a href="{LICENCE_OUVERTE}">Licence Ouverte 2.0</a>'
    return f"""
{_heading("maps")}
<table border="1" cellspacing="0" cellpadding="8" width="100%">
<tr><th width="30%">Source</th><th>Used for</th><th width="20%">Licence</th></tr>
<tr><td><a href="{RGE_ALTI_URL}">IGN RGE ALTI</a>, ground elevation<br>
<small>layer {DEM_LAYER}</small></td>
<td>Lowest and highest ground of each Thermal planes cell: {side} &times; {side}
values {grid} m apart inside the square, taken before any smoothing, with complete
coverage required. The 3D terrain of a cell. The terrain of Routes and Group flights,
at most 600 values per side and at least 100 m apart.</td><td>&copy; IGN,
{licence}</td></tr>
<tr><td><a href="{PLAN_IGN}">IGN Plan IGN v2</a><br>
<small>layer {LAYERS["colour"]}</small></td>
<td>The <b>Colour map</b> background.</td><td>&copy; IGN, {licence}</td></tr>
<tr><td><a href="{BD_ORTHO}">IGN BD ORTHO</a><br>
<small>layer {LAYERS["aerial"]}</small></td>
<td>The <b>Aerial photo</b> background, and the photo draped on 3D terrain. The
dates of the photos come from the <a href="{ORTHO_DATES}">BD ORTHO mosaic
graph</a>.</td><td>&copy; IGN, {licence}</td></tr>
<tr><td><a href="{CONTOURS}">IGN elevation contours</a><br>
<small>layers {LAYERS["topography"].replace(",", ", ")}</small></td>
<td>The <b>Topography + contours</b> background: the colour map with contour
lines.</td><td>&copy; IGN, {licence}</td></tr>
<tr><td><a href="{HILLSHADE_URL}">Esri World Hillshade</a></td>
<td>The <b>Shaded relief</b> background.</td><td>&copy; Esri and its data
contributors, Esri terms of use</td></tr>
<tr><td>Natural Earth admin-0 countries,
<a href="{NATURAL_EARTH_WORLD}">1:50 million</a> and
<a href="{NATURAL_EARTH}">1:10 million</a></td>
<td>Borders and coastlines: the Map tab and the route location map (1:50 million),
the France map of Thermal planes (1:10 million). Bundled with the viewer.</td>
<td>Public domain</td></tr>
</table>
<p>IGN images come from the <a href="{IGN_WMS}">IGN G&eacute;oplateforme</a>, at
these sizes:</p>
<ul>
<li><b>Thermal planes:</b> one image per cell and background, saved in the data
folder: {4000:,} &times; {4000:,} pixels ({CELL_M / 4000:g} m per pixel), and
600 &times; 600 for the shaded relief.</li>
<li><b>Thermal density:</b> the visible area is requested at up to 2,048 pixels per
side and at most 1.25 m per pixel; the saved cell images are reused where they fit.
Downloads are kept in a 512 MiB cache in the data folder.</li>
<li><b>Routes, Group flights and 3D terrain:</b> the aerial photo draped on the
terrain is at most 3,072 pixels on its longest side.</li>
</ul>
<p>A pixel size is a sampling step, not a positional accuracy, and photos were taken
on other dates than the flights.</p>
{_back()}
"""


def _processing() -> str:
    from ..core.config import load_preproc_config

    cfg = load_preproc_config()
    fix, alt, trim = cfg.fix, cfg.alt_channel, cfg.trimming
    flight, sampling, savgol = cfg.flight, cfg.sampling, cfg.savgol
    speeds = fix.max_horizontal_speed_mps
    return f"""
{_heading("processing")}
<p>Every flight of the archive went through these seven steps, with the same
settings; Trajectory repeats them for the flight it opens. A flight rejected at any
step keeps a record of the step and the reason, which Trajectory and the catalogue
search show.</p>
<ol>
<li><b>Altitude channel.</b> The GNSS altitude is used for every flight. A flight is
rejected when fewer than {alt.gnss_present_min:.0%} of its fixes carry one, or when it
varies by less than {alt.gnss_min_range_m:g} m over the whole flight (a stuck
sensor).</li>
<li><b>Fix cleaning.</b> Fixes whose time goes backwards are removed. So are
position jumps that would need a horizontal speed above
{speeds["paragliders"]:g} m/s for paragliders or {speeds["hang gliders"]:g} m/s for
hang gliders, and runs where the position stays stuck (repeated, or within
{fix.frozen_eps_m:g} m) for {fix.frozen_tau_s:g} s or more when a second sign confirms a
frozen receiver; the track is split where such a run was. Altitudes outside
{fix.min_altitude_m:g} to {fix.max_altitude_m:,.0f} m, or implying a vertical speed
above {fix.max_vertical_speed_mps:g} m/s (median over {fix.vz_window_s:g} s), are
invalidated. A Hampel test ({fix.hampel_window_s:g} s window, {fix.hampel_k:g} robust
standard deviations) flags outliers along the way.</li>
<li><b>Ground trimming.</b> Free flight starts at the first moment from which the
horizontal speed stays above {trim.takeoff_speed_mps:g} m/s for
{trim.sustained_s:g} s, and ends at the last such moment; the time before and after is
cut, and the clock restarts at the start. Stops on the ground inside the flight,
slow for {trim.interior_ground_s / 60:g} minutes or more and flat in the pressure
altitude, are cut too. A flight is rejected when cleaning had to remove or rebuild
more than {fix.integrity_max_fraction:.0%} of its airborne fixes.</li>
<li><b>Flight filter.</b> A flight is kept when it lasts from
{flight.min_duration_s / 60:g} minutes to {flight.max_duration_s / 3600:g} hours,
covers at least {flight.min_path_km:g} km of path, spans at least
{flight.min_alt_range_m:g} m between its lowest and highest altitude, and its mean
speed and extent are possible within the speed limit above.</li>
<li><b>Local coordinates.</b> Positions are converted to metres East and North
(ENU) on the plane tangent to the Earth at the first fix of free flight, using the
WGS 84 ellipsoid. Altitudes stay the recorded GNSS altitudes.</li>
<li><b>Even time steps.</b> Each flight keeps its own logging interval (the median
step). Irregular fixes are moved onto a regular grid at that interval, short holes
are filled by interpolation and flagged. A hole longer than the smaller of
{sampling.max_gap_factor:g} intervals and the larger of
{sampling.max_gap_seconds:g} s and 2 intervals is not bridged: the track is split there
into segments. Segments shorter than {sampling.min_segment_duration_s:g} s, or with
more than {sampling.max_missing_fraction:.0%} filled points, are left out.</li>
<li><b>Smoothing.</b> A Savitzky&ndash;Golay filter (polynomial of order
{savgol.polyorder}, window of about {savgol.tau_c_horizontal_s:g} s) smooths
positions and gives velocities, separately in the horizontal and in the
vertical.</li>
</ol>
{_back()}
"""


def _phases() -> str:
    from ..core.vilpellet import load_vilpellet_config

    cfg = load_vilpellet_config()
    feat, gate = cfg.features, cfg.eligibility
    alpha = cfg.alpha_straight_rad
    half = feat.persistence_fixes // 2
    agree = math.ceil(feat.beta_persistence * half)
    return f"""
{_heading("phases")}
<p>The labels climb, search and transition come from the segmentation of
J.&nbsp;Vilpellet, A.&nbsp;Darmon and M.&nbsp;Benzaquen, <a href="{VILPELLET_PAPER}">
<i>From Random Walks to Thermal Rides: Universal Anomalous Transport in Soaring
Flights</i></a> (2026). It is ported from the authors' code and uses their fitted
parameters: nothing is refitted here.</p>
<ol>
<li><b>Which flights.</b> Only tracks logged at a mean interval of at most
{gate.max_mean_dt_s:g} s, with at least {gate.min_fixes:,} fixes and strictly
increasing times. The others stay unclassified, because the windows below count
fixes, not seconds.</li>
<li><b>Three yes/no features per fix</b>, computed on the cleaned track after a
further Gaussian smoothing of the positions over {feat.position_smoothing_fixes}
fixes, each over a window of {feat.persistence_fixes} fixes centred on the fix:
<ul>
<li><i>straight</i>: the turn from one fix to the next, averaged over
{feat.turn_smoothing_fixes} fixes, stays below {alpha["paragliders"]:.3f} rad
(paragliders) or {alpha["hang gliders"]:.3f} rad (hang gliders) on more than
{feat.beta_persistence:.0%} of the window;</li>
<li><i>rising</i>: the mean vertical speed over the window is positive;</li>
<li><i>one turn direction</i>: of the {half} sharpest turns in the window, at least
{agree} go the same way.</li>
</ul></li>
<li><b>Model.</b> A hidden Markov model with three states, each producing the eight
possible feature combinations with its own probabilities, finds the most likely
sequence of states (Viterbi algorithm). The authors fitted it on 2019 paraglider
flights from four Alpine take-off sites. It is applied unchanged to every flight;
hang gliders differ only by their straightness threshold.</li>
<li><b>Cleaning the labels.</b> A rolling majority vote over {cfg.majority_fixes}
fixes removes short flickers.</li>
<li><b>Climb run.</b> An unbroken sequence of climb fixes within one piece of
track.</li>
</ol>
<p>The authors ran the method on raw GNSS tracks; here it runs on the cleaned
tracks, so positions are smoothed twice. Short runs are the least reliable, as the
authors warn.</p>
{_back()}
"""


def _time() -> str:
    return f"""
{_heading("time")}
<ul>
<li><b>Clock.</b> A fix's UTC time is the date in the IGC header, plus the time of
the first accepted fix, plus the trimming offset, plus the elapsed time of the
cleaned track. It is shown in French local time (Europe/Paris), with CET in winter
and CEST in summer. Catalogue dates only help to find the files.</li>
<li><b>Missing clocks.</b> A flight whose UTC time cannot be recovered still counts
where time plays no part, but never enters a date or hour selection.</li>
<li><b>Hour windows.</b> The start is included and the end excluded: 12:00 for 30
minutes means from 12:00 up to, not including, 12:30. An end of 00:00 means midnight
at the end of the day. When clocks go back in October, both occurrences of the
repeated hour match.</li>
<li><b>Same dates every year.</b> A season whose end comes before its start runs
into the next year; a limit on 29 February becomes 28 February in common years.</li>
<li><b>Altitudes.</b> They are the recorders' GNSS altitudes, whose height reference
is not the same as that of the IGN elevations. A height above the ground (z) is
therefore approximate.</li>
</ul>
{_back()}
"""


def _launches() -> str:
    from .map_view import (
        _MAX_CELL_DEG,
        _MIN_CELL_DEG,
        _TARGET_CELLS_ACROSS,
    )

    boxes = "; ".join(
        f"{name}: {_lon(w)}&ndash;{_lon(e)}, {s:g}&ndash;{n:g}&deg;N"
        for name, (w, s, e, n) in LAUNCH_REGIONS.items()
    )
    return f"""
{_heading("launches")}
<ul>
<li><b>Launch point:</b> the first fix of free flight of each kept flight, as found by
ground trimming.</li>
<li><b>Counting grid:</b> square cells in degrees of longitude and latitude, as wide
as the visible longitude span divided by {_TARGET_CELLS_ACROSS}, but no narrower than
{_MIN_CELL_DEG:g}&deg; and no wider than {_MAX_CELL_DEG:g}&deg;. The grid is rebuilt
after each zoom or pan. The colour scale runs from 1 to the largest count in
view.</li>
<li><b>Regions</b> are rectangles: {boxes}.</li>
<li><b>Terrain categories</b> are read from the GNSS altitude of the launch point;
Thermal planes applies the same limits to the highest ground of a cell.</li>
</ul>
{_back()}
"""


def _cells() -> str:
    cell = f"{CELL_M / 1000:g} &times; {CELL_M / 1000:g} km"
    return f"""
{_heading("cells")}
<ol>
<li><b>Grid:</b> fixed {cell} squares in the French Lambert-93 projection
(EPSG:2154), over metropolitan France.</li>
<li><b>Count:</b> continuous Vilpellet climb runs, over all dates and both
disciplines. A run counts once in a cell, even if it leaves and comes back; two runs of
one flight count twice; length and number of fixes add no weight. Flights are labelled
whole before being cut to the cell, so a climb counts even when the take-off was
elsewhere.</li>
<li><b>Category:</b> from the highest of the cell's elevation values (see Maps and
terrain). A single high sample is enough to lift a cell into a higher category.</li>
<li><b>Ties</b> go to the lower grid column, then row. There is no regional quota and
no minimum spacing, so neighbouring cells can all be selected.</li>
<li><b>Fixed list:</b> the selection never changes with the dates, the height or the
background.</li>
</ol>
{_back()}
"""


def _crossings() -> str:
    return f"""
{_heading("crossings")}
<ul>
<li><b>Reference:</b> the lowest of the elevation values of the cell (see Maps and
terrain). The plane lies at H = reference + z. Neighbouring cells are cut at the same
absolute H.</li>
<li><b>Edges:</b> two consecutive cleaned fixes of one climb run, at most 1.5 times
the median step of their piece of track apart in time. Gaps, ends of runs and other
phases give no edge.</li>
<li><b>Crossing:</b> for an edge from altitude z<sub>0</sub> to z<sub>1</sub>,
f = (H &minus; z<sub>0</sub>) / (z<sub>1</sub> &minus; z<sub>0</sub>). When
0 &le; f &le; 1 the crossing lies at (x, y, t) = (x<sub>0</sub>, y<sub>0</sub>,
t<sub>0</sub>) + f &times; (&Delta;x, &Delta;y, &Delta;t), in Lambert-93 and UTC.
Upward and downward crossings both count, a crossing exactly at a shared fix counts
once, and level edges are skipped. Example: H = 1,000 m with fixes at 997 and
1,003 m gives f = 0.5, halfway in space and time.</li>
<li><b>Saved levels:</b> crossings are stored every 10 m above the reference, up to
the highest supported altitude in the cell, plus that exact top level, so the same z
always gives the same dots, whatever the height increment.</li>
<li><b>Busiest summer day:</b> the June&ndash;August day on which the most distinct
flights left crossings in the cell.</li>
<li><b>Accuracy:</b> straight edges cut the corners of curved paths, more so with
sparse fixes and tight turns.</li>
</ul>
{_back()}
"""


def _hours() -> str:
    regions = "; ".join(
        f"{name}: {_lon(w)}&ndash;{_lon(e)}, {s:g}&ndash;{n:g}&deg;N"
        for name, (w, e, s, n) in DENSITY_REGIONS.items()
    )
    pixel = f"{BASE_M:g} &times; {BASE_M:g} m"
    return f"""
{_heading("hours")}
<p>For each ground pixel B, D(B) = seconds of climbing above B / (3600 &times; area of
B in km&sup2;), in hours per km&sup2;.</p>
<ol>
<li><b>Edges:</b> defined as for plane crossings, over every saved Vilpellet climb run
of the archive.</li>
<li><b>Position:</b> fixes are projected to Lambert-93; every altitude is pooled.</li>
<li><b>Sharing out the time:</b> along each edge the glider is assumed to move in a
straight line at constant speed, and the edge's seconds go to the {pixel} pixels it
crosses, in proportion to the length inside each. An edge that does not move gives all
its seconds to its pixel.</li>
<li><b>Colour scale limits:</b> from 0.01 h/km&sup2; to the larger of 1 and the
largest {BASE_M:g} m value; smaller positive values take the lightest colour.</li>
</ol>
<ul>
<li><b>Regions</b> are rectangles widened by {PAD_DEG:g}&deg; on every side, cut from
one national grid: {regions}.</li>
<li><b>Cells</b> are the twelve Thermal planes squares, computed from their saved
climb edges.</li>
<li><b>Routes and Group flights</b> drape the national grid on their terrain, with
pixels of 250, 500, 1,000 or 2,000 m, the finest that keeps the scene within 1,600
pixels across.</li>
</ul>
{_back()}
"""


def _routes() -> str:
    cell = f"{ROUTE_CELL_M / 1000:g} &times; {ROUTE_CELL_M / 1000:g} km"
    return f"""
{_heading("routes")}
<ol>
<li><b>Endpoints:</b> each flight's first and last cleaned fixes, converted to
Lambert-93 through the flight's own local origin and assigned to fixed {cell} cells.
A &rarr; B and B &rarr; A are different pairs.</li>
<li><b>Ranking:</b> the elapsed time includes gaps; ties go by discipline, then
flight number.</li>
<li><b>Reduction:</b> with N matching flights and N above {MAX_FLIGHTS}, the ranks
drawn are 1&ndash;5, N &minus; 4 to N, and {MAX_FLIGHTS - 10} ranks evenly spaced from 6
to N &minus; 5. With fewer than ten flights, the fastest and slowest five
overlap.</li>
<li><b>Departure window:</b> keeps departures from the start time (included) to the
start plus the length (excluded), on one French calendar day; the window cannot cross
midnight. It is applied before ranking and reduction. The days offered, with their
counts, cover every flight of the pair; flights without a clock are left out while the
window is active.</li>
</ol>
{_back()}
"""


def _groups() -> str:
    return f"""
{_heading("groups")}
<ol>
<li><b>Grid:</b> Lambert-93 cells over metropolitan France; destinations can lie
outside it.</li>
<li><b>Windows:</b> days are French calendar days, and a window includes its end.
Windows never overlap or chain, a flight alone still uses up its window, and changing
the window length can move later boundaries.</li>
<li><b>Ranking:</b> a flight counts once, and only in groups of at least the minimum
size. Flights without a known date take no part; their number is shown.</li>
<li><b>Lengths and times:</b> <i>Path</i> adds up the horizontal Lambert-93 lengths of
the edges between consecutive cleaned fixes, so gaps are excluded; <i>Gaps</i> is the
elapsed time those edges do not cover; <i>Elapsed</i> runs from the first to the last
cleaned fix, gaps included.</li>
<li><b>Time:</b> comparisons use UTC, so daylight-saving changes do no harm; tables
show French local time.</li>
</ol>
{_back()}
"""


def sources_html() -> str:
    """The whole page, with numbers read from the current configuration."""
    return "".join(
        part()
        for part in (
            _contents,
            _sources,
            _maps,
            _processing,
            _phases,
            _time,
            _launches,
            _cells,
            _crossings,
            _hours,
            _routes,
            _groups,
        )
    )


class SourcesMethods(QWidget):
    """A read-only page with a table of contents; built on first display."""

    def __init__(self, parent: QWidget | None = None):
        """Create an empty browser; reading configs waits for the tab to open."""
        super().__init__(parent)
        self._browser = info_browser(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._browser)

    def ensure_loaded(self) -> None:
        """Fill the page the first time the tab is shown."""
        if not self._browser.toPlainText():
            self._browser.setHtml(sources_html())

    def show_section(self, anchor: str) -> None:
        """Scroll to one section, once the page is laid out."""
        self.ensure_loaded()
        QTimer.singleShot(0, lambda: self._browser.scrollToAnchor(anchor))
