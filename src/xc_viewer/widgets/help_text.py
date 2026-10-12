"""What the Info and How to use windows of each screen say.

The viewer explains itself in three places, each with its own job:

- **Info** (one per screen): what the screen shows, and how to read it.
- **How to use** (one per screen): what to click, and in which order.
- **Sources & methods** (its own tab, :mod:`.sources_methods`): where the data come
  from, and exactly how each number is computed.

A fact is written in one of the three only, and the others link to it; a test checks
that no sentence appears in two of them. Numbers come from the code's own constants,
so the texts cannot drift from what the viewer does.
"""

from __future__ import annotations

from ..geography import REGIONS as LAUNCH_REGIONS
from ..geography import TERRAIN_BANDS
from ..route_index import DISTANCE_PRESETS_KM, MAX_FLIGHTS, ROUTE_CELL_M
from ..thermal_geometry import CELL_M
from ..thermal_regions import REGIONS as DENSITY_REGIONS
from ..thermal_time import BASE_M

INFO_ROLE = (
    "What this screen shows and how to read it. <b>How to use</b> gives the steps; "
    "the <b>Sources &amp; methods</b> tab gives the data sources and the exact "
    "calculations (blue links open the right section)."
)
HOWTO_ROLE = "The steps, in order. <b>Info</b> explains what the results mean."


def _link(anchor: str, text: str) -> str:
    """A link that opens one section of the Sources & methods tab."""
    return f'<a href="methods:{anchor}">{text}</a>'


def _km(metres: float) -> str:
    return f"{metres / 1000:g}"


def _bands() -> str:
    """The four terrain categories and their altitude limits, in words."""
    (plains, _, low), (hills, _, mid), (low_m, _, high), (high_m, _, _) = TERRAIN_BANDS
    return (
        f"{plains} below {low:,.0f} m, {hills} {low:,.0f}&ndash;{mid:,.0f} m, "
        f"{low_m} {mid:,.0f}&ndash;{high:,.0f} m, {high_m} from {high:,.0f} m"
    )


def _list(names) -> str:
    names = list(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# -- Trajectory ------------------------------------------------------------------------


def trajectory_info() -> str:
    """What the Trajectory screen draws."""
    return f"""
<p>One flight, drawn from its IGC file. A flight recorder logs a <b>fix</b> about
every second: the time, the position and the GNSS altitude.</p>

<h3>Raw and cleaned tracks</h3>
<ul>
<li><b>Raw</b> is the recording exactly as logged.</li>
<li><b>Cleaned</b> is the same flight after the processing applied to the whole
archive: impossible positions and altitudes are removed, the time on the ground
before take-off and after landing is cut off, and the path is lightly smoothed.</li>
<li>Where data are missing for too long, the cleaned track is split, and its pieces
are drawn separately rather than joined by a straight line.</li>
<li>Drawing both shows exactly what processing changed. When processing rejects a
flight (for example because it is too short or has too much bad data), only the raw
track is drawn and the status line gives the reason.</li>
</ul>

<h3>Colours of the cleaned track</h3>
<ul>
<li><b>Flight phase:</b> each cleaned fix carries one of three labels, given by the
Vilpellet segmentation. <b>Climb</b>: circling upwards, usually in a thermal.
<b>Search</b>: turning without a steady climb, typically looking for lift.
<b>Transition</b>: flying fairly straight, usually gliding between thermals. Grey
parts are unclassified. The legend names the colours.</li>
<li><b>Cleaning segment:</b> one colour per continuous piece of the cleaned
track.</li>
<li><b>Discipline:</b> one colour for paragliders, another for hang gliders.</li>
</ul>
<p>With <b>Thermals only</b>, the title counts the <b>climb runs</b> (unbroken
stretches labelled climb) and adds up their minutes. A run is not always a separate
thermal: a pilot who leaves a thermal briefly and comes back makes two runs.</p>

<h3>Axes</h3>
<ul>
<li><b>Geographic:</b> longitude and latitude in degrees, altitude in metres.</li>
<li><b>Local (ENU):</b> metres East and North of the point where free flight
starts, and altitude. Distances can be read directly.</li>
</ul>

<h3>Status line</h3>
<p>Under the flight list: the file, whether processing kept the flight, and the
share of its cleaned fixes that received a phase label. Only flights logged about
once per second for about an hour or more can be labelled; for the others the line
says why nothing was labelled.</p>
<p>This screen cleans and labels the opened flight from its IGC file each time; the
other screens read results prepared from the whole archive and saved in the data
folder. See {_link("processing", "Processing a flight")} and
{_link("phases", "Flight phases")}.</p>
"""


def trajectory_howto() -> str:
    """How to open a flight and change the plot."""
    return """
<h3>1. Open a flight (left panel)</h3>
<ul>
<li><b>Search the catalogue:</b> choose the <b>Discipline</b>, fill in any filters,
then click <b>Search catalog</b>. A filter must match a whole value: type it, or pick
it from its list; capitals do not matter and empty filters are ignored. Double-click a
row in the results to open that flight.</li>
<li><b>Browse .igc file&hellip;</b> opens any IGC file, also one outside the data
folder. For such a file, select its discipline first: the cleaning uses
different speed limits for paragliders and hang gliders.</li>
<li><b>Browse folder of .igc files&hellip;</b> lists the IGC files of a folder and
its subfolders (at most 2,000); double-click one.</li>
<li>Or click a launch point in the <b>Map</b> tab.</li>
</ul>
<p>Then read the line under the results: it names the loaded flight, or says why it
could not be shown. The first search of a session takes a few seconds.</p>

<h3>2. Choose what to draw</h3>
<ul>
<li><b>Show raw</b> and <b>Show cleaned</b> switch each track on or off.</li>
<li><b>Cleaned colour</b>: Flight phase, Cleaning segment or Discipline.</li>
<li><b>Thermals only (climb)</b> hides everything except the climbs.</li>
<li><b>Frame</b>, <b>X</b> and <b>Y</b> choose the coordinates of the two axes.</li>
<li><b>Degrees-minutes-seconds</b> writes geographic axis labels as
45&deg;10&prime;30&Prime; instead of decimal degrees.</li>
</ul>

<h3>3. Look around</h3>
<ul>
<li>Toolbar above the plot: the four-arrow cross pans (drag), the magnifier zooms
into a rectangle you draw, the house returns to the first view, and the arrows step
back and forward through your views.</li>
<li><b>3D</b>: tick it, then choose <b>Z</b>. Drag the plot to rotate it, or use
<b>Azimuth</b> (turn around) and <b>Elevation</b> (look from higher or lower);
<b>Zoom</b> enlarges. <b>Reset view</b> restores the starting camera.</li>
<li><b>Map full screen</b> (top right) gives the plot the whole window; click it
again to come back.</li>
</ul>

<h3>4. Keep a copy</h3>
<p><b>Save PDF&hellip;</b> saves the plot as it looks now, as a vector PDF.</p>
"""


# -- Map -------------------------------------------------------------------------------


def map_info() -> str:
    """What the launch map counts."""
    from .map_view import _SCATTER_MAX_POINTS

    regions = _list(LAUNCH_REGIONS)
    return f"""
<p>Where flights start. Every flight that processing kept appears once, at the point
where its free flight begins. Its length and duration do not matter.</p>

<h3>Coloured cells or points</h3>
<ul>
<li><b>Coloured cells</b> (when zoomed out): how many flights started in each cell.
The colour scale is logarithmic, so each colour step multiplies the count. The cells
shrink as you zoom in, so the same place shows smaller numbers in smaller cells; the
colour bar gives the counts of the current view.</li>
<li><b>Points</b> (when {_SCATTER_MAX_POINTS:,} flights or fewer are in view): one point
per flight. Hovering one shows the catalogue details: date, pilot, flight type, wing
class, declared distance and duration, take-off and landing sites, and the region and
terrain category of the launch.</li>
</ul>

<h3>Filters</h3>
<ul>
<li><b>Show</b>: both disciplines, or one.</li>
<li><b>Region</b>: a rectangle around the {regions}.</li>
<li><b>Terrain</b>: the altitude of the launch point: {_bands()}.</li>
</ul>
<p>The map shows where pilots take off, not where thermals are: a popular site
collects many launches whatever the weather. Exact grid and boxes:
{_link("launches", "Launch map")}.</p>
"""


def map_howto() -> str:
    """How to move around the launch map."""
    return """
<ol>
<li>The launch points load the first time the tab opens. <b>Reload</b> reads them
again, for example after choosing another data folder.</li>
<li><b>Zone</b> jumps to the World, France or La R&eacute;union.</li>
<li>Zoom with the mouse wheel or the toolbar's magnifier; pan with its four-arrow
cross. The cells are recomputed after each move.</li>
<li>Choose <b>Show</b>, <b>Region</b> and <b>Terrain</b> to keep only some
flights.</li>
<li>Zoom in until single points appear. Hover a point to read about the flight;
click it to open the flight in the Trajectory tab.</li>
<li><b>Map full screen</b> (top right) enlarges the map; click it again to come
back.</li>
</ol>
"""


# -- Thermal planes --------------------------------------------------------------------


def planes_info() -> str:
    """What the dots on a horizontal plane are."""
    cell = _km(CELL_M)
    return f"""
<p>Where, inside a {cell} &times; {cell} km square, climbing pilots crossed a chosen
altitude. Seen from above, the crossings show where the thermals were at that
height.</p>

<h3>The twelve cells</h3>
<p>France is divided into a fixed grid of {cell} &times; {cell} km squares, called
cells. This tab offers twelve of them: in each of four terrain categories, the three
cells with the most climbs recorded over all dates. A cell's category follows its
highest ground point: {_bands()}. A label such as M1 gives the category (P, H, L or M)
and the rank within it.</p>

<h3>The plane and its dots</h3>
<ul>
<li>The plane lies at <b>H = lowest ground in the cell + z</b>, and you choose
z.</li>
<li>Each dot is one place where a flight, while labelled climb, crossed that altitude,
going up or down. Its colour gives the discipline.</li>
<li>A pilot circling in a thermal crosses the plane many times, so one flight can
leave many dots. A dot marks a crossing, not the centre of a thermal.</li>
<li>Flights that stayed entirely above or below H leave no dots, so a busy cell can
look empty at some heights.</li>
<li>The dashed square marks the selected cell. When zoomed out, the neighbouring
cells show their own crossings at the same altitude H.</li>
</ul>

<h3>Numbers on screen</h3>
<ul>
<li><b>Climb runs</b> (Cell details): unbroken climbs in the cell over all dates.
They rank the cells.</li>
<li><b>Cell visitors</b> (status line): flights that passed through the cell during
the selected dates, at any height and in any phase.</li>
<li><b>Climb intersections</b> and <b>contributing flights</b> (above the plane):
the dots of this plane in the current view and hours, and how many flights they come
from.</li>
</ul>

<h3>Time</h3>
<p>Dates and hours are French local time. <b>Whole interval</b> takes one
continuous period. <b>Daily hour window</b> takes the same hours on each selected
day, for example 08:00&ndash;18:00 on every day of a summer.</p>

<h3>Maps</h3>
<p>The left map shows France and the twelve cells. Backgrounds (IGN maps and aerial
photos, Esri shaded relief) only give context: they never change a count.</p>
<p>Exact rules: {_link("cells", "Cells and ranking")},
{_link("crossings", "Plane crossings")}, {_link("time", "Dates and times")}.</p>
"""


def planes_howto() -> str:
    """How to load and explore the crossings of one cell."""
    return """
<ol>
<li>The tab reads its prepared data from the data folder when it opens; <b>Reload
data</b> reads it again.</li>
<li>Choose a cell in the list at the top, or click its label on the France map.</li>
<li>Choose the dates:
<ul>
<li><b>Whole interval</b>: set <b>From</b> and <b>To</b>.</li>
<li><b>Daily hour window</b>: choose how to pick the days. <b>Start / end
dates</b>: every day between two dates. <b>Days around a date</b>: one date, plus some
days before and after it. <b>Same dates every year</b>: for example 1 June to 31 August
of each year in a range. Then set the hours under <b>Paris hours</b>. <b>Busiest
summer day</b> fills in the busiest summer day of this cell.</li>
</ul></li>
<li>Click <b>Load climb intersections</b>. Long periods take a while; <b>Cancel</b>
stops. Changing the cell or the dates clears the dots until you load again; changing
only the hours does not.</li>
<li>Choose the altitude: drag the <b>Horizontal plane</b> slider, or type z (metres
above the lowest ground) in the box next to it. <b>Height increment</b> sets the
slider's step.</li>
<li>Look closer: <b>Zoom +</b> and <b>Zoom -</b> (from 0.5 to 10 km across; zooming
out shows the eight neighbouring cells), the toolbar's four-arrow cross to pan, and
<b>Reset cell</b> to return. <b>View</b> shows both maps or only one of them.</li>
<li>Adjust the look: choose the background, then set <b>% background</b> and
<b>% points</b>, the opacity of the map and of the dots.</li>
<li><b>Cell details</b> shows the numbers of the selected cell. <b>3D terrain &middot;
selected cell</b> opens the same cell and dates in 3D.</li>
<li><b>Map full screen</b> (top right) enlarges the maps.</li>
</ol>
"""


# -- 3D terrain ------------------------------------------------------------------------


def terrain_info() -> str:
    """What the 3D snapshot of a cell shows."""
    return f"""
<p>A 3D view of the cell and dates selected in Thermal planes: the ground, and the
climb crossings of many horizontal planes, one every 20 m from the cell's lowest
ground upwards. It shows how the crossings are arranged in height.</p>
<ul>
<li>The points (blue) are the same kind of crossing as the dots in Thermal planes;
all of them are drawn, none is left out to lighten the view.</li>
<li>The surface is IGN terrain, coloured by elevation, or with the IGN aerial photo
laid on it.</li>
<li>All three axes are in metres: heights are not exaggerated.</li>
<li>The area is the selected cell alone (5 &times; 5 km), or the cell with its
neighbours (10 &times; 10 km), centred on it.</li>
<li>The view keeps the cell and dates it was opened with: changes made afterwards in
Thermal planes do not reach it.</li>
</ul>
<p>Exact rules: {_link("crossings", "Plane crossings")}.</p>
"""


def terrain_howto() -> str:
    """How to move in the 3D view of a cell."""
    return """
<ul>
<li><b>Area</b>: 5 &times; 5 km or 10 &times; 10 km; the points load again.</li>
<li>Mouse: drag to rotate; Shift + drag, or drag with the right button, to move; Alt
+ drag to look around from where you are; wheel or pinch to zoom.</li>
<li>Keyboard: click the scene first. W and S move forward and back, A and D left and
right, Q and E down and up; the arrow keys and Page Up / Page Down do the same.</li>
<li><b>Top view</b> looks straight down; <b>Reset view</b> returns to the starting
view.</li>
<li><b>Point opacity</b> and <b>Point size</b> change how the points look, never
which are drawn. Untick <b>Terrain</b> to see points hidden behind the ground. Choose
<b>Terrain colours</b> or <b>Aerial photo &middot; IGN</b> for the surface.</li>
<li>For another cell or other dates: change them in Thermal planes, then click
<b>3D terrain &middot; selected cell</b> again.</li>
<li><b>Map full screen</b> enlarges the scene; <b>Cancel loading</b> stops a slow
load.</li>
</ul>
"""


# -- Thermal density -------------------------------------------------------------------


def _hours_unit() -> str:
    """The meaning of hours per square kilometre, with examples; shared by 3 screens."""
    pixel = f"{BASE_M:g} &times; {BASE_M:g} m"
    area = (BASE_M / 1000) ** 2
    return f"""
<p>Picture a stopwatch on every patch of ground. It runs whenever a recorded pilot
above that patch is climbing, at any height. The time it adds up, divided by the
area of the patch, is given in <b>hours per square kilometre</b> (h/km&sup2;).</p>
<ul>
<li>Three flights climbing for 10, 20 and 30 minutes above the same 1 km&sup2; give 1
hour there: <b>1 h/km&sup2;</b>.</li>
<li>The finest patches measure {pixel} ({area:g} km&sup2;). Climbing for 36 seconds in
one of them gives 0.01 h / {area:g} km&sup2; = <b>4 h/km&sup2;</b>, although only 36
seconds were recorded there.</li>
<li>Two pilots in the same thermal both count, so the value grows with the number of
pilots, not only with the strength of the thermals.</li>
<li>Transparent means no recorded climbing time: often simply that nobody flew
there. It does not show that there were no thermals.</li>
</ul>
"""


def density_info() -> str:
    """What the climbing-time maps show."""
    regions = _list(DENSITY_REGIONS)
    return f"""
<p>Where pilots spent time climbing, added up over all flights, all dates and all
heights, paragliders and hang gliders together.</p>

<h3>The unit: hours per km&sup2;</h3>
{_hours_unit()}

<h3>Reading the maps</h3>
<ul>
<li>The darker the colour, the more climbing time. The scale is logarithmic and the
same for every panel.</li>
<li>When zoomed out, patches are coarser: their time is added up and divided by their
larger area, so no time is invented.</li>
<li><b>Regions</b>: five areas: {regions}. <b>Cells</b>: the twelve squares of
Thermal planes, one at a time or by terrain category.</li>
<li>Panel titles give the total climbing hours inside the panel. Panels can overlap,
so their totals must not be added up.</li>
<li>Backgrounds only give context.</li>
</ul>
<p>Exact rules: {_link("hours", "Climbing hours per km&sup2;")}.</p>
"""


def density_howto() -> str:
    """How to browse the climbing-time maps."""
    return """
<ol>
<li>The prepared maps load when the tab opens; <b>Reload data</b> reads them
again.</li>
<li>In the first list choose <b>Regions</b> or <b>Cells</b>. In the second, choose
what to show: all regions side by side or one region; all cells of a terrain category
or one cell.</li>
<li>Zoom with the mouse wheel or the toolbar's magnifier, pan with its four-arrow
cross, and click the house to see the whole frame again. Finer patches appear as you
zoom in, down to 50 m.</li>
<li><b>Background</b> chooses the map underneath; <b>% terrain</b> and <b>%
density</b> set the opacity of the map and of the colours.</li>
<li>Background maps are downloaded as you move, so they need an internet
connection; maps already seen stay available offline.</li>
<li><b>Map full screen</b> (top right) enlarges the panels.</li>
</ol>
"""


# -- Routes ----------------------------------------------------------------------------


def routes_info() -> str:
    """What the route comparison selects and draws."""
    cell = _km(ROUTE_CELL_M)
    return f"""
<p>Flights that took the same route, drawn in 3D over the terrain: all of them
started in the same {cell} &times; {cell} km cell A and ended in the same cell B.</p>

<h3>Which flights</h3>
<ul>
<li>A flight starts and ends at its first and last cleaned fixes. Passing through A
or B is not enough, and A &rarr; B and B &rarr; A are different routes.</li>
<li>The distance in the route list is measured between the centres of A and B: it is
not the distance flown. <b>Pairs shared by both</b> lists only routes flown by both
paragliders and hang gliders.</li>
<li>Flights are ranked by elapsed time, from start to end; rank 1 is the quickest.
The five quickest are drawn in cyan, the five slowest in lime green, the others in
white, and the rows you select in blue. A circle marks each start, a square each
end, and pieces of track separated by missing data are not joined.</li>
<li>Busy routes are reduced to at most {MAX_FLIGHTS} flights: the five quickest, the
five slowest, and others spread evenly in between. The summary under the route list
gives both numbers, loaded and matching.</li>
<li><b>Departure window</b> keeps only flights that started on one day within a
time window, for example 12:00&ndash;12:30 on 14 July; ranks and the reduction are
then worked out again within those flights. Starting together makes flights
candidates for having flown together; their tracks show whether they did.</li>
</ul>

<h3>Table</h3>
<p>One row per drawn flight: rank, discipline, flight number, start and end (date
and time in France), elapsed time, and whether it is among the fastest or slowest
five.</p>

<h3>Colours on the ground: thermal hours</h3>
<p>The colours draped on the terrain measure, in hours per km&sup2;, the climbing
time of <b>all</b> flights in the archive over this area, at all dates and heights.
They do not depend on the route, on the flights drawn or on the departure window.</p>
{_hours_unit()}

<h3>Side map</h3>
<p>The small map places the scene in France or in the world.</p>
<p>Exact rules: {_link("routes", "Routes")},
{_link("hours", "Climbing hours per km&sup2;")}.</p>
"""


def routes_howto() -> str:
    """How to pick, load and explore a route."""
    presets = ", ".join(map(str, DISTANCE_PRESETS_KM))
    return f"""
<ol>
<li>The tab opens the route index saved in the data folder. <b>Prepare / refresh
index</b> is needed only when the status line asks for it: it reads the whole archive
and can take a long time.</li>
<li>Choose the flights: <b>Both disciplines</b>, a single discipline, or <b>Pairs
shared by both</b>.</li>
<li>Choose a distance: <b>Around &hellip; km</b> ({presets} km) sets <b>Min</b> and
<b>Max</b> to &plusmn;10 km, or set Min and Max yourself.</li>
<li>Pick a route in the list, busiest first: each line gives the two cells, their
distance and the number of flights. Click <b>Load selected pair</b>; <b>Cancel</b>
stops.</li>
<li>Move in the scene: drag to rotate, Shift + drag to move, wheel or pinch to zoom.
<b>Top view</b> looks straight down; <b>Reset view</b> returns to the starting
view.</li>
<li>Choose the flights to draw with the list of groups (all, 5 fastest, 5 slowest, or
both), or one by one with the ticks in the table; <b>Show all flights</b> and <b>Hide
all flights</b> do every row at once. Select rows to highlight their flights.</li>
<li>Drag the bar above the table up for more rows, or down for a bigger scene.</li>
<li>Appearance: <b>Track width</b>; <b>IGN terrain</b> on or off and its surface
(elevation colours, aerial photo, grey relief); <b>Vertical exaggeration</b> makes
heights look taller, without changing any value.</li>
<li>Thermal colours: tick <b>Thermal hours &middot; all flights</b>, then choose the
pixel size and the opacity.</li>
<li>Flights that started together: tick <b>Departure window</b>, choose the day, the
start time and the length, check the preview count, then click <b>Apply</b>. Untick
it and click Apply to see all flights again.</li>
<li>Side map: switch between <b>France</b> and <b>World</b>; <b>Expand map</b> opens a
larger map.</li>
</ol>
"""


# -- Route location --------------------------------------------------------------------


def locator_info() -> str:
    """What the expanded route location map shows."""
    return """
<p>Where the Routes scene lies, at the scale of France or of the world. The outline
is the area of the 3D scene, and A and B mark its start and end cells. It is context
only: it never changes which flights are shown.</p>
<p>Borders and coastlines come with the viewer, so this map works offline.</p>
"""


def locator_howto() -> str:
    """How to use the expanded location map."""
    return """
<ul>
<li><b>France</b> or <b>World</b> changes the scale.</li>
<li>Zoom and pan with the toolbar under the map.</li>
<li>The map follows the route loaded in Routes.</li>
</ul>
"""


# -- Group flights ---------------------------------------------------------------------


def groups_info() -> str:
    """What a launch group is and what the group scene shows."""
    return f"""
<p>Flights that took off from the same place at about the same time: pilots who may
have flown together. Where they went, and how far, does not matter.</p>

<h3>How groups are made</h3>
<ul>
<li>A flight's starting place is the cell (5 &times; 5 or 10 &times; 10 km) of its
first cleaned fix.</li>
<li>For each cell and each day, departures are put in time order. The first opens a
<b>launch window</b> of the chosen length, and every flight starting within it joins
its group. The first flight after the window opens the next one. With a 30-minute
window, departures at 10:00 and 10:20 form a group, and 10:40 opens a new one.</li>
<li>Each flight belongs to one group only. Groups smaller than <b>Minimum
flights</b> are not listed.</li>
<li>Cells are ranked by the number of flights in groups, by their largest group, or
by their number of groups, over the chosen years and disciplines.</li>
</ul>

<h3>Scene and table</h3>
<ul>
<li>All members of the group are drawn, coloured from blue for the first to leave,
through cyan, to lime for the last; selected rows turn white.</li>
<li>Columns: start and end (French time); <b>Delay</b>, minutes after the group's
first departure; elapsed time; <b>Path</b>, the horizontal distance flown;
<b>Net</b>, the straight line from start to end; <b>Gaps</b>, the time without
data; and the number of track pieces.</li>
<li>The colours on the ground are the thermal hours per km&sup2; of all flights in the
archive, as in Routes. They say nothing about the weather on the group's day.</li>
</ul>
<p>Sharing a launch window does not prove that pilots flew together: compare their
tracks. Exact rules: {_link("groups", "Launch groups")}.</p>
"""


def groups_howto() -> str:
    """How to find and load a launch group."""
    return """
<ol>
<li>When the tab opens it reads the saved data, ranks the cells and loads the busiest
group. <b>Prepare / refresh groups</b> is needed only when the status line asks for
it: it reads the whole archive and can take a long time.</li>
<li>Define the groups: <b>Departure cell</b> size, <b>Launch window</b> in minutes,
<b>Minimum flights</b>, discipline and year. The ranking updates by itself.</li>
<li>Choose <b>Rank cells by</b>, then a cell in the list next to it.</li>
<li>Choose a <b>Launch group</b> and click <b>Load selected group</b>.</li>
<li>In the scene, rotate, move and zoom as in Routes. Select rows to highlight their
flights, untick rows to hide flights, or choose <b>Selected rows only</b>.</li>
<li><b>Export group CSV</b> saves the table of the loaded group.</li>
</ol>
"""


# -- Lookup ----------------------------------------------------------------------------

TOPICS = {
    "trajectory": ("Trajectory", trajectory_info, trajectory_howto),
    "map": ("Map", map_info, map_howto),
    "planes": ("Thermal planes", planes_info, planes_howto),
    "terrain": ("3D terrain", terrain_info, terrain_howto),
    "density": ("Thermal density", density_info, density_howto),
    "routes": ("Routes", routes_info, routes_howto),
    "locator": ("Route location", locator_info, locator_howto),
    "groups": ("Group flights", groups_info, groups_howto),
}


def body(topic: str, kind: str) -> str:
    """The text of one window, without its title and role line."""
    _, info, howto = TOPICS[topic]
    return info() if kind == "info" else howto()


def page(topic: str, kind: str) -> tuple[str, str]:
    """The window title and the full HTML of one screen's Info or How to use."""
    screen = TOPICS[topic][0]
    if kind == "info":
        title, role = f"{screen}: what it shows", INFO_ROLE
    else:
        title, role = f"{screen}: how to use it", HOWTO_ROLE
    html = f'<h2>{title}</h2><p class="role">{role}</p>{body(topic, kind)}'
    return title, html
