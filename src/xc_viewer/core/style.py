"""Semantic colours and typography for figures printed at thesis text width.

Colour here is a variable, not decoration. Each comparison the thesis draws owns a
set below, and a reader who learns one set can carry it from chapter to chapter.
Two rules keep that promise:

* The discipline pair and the equipment pair are reserved for the whole document.
  Blue always means paraglider, terracotta always means hang glider, violet and
  ochre always mean the two equipment classes, and nothing else ever uses those
  five values. These are the two comparisons Chapter 3 sets side by side, so a
  shared colour between them would be read as a shared meaning.
* The remaining sets are each internally distinct and avoid the reserved five.
  They may sit in neighbouring hue bands, because no figure and no chapter shows
  two of them at once: take-off regions belong to Chapters 2 and 3, flight phases
  to Chapter 4, and the channel and component sets label the axes of one figure.

Add a comparison by adding a set here, never by choosing a colour at the call
site. ``tests/reporting/test_palette.py`` checks both rules.

Importing this module does not change Matplotlib settings. Reporting entry points
call ``paper_style`` before drawing; numerical analysis never needs Matplotlib.
"""

# --- Reserved: the two comparisons that share Chapter 3 ----------------------
DISCIPLINE_COLORS = {
    "paragliders": "#3477A8",  # blue
    "hang gliders": "#B5482A",  # terracotta
}
# The archive holds no sailplanes yet. Reserving the third discipline's colour
# here stops a later addition from borrowing one already carrying a meaning.
SAILPLANE_COLOR = "#7E7B32"  # olive

# Wing class, the equipment-based experience proxy. Violet against ochre is far
# from blue against terracotta in both hue and lightness, so a duration panel
# split by class cannot be misread as one split by discipline.
EQUIPMENT_COLORS = {
    "EN A/B": "#6A3D9A",  # violet
    "EN C/D/CCC": "#C98A1E",  # ochre
}

# --- Chapter-local sets ------------------------------------------------------
# Take-off region. The three massifs take warm, earthy hues and the low-relief
# control a cool one, so the contrast these figures test is the first the eye
# makes. The two residual categories are deliberately neutral: they are defined
# by exclusion and should not compete with the boxes that are defined positively.
REGION_COLORS = {
    "Alps": "#8E3B5C",  # wine
    "Pyrenees": "#8A6240",  # brown
    "Massif Central": "#6FA58C",  # sage
    "Channel Coast": "#2E7D8A",  # teal
    "Poitou-Charente": "#9AA98F",  # sage grey
    "Champagne-Lorraine": "#B9A99A",  # warm grey
    "outside massifs": "#8C8C8C",
    "abroad": "#5A5A5A",
}

# Flight phase. Three states shown together in a strip, a posterior panel and a
# plan view, so they need separation at thin line widths: teal, plum and green
# differ in hue and in lightness.
PHASE_COLORS = {
    "transition": "#2E7D8A",  # teal
    "search": "#A9629E",  # plum
    "climb": "#4E8A5B",  # green
}
UNCLASSIFIED_COLOR = "#9E9E9E"

# A panel holding one series compares nothing, so it is drawn neutral: the
# colour in such a figure belongs to the phase strip and the posterior beside it.
TRACE_COLOR = "#3A3A3A"

# Curves that are not populations: the two limbs of an analytic illustration, the
# input and the output of a filter, what a cleaning rule keeps and what it marks.
# They share one set because they are the same kind of object and no figure shows
# two of them together. What matters is that they stay off the reserved pair,
# since Chapter 2 draws these explainers beside discipline curves.
ILLUSTRATION_COLORS = {
    "reference": "#9A9A9A",  # the record or curve being acted on
    "primary": "#2E7D8A",  # teal: the result, or the first of two limbs
    "secondary": "#8E3B5C",  # wine: what is flagged, or the second limb
    "tertiary": "#6FA58C",  # sage: a third curve, where one is needed
}

# Recorder channel, where one figure compares the two altitude sources against
# the horizontal one. The horizontal channel is the reference and stays neutral.
CHANNEL_COLORS = {
    "horizontal": "#5A5A5A",
    "vertical_baro": "#2E7D8A",  # teal
    "vertical_gnss": "#A9629E",  # plum
}

# Displacement component. The radius is built from the two components, so it is
# drawn neutral against their two hues.
COMPONENT_COLORS = {
    "east": "#2E7D8A",  # teal
    "north": "#A9629E",  # plum
    "radius": "#4A4A4A",
}

# Displacement quantiles: an ordered probability level, so an ordered ramp.
QUANTILE_COLORS = {0.25: "#482878", 0.50: "#31688E", 0.75: "#26828E", 0.90: "#35A779"}

# Regional PCA: one stable colour per physical lag, including the longest lag.
PCA_LAG_COLORS = {10: "#482878", 100: "#31688E", 1000: "#26828E", 10000: "#35A779"}

# Retained flight duration, in disjoint bands. Duration is an ordered quantity, so
# it takes a ramp rather than a categorical set. It is kept apart from CONTROL_GREYS
# because those are nested controls: these bands share no flights, and they are drawn
# over one another in a ratio panel, so the ramp needs more contrast than a nested
# series does.
DURATION_COLORS = ("#BDBDBD", "#878787", "#4D4D4D", "#1A1A1A")

# Ordered series that are nested rather than categorical -- successive sampling
# controls, or the parts of one stacked total. A ramp says "more of the same
# quantity" where a categorical set would wrongly suggest unrelated groups.
CONTROL_GREYS = ("#B4B4B4", "#8A8A8A", "#606060", "#303030")
# Sampling conventions are compared by colour as well as line style.
CONTROL_COLORS = ("#8C8C8C", "#2E7D8A", "#A9629E", "#4E8A5B")
# Ordered lag series in displacement-law figures, from short to long.
LAG_COLORS = ("#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#AD9D00")
STACK_GREYS = ("#4A4A4A", "#7A7A7A", "#A6A6A6", "#D0D0D0")

# Every set a figure may draw from, for the disjointness checks in the tests.
PALETTE_FAMILIES = {
    "discipline": {**DISCIPLINE_COLORS, "sailplanes": SAILPLANE_COLOR},
    "equipment": EQUIPMENT_COLORS,
    "region": REGION_COLORS,
    "phase": PHASE_COLORS,
    "channel": CHANNEL_COLORS,
    "component": COMPONENT_COLORS,
    "quantile": QUANTILE_COLORS,
    "pca_lag": PCA_LAG_COLORS,
    "sampling_control": dict(enumerate(CONTROL_COLORS)),
    "duration": dict(enumerate(DURATION_COLORS)),
    "lag": dict(enumerate(LAG_COLORS)),
}
RESERVED_FAMILIES = ("discipline", "equipment")

TEXT_WIDTH_IN = 6.1
PDF_METADATA = {
    "Creator": "xc_viewer.core",
    "Producer": "xc_viewer.core",
    "CreationDate": None,
}


def paper_style() -> None:
    """Readable vector figures at their final 6.1-inch text width."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            # The default cycle is the discipline pair first, because most curves
            # that go unnamed are one discipline against the other.
            # A named comparison takes its set from this module; the cycle is
            # only for curves that carry no group meaning, so it hands out no
            # reserved colour beyond the discipline pair it usually draws.
            "axes.prop_cycle": mpl.cycler(
                color=["#3477A8", "#B5482A", "#4E8A5B", "#5A5A5A"]
            ),
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.titlelocation": "left",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.axisbelow": True,
            "grid.color": ".93",
            "grid.linewidth": 0.4,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
