"""Links to the data sources, shared by the help pages and the status lines."""

from ..thermal_relief import SOURCE_URL as HILLSHADE_URL

FFVL_PARAGLIDING = "https://parapente.ffvl.fr/cfd/liste"
FFVL_HANG_GLIDING = "https://delta.ffvl.fr/cfd/liste"
NATURAL_EARTH = (
    "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/"
    "10m-admin-0-countries/"
)
NATURAL_EARTH_WORLD = (
    "https://www.naturalearthdata.com/downloads/50m-cultural-vectors/"
    "50m-admin-0-countries-2/"
)
PLAN_IGN = "https://www.data.gouv.fr/datasets/plan-ign"
BD_ORTHO = "https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_BD-ORTHO"
CONTOURS = "https://www.data.gouv.fr/datasets/courbes-de-niveau-4"
ORTHO_DATES = (
    "https://data.geopf.fr/wfs/ows?SERVICE=WFS&amp;VERSION=2.0.0"
    "&amp;REQUEST=DescribeFeatureType"
    "&amp;TYPENAMES=ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE-MOSAIQUAGE:graphe_bdortho"
)
LICENCE_OUVERTE = "https://www.etalab.gouv.fr/licence-ouverte-open-licence/"
VILPELLET_PAPER = "https://arxiv.org/abs/2601.01293"

FLIGHT_SOURCE_HTML = (
    f'<a href="{FFVL_PARAGLIDING}">FFVL CFD paragliders</a> / '
    f'<a href="{FFVL_HANG_GLIDING}">hang gliders</a>'
)

__all__ = [
    "BD_ORTHO",
    "CONTOURS",
    "FFVL_HANG_GLIDING",
    "FFVL_PARAGLIDING",
    "FLIGHT_SOURCE_HTML",
    "HILLSHADE_URL",
    "LICENCE_OUVERTE",
    "NATURAL_EARTH",
    "NATURAL_EARTH_WORLD",
    "ORTHO_DATES",
    "PLAN_IGN",
    "VILPELLET_PAPER",
]
