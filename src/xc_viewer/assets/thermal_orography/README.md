# Crest references for thermal-plane maps

The viewer offers **Estimated crests · IGN DEM**, off by default. Thesis and
slide exporters also use these **red crest lines**, replacing named summit
triangles. The viewer defaults to Plan IGN with official elevation contours
(`ELEVATION.CONTOUR.LINE`), distinct from crest estimates. The lines are derived from small [IGN RGE ALTI terrain
extracts](https://www.data.gouv.fr/datasets/rge-alti-r), under **Licence Ouverte
2.0**. They are approximate, scale-dependent height ridges, not official mapped
crest vectors or an exhaustive inventory of every local maximum.

Only 5 × 5 km viewer cells were downloaded: the twelve current ones and those of
earlier selections, 19 in all, each with a 500 m buffer for differentiation. Each
terrain response is a 240 × 240 float32 TIFF (25 m sampling, 230,534 bytes); total
**4,380,146 bytes**, approximately 4.18 MiB.
WMS source: `https://data.geopf.fr/wms-r/wms`, layer
`ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`, `STYLES=normal`, `FORMAT=image/tiff`,
`CRS=EPSG:2154`. RGE ALTI is originally a finer terrain product; 25 m is the
requested sampling interval of these windows, not its native accuracy.
Retrieved 22 September 2026 (the seven cells added on 6 October 2026 on that date). The TIFFs contain heights, not a coloured map.

## Reproducible derivation

`thermal_ridges.derive_ridges` uses only the terrain, independently of blue
crossings, flight counts, plane heights and seasons:

1. Smooth terrain with a Gaussian of sigma 50 m.
2. Compute the gradient and Hessian. A crest has zero gradient along the most
   concave transverse direction and negative curvature along that direction.
3. Retain curvature below −0.0003 m⁻¹ and a drop of at least 5 m on **both**
   sides at 200 m transverse distance. Suppress traces shorter than 200 m
   before clipping to the cell. Two orientation charts avoid sign artefacts.
4. Clip each surviving line to the exact Lambert-93 cell square. Do not join
   disconnected pieces across valleys, weak ridges or junctions.

The method follows the differential height-ridge criterion, not a complete
implementation of Steger's bias-corrected line detector. Smoothing can shift
asymmetric crests; chart boundaries and junctions can leave gaps. These lines
are for qualitative topographic orientation, not precise slope assignment or
navigation. Flat cells can still contain small local ridges. A missing or
empty trace is not evidence that no relief exists.

Every `ign-ridges-*.geojson` records source query, licence, retrieval time,
terrain hash/extent, code hash, parameters, derivation time and limitations.
The accompanying `ign-terrain-*.tif` allows offline reproduction. Row order
is reversed when processing: TIFF north-to-south becomes south-to-north.

```bash
.venv/bin/python scripts/prepare_thermal_ridges.py
# Recompute from the already saved terrain, without downloading again:
.venv/bin/python scripts/prepare_thermal_ridges.py --rederive
```

The prepared data folder selects the twelve cells and is read only during crest preparation. `--refresh`
requests new terrain; default execution reuses existing valid crest extracts.
The viewer and figure exporter share `thermal_ridges.load_ridges` and
`thermal_ridges.draw_ridges`; neither downloads data. Terrain lines do not
change the flight sample or counts.

## Terrain references for the viewer

`thermal_ground.terrain_reference` reuses these elevation TIFFs to compute the
area-weighted mean, the minimum and the maximum inside the exact 5 × 5 km cell. It
verifies each raster's hash against its saved provenance and requires complete
finite coverage. All three use the original elevations, before ridge smoothing,
excluding the 500 m buffer: 200 × 200 = 40,000 values per cell at 25 m sampling.
The maximum sets the cell's altitude band; the minimum is the thermal plane's
reference elevation. Launch medians remain separate audit metadata.

```bash
.venv/bin/python scripts/prepare_thermal_ground.py
```

This upgrades the standalone store atomically and regenerates intersections
from saved climb edges at the new absolute heights. The viewer shows the terrain
source, sampling, retrieval date, licence, dataset link and exact WMS query.
The base WMS endpoint alone is not a web page: a GetMap request needs the layer,
bounding box, projection, raster dimensions and output format. The TIFF download
contains numerical elevations rather than a browser map.

## Reading the mountain cell

In H2, the main blue bands at the lower plane chiefly follow outer slopes of
**two separate ridges**: Lances de Malissard on the west and Aulp du Seuil–
Bellefont on the east, enclosing the Vallon de Marcieu. They are not simply
opposite sides of a single central crest. This is a qualitative reading of
IGN relief and the independent terrain traces, not a quantified windward/
leeward classification. Red lines alone do not locate thermal sources.

## Earlier summit extracts

`ign-summits-*.geojson` retain the previous small IGN BD TOPO named `Sommet`/
`Pic` extracts and their full provenance. They are no longer displayed by
default. Their separate source is [IGN BD TOPO](https://www.data.gouv.fr/datasets/bd-topo-r),
Licence Ouverte 2.0. These discrete named points were insufficient to trace
continuous ridges. IGN `ligne_orographique` features marked `Talus` were not
relabelled as crests. OpenStreetMap ridge/arete geometry was inspected in H2,
but its coverage was too fragmented; it is not used in the published overlay.

## Neighbour exploration

Additional elevation extracts requested by viewer navigation are stored in the
data folder in `exploration/terrain/` beside `thermal-planes.sqlite3`, not in this folder.
They use the same official IGN elevation layer, with a 200 × 200 float TIFF over
the exact 5 km square. A JSON sidecar retains the query, hash, sampling and
retrieval date. These extracts supply terrain references; no red crest
geometry is automatically inferred for neighbouring cells.
