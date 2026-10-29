# XC Viewer

Interactive viewer of cross-country paraglider and hang-glider flights from the French
FFVL contest archive (CFD). It shows single tracks with their flight phases
(Vilpellet segmentation), launch maps, route comparisons and group flights, and the
thermals the pilots climbed: thermal planes over 5 km cells and regional thermal-time
maps.

## This branch: `feature/group-episodes`

The **Group flights** tab is rebuilt on the groups found by the delayed-contact method of
[xc-lab](https://github.com/matteodisante/xc-lab) (`methods/group-identification`):

- The tables are read from the `groups/` folder of the data folder, where xc-lab's
  `export_viewer.py` copies one run with its `PARAMETERS.md`.
- Episodes are ranked by distinct flights, largest group, duration or distance.
  **Zones and days…** opens a map of the 10 km zones and a list of the days, coloured and
  sorted by episodes, flights or hours, to pick where and when to look.
- **Load episode** draws the members on the terrain of Routes; **Play** and the time
  slider move them as numbered dots in the colour of their group, joined by
  lines of the same colour when in contact; tracks take one colour per pilot, whole, only
  up to the instant or as the bright trail alone. A click on a pilot makes the camera
  follow its group, alone or with the others hidden. Beside the scene, the
  contact graph of the instant; under it, the **Timeline** of every member and the
  **Reeb graph** of the episode, live or whole. **Who is who** lists the pilot, flight
  ID, club and wing behind each member number.

To launch this branch instead of `main` (the [requirements](#requirements) and the data
folder are the same as below):

```bash
uvx --from git+https://github.com/matteodisante/xc-viewer@feature/group-episodes xc-viewer
```

From a clone of this repository:

```bash
git switch feature/group-episodes
uv run xc-viewer
```


## Project and development

Developed by **Matteo Di Sante** as part of the internship
*Anomalous Transport in Soaring Flights: Collective Strategies and Intermittent
Search Models* at the **Econophysics Lab at CFM (Paris)**, under the supervision of
**Prof. Michael Benzaquen** and **Dr. Alexandre Darmon**.

These project credits are also available in **About** inside the viewer. Data
sources, scientific methods and licences are documented separately in
**Sources & methods**.

## Requirements

**[uv](https://docs.astral.sh/uv/)** and **git**. You do not need to install Python
or any library yourself. Works on macOS, Linux and Windows. 
On Windows, use **Git Bash** as your terminal.

Check what you have; each command prints a version number, or an error if the tool is
missing:

```bash
uv --version
```

Install uv if it is missing, then open a new terminal:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (Git Bash)
winget install --id=astral-sh.uv -e
```

### On a company or university network

If uv stops with `invalid peer certificate: UnknownIssuer`, the network inspects HTTPS
with its own certificate. Your IT department has installed it in the system, but uv
does not look there by default. Tell uv to use the system certificates, once:

```bash
# uv 0.11 or newer (check with: uv --version)
echo 'export UV_SYSTEM_CERTS=1' >> ~/.bashrc
# uv older than 0.11
echo 'export UV_NATIVE_TLS=1' >> ~/.bashrc
```

On macOS, write to `~/.zshrc` instead of `~/.bashrc`. Then open a new terminal: the
launch command stays the same. The viewer itself always uses the system certificates
for its map downloads.


### What uv does on the first launch

The first launch downloads, into uv's own folders in your user directory:

- **Python 3.12**, only if your computer has no Python 3.12 or newer. A Python already
  on your system is left untouched, and other programs do not see this one.
- **the viewer and its libraries**, in a private environment used only by the viewer.

Nothing is installed system-wide and nothing is added to your PATH. On macOS and Linux
these folders are `~/.cache/uv` and `~/.local/share/uv/python`; on any system,
including Windows, `uv cache dir` and `uv python dir` print them. Later launches reuse
them and start quickly. `uv cache clean` deletes the viewer and its libraries;
`uv python uninstall 3.12` deletes the Python that uv downloaded.

## Run it

1. Launch the viewer (same command on every system):

   ```bash
   uvx --from git+https://github.com/matteodisante/xc-viewer xc-viewer
   ```

2. Click **Start exploring** on the welcome screen.

3. On first launch a dialog asks for the data folder. Select the folder that holds
   `paragliders/` and `hang_gliders/` (called **`xc-viewer-data`** in this README), not
   one of its subfolders. Its name does not matter. The viewer remembers it, so later
   launches open your data after **Start exploring**. To switch folders, click
   **Choose data folder…** at the top of the left panel.

You can also give the data folder on the command line; it is remembered the same way:

```bash
# macOS and Linux
uvx --from git+https://github.com/matteodisante/xc-viewer xc-viewer /path/to/xc-viewer-data
```

```bash
# Windows (Git Bash)
uvx --from git+https://github.com/matteodisante/xc-viewer xc-viewer D:/xc-viewer-data
```

From a clone of this repository, `uv run xc-viewer` does the same.

### Updates

Each launch checks this repository and runs its latest version; it downloads and
rebuilds only when the code has changed.

## The data folder

The viewer reads everything from the data folder (`xc-viewer-data` here), which is not
part of this repository. The folder holds one archive per discipline, either or both:

```text
xc-viewer-data/
├── manifest.json            what was packed, and when
├── paragliders/
│   ├── raw/igc/<season>/    the .igc tracks
│   ├── catalog/catalog.csv  flight metadata
│   └── derived/
│       ├── fixes.parquet, flights_meta.parquet   cleaned tracks
│       ├── segmentation/vilpellet/               saved flight phases
│       └── viewer/thermal-planes/                prepared maps, indexes and caches
└── hang_gliders/            the same layout
```

A tab whose files are missing still opens and says what it needs. Map backgrounds
(IGN Géoplateforme, Esri hillshade) are downloaded when shown, so they need an
internet connection; everything saved in the folder works offline.

The folder can be moved or copied to another computer as is: saved products identify
their inputs by their place inside the folder and their content.


## Credits and licences

**Code:** MIT, see [LICENSE](LICENSE). The libraries that uv downloads keep their own
licences; in particular PyQt6, the window toolkit, is GPL v3.

**Flight phases:** the climb, search and transition labels come from the segmentation of
J. Vilpellet, A. Darmon and M. Benzaquen, *From Random Walks to Thermal Rides: Universal
Anomalous Transport in Soaring Flights*,
[arXiv:2601.01293](https://arxiv.org/abs/2601.01293) (2026), ported from the
authors' code with their fitted parameters.

**Flight tracks:** FFVL contest archive (CFD),
[paragliders](https://parapente.ffvl.fr/cfd/liste) and
[hang gliders](https://delta.ffvl.fr/cfd/liste). They are only in the data folder,
not in this repository.

**Included in this repository** (`src/xc_viewer/assets/`):

- small terrain extracts from IGN RGE ALTI, the crest lines derived from them, and
  summit points from IGN BD TOPO: © IGN,
  [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
  (details in `assets/thermal_orography/README.md`);
- country outlines from [Natural Earth](https://www.naturalearthdata.com/): public
  domain.

**Downloaded while the viewer runs:**

- IGN maps, aerial photos (BD ORTHO), elevation contours and terrain (RGE ALTI) from
  the [Géoplateforme](https://geoservices.ign.fr/): © IGN, Licence Ouverte 2.0;
- shaded relief from
  [Esri World Hillshade](https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer):
  © Esri and its data contributors, under Esri's terms of use.
