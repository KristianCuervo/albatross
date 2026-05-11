# Albatross

Dynamic soaring and migration modelling for the wandering albatross (*Diomedea exulans*), based on the aerodynamic parameters from Sachs (2005).

The analysis is organised across three scales:

- **Microscale** — 3-D optimal dynamic soaring cycles solved via CasADi (trajectory shape, minimum wind speed, tacking behaviour)
- **Mesoscale** — Polar/tacking diagrams: maximum VMG as a function of heading and wind speed, producing the convex velocity hull used by the migration model
- **Macroscale** — 2-D migration over the Southern Ocean using ERA5 reanalysis wind; both a greedy RK4 integrator and a Hamiltonian (Pontryagin) optimal control solver

## Repository layout

Each scale is a self-contained directory with its own library modules, analysis scripts, precomputed data, and output figures.

```
microscale/
  bird.py               Albatross dataclass, loads from albatross.toml
  solver.py             CasADi Opti solver (max_vmg / min_vref modes)
  container.py          Single-run result container
  ensemble.py           Collection of containers + plotting
  factory.py            Batch sweep + parallel execution
  sachs_analysis.py     Reproduces 8.6 m/s Sachs (2005) result
  convergence_study.py  Grid-resolution convergence check
  coordinate_frames.py  Interactive coordinate convention diagrams
  wind_shear.py         Wind shear profile visualisation
  albatross.toml        Bird parameters (Sachs 2005)
  data/                 sachs_result.npz
  figures/              Time-series, 3-D trajectory, projection plots

mesoscale/
  velocity_hull.py      Convex velocity hull (micro → macro bridge)
  tacking_diagram.py    VMG polar sweep + trajectory comparisons
  vref_min_polar.py     Minimum wind speed polar diagram
  symmetry_breaking.py  Symmetry-breaking trajectory analysis
  data/                 tacking_diagram.npz, velocity_hulls.npz, tacking_mirrored.npz
  figures/              Polar diagrams, tacking diagram, trajectory comparisons

macroscale/
  wind.py               Wind base class (uniform shear model)
  realWind.py           ERA5Interpolator + RealWind (spatiotemporal ERA5 wind)
  hull.py               Velocity hull interface for the migration system
  system.py             Hamiltonian system (costate ODE)
  integrator.py         Leapfrog integrator
  shooter.py            Trajectory fan shooter
  derivative.py         Numerical derivative utilities
  utils.py              Coordinate and unit utilities
  storage.py            Trajectory storage / serialisation
  visualise.py          Trajectory and wind field visualisation
  download_era5.py      CDS API downloader for ERA5 NetCDF files
  seasonal_wind_maps.py Seasonal-mean wind maps (DJF/MAM/JJA/SON)
  wind_explorer.py      Interactive ERA5 wind viewer
  data/era5/            ERA5 NetCDF files — not tracked, ~3 GB
  figures/              Migration isocurves, GIFs, seasonal wind maps
```

## Installation

```bash
git clone https://github.com/KristianCuervo/albatross.git
cd albatross
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For Cartopy (needed for macroscale map scripts):
```bash
pip install cartopy
```

For ERA5 downloading:
```bash
pip install cdsapi   # also requires ~/.cdsapirc credentials
```

## Usage

**Microscale — reproduce Sachs (2005) minimum wind speed:**
```bash
python microscale/sachs_analysis.py
```

**Mesoscale — tacking diagram and trajectory comparisons:**
```bash
python mesoscale/tacking_diagram.py
```

**Macroscale — download ERA5 data and explore wind:**
```bash
python macroscale/download_era5.py   # downloads to macroscale/data/era5/
python macroscale/wind_explorer.py   # interactive ERA5 wind viewer
```

All analysis scripts default to `RECOMPUTE = False` and load precomputed results from their local `data/` directory.

## ERA5 data

ERA5 files are not tracked in git. Two datasets are used:

| Dataset | Pattern | Resolution | Coverage |
|---|---|---|---|
| Southern Ocean 1-hourly | `era5_1h_so_YYYY_MM_DD.nc` | 0.25° × 1 h | Dec 2022 – Feb 2023 |
| Global 6-hourly | `era5_6h_global_YYYY_MM.nc` | 1.0° × 6 h | Dec 2022 – Nov 2023 |

Download both with `macroscale/download_era5.py` (requires a CDS API key).

## Reference

Sachs, G. (2005). Minimum shear wind strength required for dynamic soaring.
*Journal of Ornithology*, 146(1), 74–84.
