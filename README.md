# Albatross

Dynamic soaring and migration modelling for the wandering albatross (*Diomedea exulans*), based on the aerodynamic parameters from Sachs (2005).

The analysis is organised across three scales:

- **Microscale** — 3-D optimal dynamic soaring cycles solved via CasADi (trajectory shape, minimum wind speed, tacking behaviour)
- **Mesoscale** — Polar/tacking diagrams: maximum VMG as a function of heading and wind speed, producing the convex velocity hull used by the migration model
- **Macroscale** — 2-D migration over the Southern Ocean using ERA5 reanalysis wind; both a greedy RK4 integrator and a Hamiltonian (Pontryagin) optimal control solver

## Repository layout

```
albatross/          Installable Python package
  microscale/       CasADi solver, result container, ensemble, batch factory
  macroscale/       ERA5 interpolator, velocity hull, greedy migration, Hamiltonian IVP

scripts/
  microscale/       sachs_analysis.py      — reproduces 8.6 m/s Sachs (2005) result
                    coordinate_frames.py   — interactive coordinate convention diagrams
  mesoscale/        tacking_diagram.py     — VMG polar sweep + trajectory comparisons
  macroscale/       run_migration.py       — compute greedy + Hamiltonian IVP
                    wind_explorer.py       — interactive ERA5 wind viewer
                    visualise_greedy.py    — interactive greedy trajectory viewer
                    visualise_ham.py       — interactive Hamiltonian trajectory viewer
                    costate_analysis.py    — grouped costate magnitude analysis
                    seasonal_wind_maps.py  — seasonal-mean wind maps (DJF/MAM/JJA/SON)
                    download_era5.py       — CDS API downloader for ERA5 NetCDF files

data/
  albatross.toml    Bird parameters (Sachs 2005)
  microscale/       sachs_result.npz
  mesoscale/        velocity_hulls.npz, tacking_diagram.npz
  macroscale/       migration and Hamiltonian IVP results (various windows)
  era5/             ERA5 NetCDF files — not tracked, download with download_era5.py

figures/
  microscale/       Time-series, 3-D trajectory, projection plots
  mesoscale/        Polar diagrams, tacking diagram, trajectory comparisons
  macroscale/       Migration isocurves, GIFs, seasonal wind maps
```

## Installation

```bash
git clone https://github.com/KristianCuervo/albatross.git
cd albatross
python -m venv .venv && source .venv/bin/activate
pip install -e ".[maps,era5]"
```

`maps` adds Cartopy (needed for migration and wind map scripts).  
`era5` adds cdsapi (needed for `download_era5.py`; requires `~/.cdsapirc` credentials).

## Usage

**Microscale — reproduce Sachs (2005) minimum wind speed:**
```bash
python scripts/microscale/sachs_analysis.py
```

**Mesoscale — tacking diagram and trajectory comparisons:**
```bash
python scripts/mesoscale/tacking_diagram.py
```

**Macroscale — compute and explore migration:**
```bash
# Download ERA5 wind data first
python scripts/macroscale/download_era5.py

# Run migration simulation
python scripts/macroscale/run_migration.py

# Explore results interactively
python scripts/macroscale/visualise_greedy.py
python scripts/macroscale/visualise_ham.py
```

All scripts default to `RECOMPUTE = False` and load precomputed results from `data/`.

## Reference

Sachs, G. (2005). Minimum shear wind strength required for dynamic soaring.
*Journal of Ornithology*, 146(1), 74–84.
