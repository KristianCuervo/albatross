"""
run_migration.py — Compute and save greedy + Hamiltonian IVP migration data.

Outputs
-------
    data/migration_jan2023_snippet.npz     greedy RK4 positions
    data/hamiltonian_jan2023_snippet.npz   Hamiltonian IVP fan

Visualise results with:
    python scripts/macroscale/visualise_greedy.py   (greedy)
    python scripts/macroscale/visualise_ham.py      (Hamiltonian)

Usage
-----
    python scripts/run_migration.py [--no-greedy] [--no-ham]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from albatross.macroscale import (
    VelocityHull, ERA5Interpolator, GreedyMigration,
    HamiltonianShooter, configure_simulation,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
ERA5_DIR   = DATA_DIR / "era5"
GREEDY_NPZ = DATA_DIR / "macroscale" / "migration_jan2023_snippet.npz"
HAM_NPZ    = DATA_DIR / "macroscale" / "hamiltonian_jan2023_snippet.npz"

# ── Parameters ─────────────────────────────────────────────────────────────────
START_LAT  = -54.0
START_LON  = -38.05
START_TIME = "2023-01-09T18:00:00"
END_TIME   = "2023-01-13T05:00:00"

DT     = 900.0   # greedy time step [s]
N_DIRS = 360     # greedy directions
N_HAM  = 720     # Hamiltonian initial costate angles

LAT_RANGE = (-30.0, -75.0)   # (north, south) — matches explorer scripts


def parse_args():
    p = argparse.ArgumentParser(description="Run greedy + Hamiltonian migration simulation.")
    p.add_argument("--no-greedy", action="store_true", help="Skip greedy IVP")
    p.add_argument("--no-ham",    action="store_true", help="Skip Hamiltonian IVP")
    return p.parse_args()


def main():
    args = parse_args()

    start_unix = datetime.fromisoformat(START_TIME).replace(tzinfo=timezone.utc).timestamp()
    end_unix   = datetime.fromisoformat(END_TIME).replace(tzinfo=timezone.utc).timestamp()
    N_STEPS    = int(round((end_unix - start_unix) / DT))
    T_TOTAL    = end_unix - start_unix

    print(f"Window : {START_TIME}  →  {END_TIME}")
    print(f"Greedy : {N_STEPS} steps  ({N_STEPS * DT / 3600:.1f} h at dt={DT:.0f} s)")
    print(f"Ham    : T = {T_TOTAL / 3600:.1f} h  ({N_HAM} initial costate angles)")

    # ── ERA5 ──────────────────────────────────────────────────────────────────
    nc_files = sorted(ERA5_DIR.glob("era5_1h_so_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 files in {ERA5_DIR}")

    print(f"\nLoading ERA5 ({len(nc_files)} files on disk, filtering to window) …")
    era5 = ERA5Interpolator(
        nc_files,
        lat_range  = LAT_RANGE,
        time_range = (start_unix, end_unix + 3600),
    )
    print(f"  Loaded {era5._u10.shape}  ({era5._u10.nbytes / 1e6:.0f} MB)")

    # ── Hull ──────────────────────────────────────────────────────────────────
    hull = VelocityHull.from_npz(DATA_DIR / "mesoscale" / "velocity_hulls.npz")
    print(f"  Hull: V_ref {hull.v_ref_levels.min():.1f}–{hull.v_ref_levels.max():.1f} m/s")

    # ── Greedy IVP ────────────────────────────────────────────────────────────
    if not args.no_greedy:
        print(f"\nRunning greedy IVP ({N_STEPS} steps × {N_DIRS} directions) …")
        sim       = GreedyMigration(hull=hull, era5=era5)
        positions = sim.run(
            start      = (START_LAT, START_LON),
            start_time = start_unix,
            n_steps    = N_STEPS,
            dt         = DT,
            n_dirs     = N_DIRS,
        )
        sim.save(GREEDY_NPZ)
        print(f"  positions: {positions.shape}")
    else:
        print("\nSkipping greedy IVP (--no-greedy)")

    # ── Hamiltonian IVP ───────────────────────────────────────────────────────
    if not args.no_ham:
        print(f"\nShooting fan of {N_HAM} Hamiltonian trajectories …")
        configure_simulation(hull, era5)
        shooter = HamiltonianShooter(hull, era5, n_headings=360)
        fan = shooter.shoot_fan(
            x0       = (START_LAT, START_LON),
            t0       = start_unix,
            T        = T_TOTAL,
            n_angles = N_HAM,
            rtol     = 1e-3,
            atol     = 1e-5,
        )
        print(f"  Fan: {len(fan)} trajectories")

        max_nt = max(len(r["ts"]) for r in fan)

        def _pad(arr, n):
            return np.pad(arr.astype(float), (0, n - len(arr)), constant_values=np.nan)

        np.savez_compressed(
            HAM_NPZ,
            ys   = np.array([_pad(r["ys"],   max_nt) for r in fan]),
            xs   = np.array([_pad(r["xs"],   max_nt) for r in fan]),
            ts   = np.array([_pad(r["ts"],   max_nt) for r in fan]),
            lam1 = np.array([_pad(r["lam1"], max_nt) for r in fan]),
            lam2 = np.array([_pad(r["lam2"], max_nt) for r in fan]),
            phis = np.array([r["phi"]                for r in fan]),
        )
        print(f"  Saved → {HAM_NPZ}")
    else:
        print("\nSkipping Hamiltonian IVP (--no-ham)")

    print("\nDone.")


if __name__ == "__main__":
    main()
