"""
costate_analysis.py — State and costate analysis for grouped Hamiltonian trajectories.

Two-panel figure:
  Top    : distance from Bird Island (km) over time — the objective
  Bottom : costate magnitude  |λ| = sqrt(λ1² + λ2²) over time

All 360 trajectories are grouped into 6 bands of 60° initial costate angle:
  [0–60°), [60–120°), [120–180°), [180–240°), [240–300°), [300–360°)

Each group is drawn with:
  - transparent lines for every individual trajectory in the group
  - a solid thick line for the group mean (interpolated onto a common hourly grid)

Usage
-----
    python scripts/costate_analysis.py
    python scripts/costate_analysis.py --npz data/hamiltonian_jan2023_snippet.npz
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUP_DEG  = 60          # degrees per group
N_GROUPS   = 360 // GROUP_DEG   # 6
ALPHA_IND  = 0.12        # transparency for individual lines
LW_IND     = 0.7         # line width for individual lines
LW_MEAN    = 2.2         # line width for group mean


def parse_args():
    p = argparse.ArgumentParser(description="Grouped state + costate analysis.")
    p.add_argument(
        "--npz", type=Path,
        default=ROOT / "data" / "macroscale" / "hamiltonian_jan2023_snippet.npz",
        help="Path to Hamiltonian NPZ (must include lam1, lam2)",
    )
    p.add_argument(
        "--lam-max", type=float, default=5.0,
        help="Exclude trajectories whose |λ| ever exceeds this value (default: 5.0)",
    )
    return p.parse_args()


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    dlat  = np.deg2rad(lat2_arr - lat1)
    dlon  = np.deg2rad(lon2_arr - lon1)
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2_arr)
    a = np.sin(dlat / 2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def interp_to_grid(t_i, vals_i, t_grid):
    """
    Linearly interpolate a variable-length trajectory onto a common time grid.
    Points outside [t_i[0], t_i[-1]] are set to NaN.
    """
    out = np.interp(t_grid, t_i, vals_i, left=np.nan, right=np.nan)
    return out


def main():
    args = parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(
            f"{args.npz} not found.\n"
            "Re-run: python scripts/run_migration.py --no-greedy"
        )

    d = np.load(args.npz)
    if "lam1" not in d or "lam2" not in d:
        raise KeyError(
            "NPZ missing lam1/lam2 — re-run: python scripts/run_migration.py --no-greedy"
        )

    ys   = d["ys"]
    xs   = d["xs"]
    ts   = d["ts"]
    lam1 = d["lam1"]
    lam2 = d["lam2"]
    phis = d["phis"]

    n_dirs    = len(phis)
    start_lat = float(ys[0, 0])
    start_lon = float(xs[0, 0])
    t0        = float(np.nanmin(ts))
    t_end     = float(np.nanmax(ts))

    # ── Pre-compute max |λ| per trajectory and build outlier mask ─────────────
    lam_mag_max = np.array([
        float(np.nanmax(np.sqrt(lam1[i]**2 + lam2[i]**2)))
        for i in range(n_dirs)
    ])
    outlier_mask = lam_mag_max > args.lam_max
    n_outliers   = int(outlier_mask.sum())
    if n_outliers:
        print(f"Filtering {n_outliers} outlier trajectory/trajectories "
              f"with max |λ| > {args.lam_max} "
              f"(φ = {np.degrees(phis[outlier_mask]).round(1).tolist()}°)")

    # Common hourly time grid for averaging
    n_hours  = int(np.floor((t_end - t0) / 3600))
    t_grid   = t0 + np.arange(n_hours + 1) * 3600.0
    h_grid   = (t_grid - t0) / 3600.0

    # ── Pre-compute per-trajectory distance and |λ| on their native grids ─────
    phis_deg = np.degrees(phis) % 360.0

    # Group colours: one per 60° band
    group_colors = plt.cm.tab10(np.linspace(0, 0.6, N_GROUPS))

    fig, (ax_dist, ax_lam) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    legend_handles = []

    for g in range(N_GROUPS):
        lo_deg = g * GROUP_DEG
        hi_deg = (g + 1) * GROUP_DEG
        color  = group_colors[g]
        label  = f"φ ∈ [{lo_deg}°, {hi_deg}°)"

        # Indices belonging to this group, excluding outliers
        in_group = np.where(
            (phis_deg >= lo_deg) & (phis_deg < hi_deg) & ~outlier_mask
        )[0]

        # Accumulate interpolated arrays for the group mean
        dist_grid_all = np.full((len(in_group), len(t_grid)), np.nan)
        lam_grid_all  = np.full((len(in_group), len(t_grid)), np.nan)

        for row, idx in enumerate(in_group):
            mask = ~np.isnan(ts[idx])
            t_i  = ts[idx][mask]
            y_i  = ys[idx][mask]
            x_i  = xs[idx][mask]
            l1_i = lam1[idx][mask]
            l2_i = lam2[idx][mask]

            if len(t_i) < 2:
                continue

            dist_i  = haversine_km(start_lat, start_lon, y_i, x_i)
            lam_i   = np.sqrt(l1_i**2 + l2_i**2)
            hours_i = (t_i - t0) / 3600.0

            # Individual transparent lines
            ax_dist.plot(hours_i, dist_i, color=color, lw=LW_IND, alpha=ALPHA_IND)
            ax_lam.plot( hours_i, lam_i,  color=color, lw=LW_IND, alpha=ALPHA_IND)

            # Interpolate onto common grid for averaging
            dist_grid_all[row] = interp_to_grid(t_i, dist_i,  t_grid)
            lam_grid_all[row]  = interp_to_grid(t_i, lam_i,   t_grid)

        # Group mean (nanmean ignores trajectories that ended early)
        dist_mean = np.nanmean(dist_grid_all, axis=0)
        lam_mean  = np.nanmean(lam_grid_all,  axis=0)

        ax_dist.plot(h_grid, dist_mean, color=color, lw=LW_MEAN)
        ax_lam.plot( h_grid, lam_mean,  color=color, lw=LW_MEAN)

        legend_handles.append(
            mpatches.Patch(color=color, label=label)
        )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax_dist.set_ylabel("Distance from Bird Island  [km]", fontsize=12)
    ax_dist.legend(handles=legend_handles, loc="upper left",
                   fontsize=10, framealpha=0.85, ncol=2)
    ax_dist.grid(True, linewidth=0.5, alpha=0.5)
    n_kept = n_dirs - n_outliers
    ax_dist.set_title(
        "Hamiltonian IVP — grouped state & costate analysis  "
        f"(groups of {GROUP_DEG}°)\n"
        f"Start: {datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M UTC')}  "
        f"|  {n_kept}/{n_dirs} trajectories  "
        f"|  filtered max |λ| > {args.lam_max}  |  solid = group mean",
        fontsize=12, pad=8,
    )

    ax_lam.set_ylabel("Costate magnitude  |λ| = √(λ₁² + λ₂²)", fontsize=12)
    ax_lam.set_xlabel("Hours from start", fontsize=12)
    ax_lam.grid(True, linewidth=0.5, alpha=0.5)
    ax_lam.axhline(1.0, color="black", lw=0.9, linestyle="--", alpha=0.45)
    ax_lam.text(h_grid[-1] * 0.98, 1.02, "|λ| = 1 (initial)",
                ha="right", va="bottom", fontsize=9, alpha=0.6)
    ax_lam.legend(handles=legend_handles, loc="upper left",
                  fontsize=10, framealpha=0.85, ncol=2)

    plt.tight_layout()

    out = ROOT / "figures" / "macroscale" / "costate_analysis.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")

    plt.show()


if __name__ == "__main__":
    main()
