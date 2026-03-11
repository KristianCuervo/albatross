"""
build_velocity_hull.py
======================
Extracts the two-cycle achievable velocity set (convex hull) from the
tacking diagram data and overlays it on the ERA5 wind field.

Physics
-------
For a given wind-shear reference speed V_ref, the albatross can reach any
average velocity lying inside the convex hull of the symmetric single-cycle
set  { (u, v) } ∪ { (-u, v) }  from tacking_diagram.npz.

Key identity
------------
  V_ref ≈ ERA5 wind speed at 10 m

The model uses a power-law shear profile  V_wy(h) = V_ref * (h/10)^0.143,
so at h = 10 m the wind equals V_ref exactly.  ERA5 u10, v10 give the wind
components at 10 m, so  V_ref_local = sqrt(u10² + v10²)  at each grid point.

Coordinate conventions
----------------------
Model (wind-relative) frame:
  u  — crosswind  (m/s, positive = right when facing upwind)
  v  — upwind     (m/s, positive = into the wind)

Polar angle convention (matches tacking_diagram_plots.ipynb):
  angle = 0     → headwind  (u=0, v>0)  — top of polar plot
  angle = π/2   → crosswind (u>0, v=0)  — right
  angle = π     → tailwind  (u=0, v<0)  — bottom

Geographic frame (ERA5):
  east  = u10  component
  north = v10  component

Rotation from model → geographic at a grid point with wind (u10, v10):
  upwind unit vector  = (-u10/V, -v10/V)            (into wind source)
  crosswind unit vec  = (-v10/V,  u10/V)            (90° CW from upwind)

  [east ]   [-v10/V  -u10/V] [u_model]
  [north] = [ u10/V  -v10/V] [v_model]

Outputs
-------
CG2/data/velocity_hulls.npz
    v_ref_levels  (n_v,)          V_ref grid [m/s]
    hull_radii    (n_v, n_rays)   boundary radius per ray, wind frame [m/s]
    hull_angles   (n_rays,)       ray angles [rad]
    pts_u         (n_total,)      all (u_avg ∪ −u_avg) points
    pts_v         (n_total,)      all corresponding v_avg points
    pts_vref      (n_total,)      V_ref label per point

CG2/data/wind_hull_overlay.npz   (requires era5_wind_atlantic_2023_07.nc)
    lat                  (n_lat,)               ERA5 latitudes
    lon                  (n_lon,)               ERA5 longitudes
    v_ref_local          (n_lat, n_lon)         local V_ref = |wind| at 10 m [m/s]
    wind_dir             (n_lat, n_lon)         direction wind blows TO, CW from N [rad]
    upwind_bearing       (n_lat, n_lon)         direction wind comes FROM, CW from N [rad]
    max_speed_achievable (n_lat, n_lon)         max hull radius at local V_ref [m/s]
    hull_radii_geo       (n_lat, n_lon, n_rays) achievable speed toward geo_bearings[k]
                                                at each grid point [m/s] — ROTATED to
                                                geographic frame, index k is consistent
                                                across all grid points
    geo_bearings         (n_rays,)              geographic bearings [rad], CW from North

Usage
-----
    python build_velocity_hull.py            # build both outputs if ERA5 exists
    python build_velocity_hull.py --hull-only  # skip ERA5 overlay
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
TACKING_NPZ   = HERE.parent / "refactor" / "data" / "tacking_diagram.npz"
ERA5_NC_DEFAULT = HERE / "data" / "era5_wind_eq_atlantic_2023_07.nc"
OUT_HULLS       = HERE / "data" / "velocity_hulls.npz"

N_RAYS = 720  # angular resolution for hull boundary tracing


# ─── Hull tracing ─────────────────────────────────────────────────────────────

def _make_ray_dirs(n_rays: int):
    """Unit ray directions in (crosswind, upwind) model space."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dx = np.sin(angles)   # crosswind component
    dy = np.cos(angles)   # upwind    component
    return angles, dx, dy


def trace_hull_boundary(pts: np.ndarray, dx_rays: np.ndarray, dy_rays: np.ndarray) -> np.ndarray:
    """
    Shoot rays from the origin through the convex hull of pts and return the
    radius (distance from origin) of each intersection.  NaN if no intersection.

    Parameters
    ----------
    pts     : (n, 2)  array of (u, v) points
    dx_rays : (n_rays,)  crosswind component of each unit ray
    dy_rays : (n_rays,)  upwind    component of each unit ray

    Returns
    -------
    radii   : (n_rays,)  intersection distances, NaN where miss
    """
    n_rays = len(dx_rays)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.full(n_rays, np.nan)

    hv  = pts[hull.vertices]
    hvc = np.vstack([hv, hv[0]])      # close the vertex loop
    radii = np.full(n_rays, np.nan)

    for i in range(len(hvc) - 1):
        x1, y1 = hvc[i]
        x2, y2 = hvc[i + 1]
        ex, ey = x2 - x1, y2 - y1

        denom = dx_rays * ey - dy_rays * ex
        ok    = np.abs(denom) > 1e-12

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(ok, (x1 * ey - y1 * ex) / denom, np.nan)
            s = np.where(ok, (x1 * dy_rays - y1 * dx_rays) / denom, np.nan)

        valid  = ok & (t > 1e-9) & (s >= -1e-9) & (s <= 1 + 1e-9)
        update = valid & (np.isnan(radii) | (t < radii))
        radii  = np.where(update, t, radii)

    return radii


# ─── Stage 1: hull table ──────────────────────────────────────────────────────

def build_hull_table():
    """Compute per-V_ref convex hull boundaries and save to velocity_hulls.npz."""
    if not TACKING_NPZ.exists():
        sys.exit(
            f"ERROR: tacking diagram data not found at {TACKING_NPZ}\n"
            "Run  refactor/tacking_diagram.py  first to generate it."
        )

    d = np.load(TACKING_NPZ)
    u_avg  = d['u_avg']
    v_avg  = d['v_avg']
    V_ref  = d['V_ref']
    v_ref_levels = np.array(sorted(set(V_ref.tolist())), dtype=float)

    print(f"Loaded {TACKING_NPZ.name}")
    print(f"  {len(v_ref_levels)} V_ref levels: "
          f"{v_ref_levels.min():.0f}–{v_ref_levels.max():.0f} m/s  |  "
          f"{len(u_avg)} data points\n")

    angles_ray, dx_rays, dy_rays = _make_ray_dirs(N_RAYS)
    hull_radii   = np.full((len(v_ref_levels), N_RAYS), np.nan)
    pts_u_list, pts_v_list, pts_vref_list = [], [], []

    for idx, vr in enumerate(v_ref_levels):
        mask  = V_ref == vr
        uc, vc = u_avg[mask], v_avg[mask]

        # Symmetric set (u → -u mirrors the left/right tacking option)
        u_all = np.concatenate([ uc, -uc])
        v_all = np.concatenate([ vc,  vc])
        pts   = np.unique(np.column_stack([u_all, v_all]), axis=0)

        radii = trace_hull_boundary(pts, dx_rays, dy_rays)
        hull_radii[idx] = radii

        pts_u_list.append(u_all)
        pts_v_list.append(v_all)
        pts_vref_list.append(np.full(len(u_all), vr))

        n_valid = int(np.sum(~np.isnan(radii)))
        max_r   = float(np.nanmax(radii)) if n_valid > 0 else float('nan')
        print(f"  V_ref = {vr:5.1f} m/s  "
              f"hull pts = {len(pts):3d}  "
              f"max speed = {max_r:6.2f} m/s  "
              f"rays hit = {n_valid}/{N_RAYS}")

    OUT_HULLS.parent.mkdir(exist_ok=True)
    np.savez(
        OUT_HULLS,
        v_ref_levels = v_ref_levels,
        hull_radii   = hull_radii,
        hull_angles  = angles_ray,
        pts_u        = np.concatenate(pts_u_list),
        pts_v        = np.concatenate(pts_v_list),
        pts_vref     = np.concatenate(pts_vref_list),
    )
    print(f"\nSaved → {OUT_HULLS}")
    return v_ref_levels, hull_radii, angles_ray


# ─── Stage 2: ERA5 overlay ────────────────────────────────────────────────────

def build_wind_overlay(v_ref_levels: np.ndarray, hull_radii: np.ndarray,
                       angles_ray: np.ndarray, era5_path: Path = None):
    """
    For each ERA5 grid point, interpolate the hull to the local V_ref and
    resample it onto a fixed geographic bearing grid.

    The hull is computed in the wind frame (angle=0 = upwind direction).
    At each grid point the wind direction differs, so a direct index comparison
    across points would be comparing different geographic directions.  We fix
    this by rotating each point's hull so that index k always corresponds to
    the same geographic bearing (geo_bearings[k]) everywhere.

    Rotation
    --------
    The upwind direction in geographic (CW-from-North) bearings is:
        upwind_bearing = arctan2(-u10, -v10)   [= wind_dir + π  mod 2π]

    For a geographic bearing β, the equivalent wind-frame angle is:
        θ_wind = β - upwind_bearing  (mod 2π)

    We evaluate the hull at θ_wind for each β by circular interpolation
    on the (angles_ray, hull_radii) table.
    """
    era5_nc = Path(era5_path) if era5_path else ERA5_NC_DEFAULT
    # Output filename mirrors the ERA5 input: era5_wind_*.nc → wind_hull_overlay_*.npz
    stem        = era5_nc.stem.replace("era5_wind_", "")
    out_overlay = HERE / "data" / f"wind_hull_overlay_{stem}.npz"

    if not era5_nc.exists():
        print(f"\nERA5 file not found at {era5_nc}")
        print("Run download_era5.py first, then re-run to build the wind overlay.")
        return

    import xarray as xr

    print(f"\nBuilding wind–hull overlay from {era5_nc.name} ...")
    ds = xr.open_dataset(era5_nc)

    def _get(ds, *names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"None of {names} found. Available: {list(ds.data_vars)}")

    u10 = _get(ds, "u10", "u_10m").squeeze().values   # (n_lat, n_lon)
    v10 = _get(ds, "v10", "v_10m").squeeze().values
    lat = ds.latitude.values
    lon = ds.longitude.values
    n_lat, n_lon = lat.size, lon.size
    n_pts = n_lat * n_lon

    # V_ref at each point = ERA5 wind speed at 10 m
    v_ref_local = np.sqrt(u10**2 + v10**2)             # (n_lat, n_lon)

    # Wind direction: bearing wind blows TO, CW from North [rad]
    wind_dir = np.arctan2(u10, v10) % (2 * np.pi)      # (n_lat, n_lon)

    # Upwind bearing: direction toward wind source, CW from North [rad]
    # This is what angle=0 in the wind frame corresponds to geographically.
    upwind_bearing = (wind_dir + np.pi) % (2 * np.pi)  # (n_lat, n_lon)

    # ── Step 1: interpolate hull boundary to local V_ref ─────────────────
    v_ref_clamped = np.clip(v_ref_local.ravel(), v_ref_levels.min(), v_ref_levels.max())
    idx_float     = np.interp(v_ref_clamped, v_ref_levels, np.arange(len(v_ref_levels)))

    lo   = np.floor(idx_float).astype(int).clip(0, len(v_ref_levels) - 2)
    hi   = (lo + 1).clip(0, len(v_ref_levels) - 1)
    frac = (idx_float - lo)[:, None]                   # (n_pts, 1)

    # hull_interp[p, k] = hull radius at wind-frame angle angles_ray[k] for point p
    hull_interp = (1 - frac) * hull_radii[lo] + frac * hull_radii[hi]
    # shape: (n_pts, N_RAYS)

    # ── Step 2: rotate onto fixed geographic bearing grid ────────────────
    # geo_bearings is the shared angle axis: bearing 0 = North, π/2 = East, etc.
    geo_bearings = angles_ray.copy()   # same 720-point grid, now in geographic frame

    # For each point p and each geographic bearing β:
    #   θ_wind = β - upwind_bearing[p]   (mod 2π)
    # We then evaluate hull_interp[p] at θ_wind by circular interpolation.

    upwind_flat = upwind_bearing.ravel()          # (n_pts,)

    # Duplicate the hull one period each side to allow circular interpolation
    # without boundary wrapping logic.  angles_ray spans [0, 2π).
    angles_ext = np.concatenate([angles_ray - 2*np.pi, angles_ray, angles_ray + 2*np.pi])
    hull_ext   = np.concatenate([hull_interp, hull_interp, hull_interp], axis=1)
    # shapes: (3*N_RAYS,) and (n_pts, 3*N_RAYS)

    hull_geo = np.full((n_pts, N_RAYS), np.nan)

    # Process in batches to avoid building a (n_pts × N_RAYS) lookup all at once
    BATCH = 2000
    for start in range(0, n_pts, BATCH):
        end   = min(start + BATCH, n_pts)
        ub    = upwind_flat[start:end]            # (batch,)

        # wind-frame angles needed for each geographic bearing
        # shape: (batch, N_RAYS)
        theta_wind = (geo_bearings[None, :] - ub[:, None]) % (2 * np.pi)

        # Circular interpolation: find position in angles_ext
        for b in range(end - start):
            hull_geo[start + b] = np.interp(
                theta_wind[b],
                angles_ext,
                hull_ext[start + b],
            )

    max_speed_achievable = np.nanmax(hull_geo, axis=1).reshape(n_lat, n_lon)
    hull_radii_geo       = hull_geo.reshape(n_lat, n_lon, N_RAYS)

    np.savez(
        out_overlay,
        lat                  = lat,
        lon                  = lon,
        v_ref_local          = v_ref_local,
        wind_dir             = wind_dir,
        upwind_bearing       = upwind_bearing,
        max_speed_achievable = max_speed_achievable,
        # hull_radii_geo[i, j, k] = achievable speed toward geo_bearings[k] at grid point (i,j)
        hull_radii_geo       = hull_radii_geo,
        geo_bearings         = geo_bearings,
    )
    print(f"Saved → {out_overlay}")
    print(f"  Grid     : {n_lat} lat × {n_lon} lon = {n_pts:,} points")
    print(f"  V_ref    : {v_ref_local.min():.2f} – {v_ref_local.max():.2f} m/s")
    print(f"  Max achiev. speed: {max_speed_achievable.min():.2f} – "
          f"{max_speed_achievable.max():.2f} m/s")
    return out_overlay


# ─── Convenience loader ───────────────────────────────────────────────────────

def load_hull_table(path=OUT_HULLS):
    """
    Load the pre-computed hull table.

    Returns
    -------
    dict with keys:
        v_ref_levels  (n_v,)
        hull_radii    (n_v, n_rays)
        hull_angles   (n_rays,)
        pts_u, pts_v, pts_vref  — raw symmetric point cloud
    """
    d = np.load(path)
    return {k: d[k] for k in d.files}


def interpolate_hull(v_ref_query: float, hull_data: dict) -> np.ndarray:
    """
    Return the hull boundary radii at an arbitrary V_ref by linear interpolation.

    Parameters
    ----------
    v_ref_query : float   target V_ref [m/s], clamped to table range
    hull_data   : dict    from load_hull_table()

    Returns
    -------
    radii : (n_rays,)  boundary radii in wind-relative frame [m/s]
            angles are hull_data['hull_angles'] — use upwind_bearing to rotate
            to geographic frame if needed (see build_wind_overlay for the method)
    """
    levels = hull_data['v_ref_levels']
    radii  = hull_data['hull_radii']
    v      = float(np.clip(v_ref_query, levels.min(), levels.max()))
    idx    = float(np.interp(v, levels, np.arange(len(levels))))
    lo     = int(np.floor(idx))
    hi     = min(lo + 1, len(levels) - 1)
    frac   = idx - lo
    return (1 - frac) * radii[lo] + frac * radii[hi]


# ─── Verification plots ───────────────────────────────────────────────────────

def verify(hull_only: bool = False, out_overlay: Path = None):
    """
    Generate verification plots for the two output files.

    Figure 1 — Hull table (Stage 1, wind frame)
        Left panel:  polar plot of all 17 V_ref hull boundaries.
                     Should replicate the notebook's two-cycle achievable velocity plot.
                     Curves should expand outward with increasing V_ref.
        Right panel: Cartesian scatter of the symmetric point cloud + convex hull polygon
                     for a mid-range V_ref.  Hull should tightly wrap all points and be
                     symmetric around the v-axis (upwind direction).

    Figure 2 — Wind speed and max achievable speed (Stage 2, requires ERA5)
        Two panels side-by-side:
        (a) ERA5 wind speed at 10 m.  The blue contour marks V_ref_min (9 m/s),
            the lower edge of the tacking data.  Most of the equatorial Atlantic
            in July lies BELOW this threshold (trade winds ~4–8 m/s).
        (b) Max achievable speed.  Hatched region = wind below V_ref_min; the hull
            is clamped to the weakest tacking level there, so color is uniform.
            Variation only appears where wind exceeds 9 m/s.
        A uniform color in (b) is therefore expected and physically correct —
        it means the tacking data does not cover the actual wind regime.

    Figure 3 — Rotation verification (Stage 2, critical test)
        Four panels, each showing the hull boundary in the GEOGRAPHIC frame at a grid
        point where the wind blows approximately N, E, S, W.
        PASS: the max-speed lobe of each hull aligns with the red "wind→" arrow.
        FAIL: if the rotation is wrong, the lobe will be misaligned — e.g. pointing
              north regardless of which way the wind blows.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.spatial import ConvexHull as _ConvexHull

    FIGS_DIR = HERE / "figures"
    FIGS_DIR.mkdir(exist_ok=True)

    if not OUT_HULLS.exists():
        print("Hull table not found — run build first, then verify.")
        return

    hull_data    = load_hull_table()
    v_ref_levels = hull_data['v_ref_levels']
    hull_radii   = hull_data['hull_radii']
    hull_angles  = hull_data['hull_angles']
    pts_u        = hull_data['pts_u']
    pts_v        = hull_data['pts_v']
    pts_vref     = hull_data['pts_vref']

    cmap = plt.cm.plasma
    norm = Normalize(vmin=v_ref_levels.min(), vmax=v_ref_levels.max())

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 1 — Hull table verification (wind frame)
    # ═══════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(14, 6))
    ax_pol = fig1.add_subplot(121, projection='polar')
    ax_xy  = fig1.add_subplot(122)

    # Left: all hull boundaries
    for idx, vr in enumerate(v_ref_levels):
        color   = cmap(norm(vr))
        r       = hull_radii[idx]
        valid   = ~np.isnan(r)
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue
        a_arc = hull_angles[valid]
        r_arc = r[valid]
        if n_valid < len(hull_angles):
            # Partial arc (V_ref too low for full-circle DS): fill the achievable
            # wedge between the arc boundary and the origin so it reads as a
            # partial region rather than a stray line segment.
            a_fill = np.concatenate([[a_arc[0]], a_arc, [a_arc[-1]]])
            r_fill = np.concatenate([[0.0],      r_arc, [0.0]])
            ax_pol.fill(a_fill, r_fill, color=color, alpha=0.18, zorder=1)
            # Closing spokes from arc endpoints to origin
            for a_end, r_end in [(a_arc[0], r_arc[0]), (a_arc[-1], r_arc[-1])]:
                ax_pol.plot([a_end, a_end], [0.0, r_end],
                            color=color, lw=0.8, ls='--', alpha=0.55)
        ax_pol.plot(a_arc, r_arc, color=color, lw=1.3, alpha=0.9)

    r_max_all = float(np.nanmax(hull_radii))
    ax_pol.annotate('', xy=(0, r_max_all * 0.65), xytext=(0, r_max_all * 0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=16))
    ax_pol.text(0, r_max_all * 0.97, 'upwind', ha='center', va='bottom', fontsize=8)
    ax_pol.text(np.pi, r_max_all * 0.97, 'downwind', ha='center', va='top',
                fontsize=8, color='dimgray')
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_thetagrids([0, 90, 180, 270],
                          labels=['upwind\n(0°)', 'cross\n(90°)', 'down\n(180°)', 'cross\n(270°)'],
                          fontsize=7)
    ax_pol.set_title("Hull boundaries — wind frame\n(curves expand with $V_{ref}$)", pad=14)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig1.colorbar(sm, ax=ax_pol, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

    # Right: raw scatter + hull polygon for a mid-range V_ref
    vr_demo = v_ref_levels[len(v_ref_levels) // 2]
    mask    = pts_vref == vr_demo
    uc, vc  = pts_u[mask], pts_v[mask]
    ax_xy.scatter(uc, vc, s=18, alpha=0.6, color=cmap(norm(vr_demo)),
                  zorder=2, label=f'symmetric pts')
    pts_demo = np.unique(np.column_stack([uc, vc]), axis=0)
    try:
        h = _ConvexHull(pts_demo)
        hv = pts_demo[h.vertices]
        hvc = np.vstack([hv, hv[0]])
        ax_xy.plot(hvc[:, 0], hvc[:, 1], 'k-', lw=2, zorder=3, label='Convex hull')
    except Exception:
        pass
    ax_xy.axhline(0, color='gray', lw=0.5, ls='--')
    ax_xy.axvline(0, color='gray', lw=0.5, ls='--')
    ax_xy.set_xlabel('u  (crosswind, m/s)')
    ax_xy.set_ylabel('v  (upwind, m/s)')
    ax_xy.set_aspect('equal')
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_title(f'Point cloud + hull polygon  ($V_{{ref}}$ = {vr_demo:.0f} m/s)\n'
                    'Should be symmetric around the v-axis')

    fig1.suptitle('Figure 1 — Stage 1 verification: hull table (wind frame)',
                  fontsize=12, fontweight='bold')
    fig1.tight_layout()
    path1 = FIGS_DIR / "verify_hull_table.png"
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\n[verify] Saved → {path1}")
    plt.show()

    overlay_path = out_overlay or (HERE / "data" / "wind_hull_overlay_eq_atlantic_2023_07.npz")
    if hull_only or not overlay_path.exists():
        if not hull_only and not overlay_path.exists():
            print(f"[verify] Wind overlay not found at {overlay_path} — skipping Figures 2 & 3.")
        return

    # Load overlay
    ov             = np.load(overlay_path)
    lat            = ov['lat']
    lon            = ov['lon']
    v_ref_local    = ov['v_ref_local']
    wind_dir       = ov['wind_dir']
    upwind_bearing = ov['upwind_bearing']
    max_speed      = ov['max_speed_achievable']
    hull_geo       = ov['hull_radii_geo']
    geo_bearings   = ov['geo_bearings']

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2 — Wind speed and max achievable speed (two panels)
    #
    # The tacking data covers V_ref = 9–25 m/s.  Equatorial Atlantic 10m
    # winds in July are typically 4–9 m/s, so much of the domain is clamped
    # to V_ref_min = 9 m/s → the hull is the same everywhere there → max
    # achievable speed is uniform.  Panel (a) shows where this clamping
    # occurs; panel (b) shows max_speed with the clamped region hatched.
    # ═══════════════════════════════════════════════════════════════════════
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    # Derive extent from the actual data, with a small margin
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_pad = max((lon_max - lon_min) * 0.02, 1.0)
    lat_pad = max((lat_max - lat_min) * 0.02, 1.0)
    extent  = [lon_min - lon_pad, lon_max + lon_pad,
               lat_min - lat_pad, lat_max + lat_pad]

    # Adaptive quiver stride: aim for ~30 arrows per axis
    stride_lon = max(1, len(lon) // 30)
    stride_lat = max(1, len(lat) // 30)
    # Adaptive quiver scale: proportional to median wind speed so arrows are readable
    med_speed  = float(np.nanmedian(v_ref_local))
    q_scale    = med_speed * 25   # tuned so a typical arrow is ~4% of axis width

    proj    = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig2, (ax2a, ax2b) = plt.subplots(
        1, 2, figsize=(17, 6), subplot_kw=proj,
    )
    if HAS_CARTOPY:
        for ax in (ax2a, ax2b):
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    LON, LAT = np.meshgrid(lon, lat)
    vr_min   = float(v_ref_levels.min())   # lower edge of tacking data (9 m/s)
    clamped  = v_ref_local < vr_min        # True where wind is too weak for table

    trans   = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    u10_rec = v_ref_local * np.sin(wind_dir)
    v10_rec = v_ref_local * np.cos(wind_dir)

    # ── Panel (a): ERA5 wind speed at 10 m ──────────────────────────────
    cf_a = ax2a.pcolormesh(LON, LAT, v_ref_local, cmap='YlOrRd', **trans)
    plt.colorbar(cf_a, ax=ax2a, label='ERA5 wind speed at 10 m  (= V_ref local, m/s)',
                 shrink=0.8)
    # Mark the V_ref_min threshold as a contour
    cs = ax2a.contour(LON, LAT, v_ref_local, levels=[vr_min],
                      colors='blue', linewidths=1.5, **trans)
    ax2a.clabel(cs, fmt=f'V_ref_min={vr_min:.0f} m/s', fontsize=8)
    ax2a.quiver(
        LON[::stride_lat, ::stride_lon], LAT[::stride_lat, ::stride_lon],
        u10_rec[::stride_lat, ::stride_lon], v10_rec[::stride_lat, ::stride_lon],
        color='black', scale=q_scale, width=0.003, alpha=0.6, **trans,
    )
    if HAS_CARTOPY:
        ax2a.add_feature(cfeature.LAND, facecolor='#c8c8c8', zorder=5)
        ax2a.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
        ax2a.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.4)
    ax2a.set_title(f'(a) ERA5 wind speed at 10 m\n'
                   f'Blue contour = V_ref_min ({vr_min:.0f} m/s) — clamped below this',
                   fontsize=10)

    # ── Panel (b): max achievable speed with clamped region hatched ─────
    cf_b = ax2b.pcolormesh(
        LON, LAT, max_speed, cmap='viridis',
        vmin=np.nanmin(max_speed), vmax=np.nanmax(max_speed), **trans,
    )
    plt.colorbar(cf_b, ax=ax2b, label='Max achievable albatross speed (m/s)', shrink=0.8)
    # Hatch the clamped region so the uniform color is explained visually
    ax2b.contourf(LON, LAT, clamped.astype(float), levels=[0.5, 1.5],
                  hatches=['////'], colors='none', **trans)
    ax2b.contour(LON, LAT, v_ref_local, levels=[vr_min],
                 colors='white', linewidths=1.0, linestyles='--', **trans)
    ax2b.quiver(
        LON[::stride_lat, ::stride_lon], LAT[::stride_lat, ::stride_lon],
        u10_rec[::stride_lat, ::stride_lon], v10_rec[::stride_lat, ::stride_lon],
        color='white', scale=q_scale, width=0.003, alpha=0.7, **trans,
    )
    if HAS_CARTOPY:
        ax2b.add_feature(cfeature.LAND, facecolor='#c8c8c8', zorder=5)
        ax2b.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=6)
        ax2b.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.4)

    pct_clamped = 100 * clamped.mean()
    ax2b.set_title(f'(b) Max achievable speed — hatched = wind below V_ref_min\n'
                   f'({pct_clamped:.0f}% of domain clamped; uniform color there is expected)',
                   fontsize=10)

    fig2.suptitle('Figure 2 — Wind speed vs achievable speed\n'
                  'Uniform color in (b) is expected where wind < V_ref_min of tacking data',
                  fontsize=11, fontweight='bold')
    fig2.tight_layout()
    path2 = FIGS_DIR / "verify_max_speed_map.png"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[verify] Saved → {path2}")
    print(f"  {pct_clamped:.0f}% of domain has wind speed < V_ref_min ({vr_min:.0f} m/s) "
          f"— hull is clamped to the weakest tacking level there")
    plt.show()

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 3 — Rotation verification (the critical test)
    # Find one grid point per cardinal wind direction (N / E / S / W).
    # For each, plot the geographic hull + wind arrow.
    # PASS: max-speed lobe aligns with the red arrow.
    # ═══════════════════════════════════════════════════════════════════════
    target_dirs = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    dir_labels  = ['Wind blowing North (↑)', 'Wind blowing East (→)',
                   'Wind blowing South (↓)', 'Wind blowing West (←)']

    wind_flat = wind_dir.ravel()
    selected  = []
    for td in target_dirs:
        # Circular distance to target bearing
        diff = np.abs(((wind_flat - td + np.pi) % (2 * np.pi)) - np.pi)
        idx  = int(np.argmin(diff))
        selected.append((idx // len(lon), idx % len(lon)))

    fig3, axes = plt.subplots(2, 2, figsize=(12, 11),
                               subplot_kw={'projection': 'polar'})
    axes = axes.ravel()

    for ax, (i_lat, i_lon), label in zip(axes, selected, dir_labels):
        r   = hull_geo[i_lat, i_lon]
        wd  = float(wind_dir[i_lat, i_lon])
        ub  = float(upwind_bearing[i_lat, i_lon])
        vr  = float(v_ref_local[i_lat, i_lon])
        pt_lat = float(lat[i_lat])
        pt_lon = float(lon[i_lon])

        valid = ~np.isnan(r)
        if valid.sum() >= 2:
            ax.plot(geo_bearings[valid], r[valid], color='steelblue', lw=2)
            ax.fill(geo_bearings[valid], r[valid], color='steelblue', alpha=0.15)

        r_ref = float(np.nanmax(r)) if valid.sum() > 0 else 1.0

        # Red arrow: downwind direction (wind blows TO here — expected max speed)
        ax.annotate('',
                    xy=(wd, r_ref * 0.78), xytext=(wd, 0.05 * r_ref),
                    arrowprops=dict(arrowstyle='->', color='tomato',
                                   lw=2.5, mutation_scale=18))
        ax.text(wd, r_ref * 0.88, 'wind→', ha='center', va='bottom',
                fontsize=8, color='tomato', fontweight='bold')

        # Dashed line: upwind direction (expected lower speed side)
        ax.plot([ub, ub], [0, r_ref * 0.55], color='navy',
                lw=1.5, ls='--', alpha=0.8)
        ax.text(ub, r_ref * 0.60, '←wind', ha='center', va='bottom',
                fontsize=7.5, color='navy', alpha=0.9)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 90, 180, 270], labels=['N', 'E', 'S', 'W'], fontsize=8)
        ax.set_title(
            f'{label}\n'
            f'({pt_lat:.1f}°N, {pt_lon:.1f}°E)  '
            f'$V_{{ref}}$={vr:.1f} m/s  wind={np.degrees(wd):.0f}°',
            fontsize=9, pad=14,
        )

    fig3.suptitle(
        'Figure 3 — Rotation verification: hull in geographic frame\n'
        'PASS: max-speed lobe (blue bulge) aligns with red "wind→" arrow in all panels',
        fontsize=11, fontweight='bold',
    )
    fig3.tight_layout()
    path3 = FIGS_DIR / "verify_rotation.png"
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"[verify] Saved → {path3}")
    plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build velocity hull table and ERA5 wind overlay"
    )
    parser.add_argument(
        "--era5", type=str, default=None, metavar="PATH",
        help=(
            "ERA5 NetCDF file to build the overlay from. "
            "Default: data/era5_wind_eq_atlantic_2023_07.nc. "
            "Use --era5 data/era5_wind_southern_ocean_2023_07.nc for the Southern Ocean, etc."
        ),
    )
    parser.add_argument(
        "--hull-only", action="store_true",
        help="Only build velocity_hulls.npz, skip ERA5 overlay",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip verification plots after building",
    )
    args = parser.parse_args()

    era5_path = Path(args.era5) if args.era5 else None

    v_ref_levels, hull_radii, angles_ray = build_hull_table()

    out_overlay = None
    if not args.hull_only:
        out_overlay = build_wind_overlay(v_ref_levels, hull_radii, angles_ray,
                                         era5_path=era5_path)

    if not args.skip_verify:
        print("\nRunning verification plots...")
        verify(hull_only=args.hull_only, out_overlay=out_overlay)
