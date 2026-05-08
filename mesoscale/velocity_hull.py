"""
velocity_hull.py — convex hull of achievable average velocities: microscale → mesoscale bridge.

VelocityHull is built from a microscale Ensemble (or loaded from a precomputed NPZ)
and represents the reachable velocity set at each wind speed level.

Coordinate conventions
----------------------
Wind frame:
  u  — crosswind (m/s, +ve = right when facing upwind)
  v  — upwind    (m/s, +ve = into the wind)

  θ = 0   → headwind  (u=0, v>0)
  θ = π/2 → crosswind (u>0, v=0)
  θ = π   → tailwind  (u=0, v<0)

Geographic frame (ERA5):
  East  = u10 axis
  North = v10 axis

α = arctan2(−v10, −u10)   direction FROM which wind blows, CCW from East

Rotation wind-frame → geographic:
  vx_geo = v_wind · cos(α) − u_wind · sin(α)
  vy_geo = v_wind · sin(α) + u_wind · cos(α)
"""

from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

N_RAYS = 720   # angular resolution for hull boundary tracing


# ─── Ray-tracing helpers ──────────────────────────────────────────────────────

def _make_ray_dirs(n_rays: int):
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dx = np.sin(angles)   # crosswind component
    dy = np.cos(angles)   # upwind component
    return angles, dx, dy


def _trace_hull_boundary(pts: np.ndarray, dx_rays: np.ndarray, dy_rays: np.ndarray) -> np.ndarray:
    """
    Shoot rays from origin through the convex hull of pts.

    Returns
    -------
    radii : (n_rays,)  intersection distances, NaN where no intersection.
    """
    n_rays = len(dx_rays)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.full(n_rays, np.nan)

    hv  = pts[hull.vertices]
    hvc = np.vstack([hv, hv[0]])
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


# ─── VelocityHull ─────────────────────────────────────────────────────────────

class VelocityHull:
    """
    Convex hull of achievable average velocities per V_ref level.

    Attributes
    ----------
    v_ref_levels : (n_v,)         V_ref grid [m/s]
    hull_radii   : (n_v, n_rays)  boundary radius per ray, wind frame [m/s]
    hull_angles  : (n_rays,)      ray angles [rad]
    pts_u, pts_v : raw symmetric point cloud
    pts_vref     : V_ref label per point
    """

    def __init__(
        self,
        v_ref_levels: np.ndarray,
        hull_radii: np.ndarray,
        hull_angles: np.ndarray,
        pts_u: np.ndarray,
        pts_v: np.ndarray,
        pts_vref: np.ndarray,
    ):
        self.v_ref_levels = v_ref_levels
        self.hull_radii   = hull_radii
        self.hull_angles  = hull_angles
        self.pts_u        = pts_u
        self.pts_v        = pts_v
        self.pts_vref     = pts_vref

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ensemble(cls, ensemble) -> "VelocityHull":
        """
        Build hull from a microscale Ensemble object.

        For each V_ref level found in the ensemble, collects (u_avg, v_avg)
        and its mirror (-u_avg, v_avg), computes the convex hull, and traces
        720 rays to get the boundary radii.
        """
        v_ref_levels = np.array(sorted(set(c.V_ref for c in ensemble.containers)))
        angles_ray, dx_rays, dy_rays = _make_ray_dirs(N_RAYS)
        hull_radii_list = []
        pts_u_list, pts_v_list, pts_vref_list = [], [], []

        for vr in v_ref_levels:
            cs = [c for c in ensemble.containers if c.V_ref == vr]
            uc = np.array([np.mean(c.u) for c in cs])
            vc = np.array([np.mean(c.v) for c in cs])

            u_all = np.concatenate([ uc, -uc])
            v_all = np.concatenate([ vc,  vc])
            u_all = np.append(u_all, [0.0, 0.0])
            v_all = np.append(v_all, [float(vc.max()), float(vc.min())])
            pts   = np.unique(np.column_stack([u_all, v_all]), axis=0)

            radii = _trace_hull_boundary(pts, dx_rays, dy_rays)
            hull_radii_list.append(radii)
            pts_u_list.append(u_all)
            pts_v_list.append(v_all)
            pts_vref_list.append(np.full(len(u_all), vr))

            n_valid = int(np.sum(~np.isnan(radii)))
            max_r   = float(np.nanmax(radii)) if n_valid > 0 else float('nan')
            print(f"  V_ref = {vr:5.1f} m/s  hull pts = {len(pts):3d}  "
                  f"max speed = {max_r:6.2f} m/s  rays hit = {n_valid}/{N_RAYS}")

        from scipy.ndimage import gaussian_filter1d
        hull_radii = gaussian_filter1d(
            np.nan_to_num(np.array(hull_radii_list), nan=0.0),
            sigma=1.0, axis=0,
        )

        return cls(
            v_ref_levels = v_ref_levels,
            hull_radii   = hull_radii,
            hull_angles  = angles_ray,
            pts_u        = np.concatenate(pts_u_list),
            pts_v        = np.concatenate(pts_v_list),
            pts_vref     = np.concatenate(pts_vref_list),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "VelocityHull":
        """Load a precomputed hull from velocity_hulls.npz."""
        d = np.load(path)
        return cls(
            v_ref_levels = d['v_ref_levels'],
            hull_radii   = d['hull_radii'],
            hull_angles  = d['hull_angles'],
            pts_u        = d['pts_u'],
            pts_v        = d['pts_v'],
            pts_vref     = d['pts_vref'],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to velocity_hulls.npz format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            v_ref_levels = self.v_ref_levels,
            hull_radii   = self.hull_radii,
            hull_angles  = self.hull_angles,
            pts_u        = self.pts_u,
            pts_v        = self.pts_v,
            pts_vref     = self.pts_vref,
        )
        print(f"Saved VelocityHull → {path}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(self, V_ref: float, bearing: float) -> float:
        """
        Interpolate hull radius at arbitrary V_ref and wind-frame bearing.

        Parameters
        ----------
        V_ref   : wind speed [m/s]
        bearing : wind-frame angle [rad]  (0 = upwind, π/2 = crosswind)
        """
        radii = self._interpolate_radii(np.array([V_ref]))[0]
        return float(np.interp(bearing % (2 * np.pi), self.hull_angles, radii))

    def _interpolate_radii(self, v_ref_arr: np.ndarray) -> np.ndarray:
        """Vectorised hull-radii interpolation, returns (N, n_rays)."""
        levels = self.v_ref_levels
        radii  = self.hull_radii

        v      = np.clip(v_ref_arr, levels.min(), levels.max())
        idx_f  = np.interp(v, levels, np.arange(len(levels)))
        lo     = np.floor(idx_f).astype(int)
        hi     = np.minimum(lo + 1, len(levels) - 1)
        frac   = idx_f - lo

        r_lo = np.nan_to_num(radii[lo], nan=0.0)
        r_hi = np.nan_to_num(radii[hi], nan=0.0)
        return (1.0 - frac[:, None]) * r_lo + frac[:, None] * r_hi

    def as_dict(self) -> dict:
        """Return hull data as a plain dict."""
        return {
            "v_ref_levels": self.v_ref_levels,
            "hull_radii":   self.hull_radii,
            "hull_angles":  self.hull_angles,
        }

    def to_geographic(self, u10: np.ndarray, v10: np.ndarray) -> dict:
        """
        Rotate hull from wind frame to geographic frame for a grid of ERA5 points.

        Parameters
        ----------
        u10, v10 : (n_lat, n_lon)  ERA5 10m wind components [m/s]
        """
        n_lat, n_lon = u10.shape
        n_pts        = n_lat * n_lon

        v_ref_local    = np.sqrt(u10**2 + v10**2)
        wind_dir       = np.arctan2(u10, v10) % (2 * np.pi)
        upwind_bearing = (wind_dir + np.pi) % (2 * np.pi)

        v_ref_clamped = np.clip(
            v_ref_local.ravel(),
            self.v_ref_levels.min(),
            self.v_ref_levels.max(),
        )
        hull_interp = self._interpolate_radii(v_ref_clamped)

        geo_bearings = self.hull_angles.copy()
        upwind_flat  = upwind_bearing.ravel()

        angles_ext = np.concatenate([self.hull_angles - 2*np.pi,
                                     self.hull_angles,
                                     self.hull_angles + 2*np.pi])
        hull_ext   = np.concatenate([hull_interp, hull_interp, hull_interp], axis=1)

        hull_geo = np.full((n_pts, N_RAYS), np.nan)
        BATCH    = 2000
        for start in range(0, n_pts, BATCH):
            end = min(start + BATCH, n_pts)
            ub  = upwind_flat[start:end]
            theta_wind = (geo_bearings[None, :] - ub[:, None]) % (2 * np.pi)
            for b in range(end - start):
                hull_geo[start + b] = np.interp(
                    theta_wind[b], angles_ext, hull_ext[start + b]
                )

        return {
            "v_ref_local":          v_ref_local,
            "wind_dir":             wind_dir,
            "upwind_bearing":       upwind_bearing,
            "max_speed_achievable": np.nanmax(hull_geo, axis=1).reshape(n_lat, n_lon),
            "hull_radii_geo":       hull_geo.reshape(n_lat, n_lon, N_RAYS),
            "geo_bearings":         geo_bearings,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_field(self, n_vref: int = 200, r_max_extra: float = 4.0,
                   save: bool = False, output: str | Path | None = None):
        """Polar field plot of the continuous hull (R = V_ref, θ = bearing, colour = speed)."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        v_min  = float(self.v_ref_levels.min())
        v_max  = float(self.v_ref_levels.max())
        r_plot = v_max + r_max_extra

        v_ref_grid = np.linspace(v_min, v_max, n_vref)
        radii      = self._interpolate_radii(v_ref_grid)
        angles     = self.hull_angles

        r_edges = np.linspace(v_min, v_max, n_vref + 1)
        a_edges = np.linspace(0, 2 * np.pi, N_RAYS + 1)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        cmap = plt.cm.plasma.copy()
        cmap.set_under('white')
        norm = Normalize(vmin=0.5, vmax=radii.max())
        ax.pcolormesh(a_edges, r_edges, radii, cmap=cmap, norm=norm, shading='auto')

        theta_fill = np.linspace(0, 2 * np.pi, 360)
        ax.fill_between(theta_fill, 0, v_min, color='white', zorder=3)
        ax.fill_between(theta_fill, v_max, r_plot, color='white', alpha=1.0, zorder=3)
        ax.set_rmax(r_plot)

        ax.text(0, v_min * 0.5, 'no DS', ha='center', va='center',
                fontsize=9, color='gray', zorder=4)
        ax.text(0, (v_max + r_plot) * 0.5, 'extrap.', ha='center', va='center',
                fontsize=9, color='lightgray', zorder=4)

        r_ticks = np.arange(np.ceil(v_min), v_max + 1, 4).astype(int)
        ax.set_rticks(r_ticks)
        ax.set_rlabel_position(30)
        ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)

        ax.annotate('', xy=(0, r_plot * 0.88), xytext=(0, r_plot * 0.98),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2,
                                   mutation_scale=18), zorder=5)
        ax.text(0, r_plot * 1.02, 'Wind', ha='center', va='bottom',
                fontsize=11, fontweight='bold', zorder=5)

        fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                     label='achievable speed [m/s]', pad=0.12, shrink=0.75)
        ax.set_title("Continuous velocity hull field\n"
                     r"($R$ = $V_{ref}$ [m/s], $\theta$ = bearing)", pad=20)

        if output or save:
            out = Path(output) if output else Path("figures/velocity_hull_field.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved → {out}")

        return fig, ax

    def plot(self, ax=None, save: bool = False, output: str | Path | None = None):
        """Polar plot of hull boundaries coloured by V_ref."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))
        else:
            fig = ax.get_figure()

        cmap = plt.cm.plasma
        norm = Normalize(vmin=self.v_ref_levels.min(), vmax=self.v_ref_levels.max())

        for idx, vr in enumerate(self.v_ref_levels):
            color   = cmap(norm(vr))
            r       = self.hull_radii[idx]
            valid   = ~np.isnan(r)
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue
            a_arc = self.hull_angles[valid]
            r_arc = r[valid]
            if n_valid < len(self.hull_angles):
                a_fill = np.concatenate([[a_arc[0]], a_arc, [a_arc[-1]]])
                r_fill = np.concatenate([[0.0], r_arc, [0.0]])
                ax.fill(a_fill, r_fill, color=color, alpha=0.18, zorder=1)
                for a_end, r_end in [(a_arc[0], r_arc[0]), (a_arc[-1], r_arc[-1])]:
                    ax.plot([a_end, a_end], [0.0, r_end], color=color,
                            lw=0.8, ls='--', alpha=0.55)
            ax.plot(a_arc, r_arc, color=color, lw=1.5, alpha=0.9)

        r_max_all = float(np.nanmax(self.hull_radii))
        ax.annotate('', xy=(0, r_max_all * 0.65), xytext=(0, r_max_all * 0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
        ax.text(0, r_max_all * 0.97, 'Wind', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
        ax.set_title("Convex velocity hull [m/s]", pad=14)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

        if output or save:
            out = Path(output) if output else Path("figures/velocity_hull.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved → {out}")

        return fig, ax
