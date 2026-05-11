import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, RectBivariateSpline
from utils import _alpha, _rotation
DATA = Path(__file__).parents[1] / 'mesoscale' / 'data'

_N_ANGLES = 720

class Hull:
    """
    Wind-frame velocity hull: maps (V_ref, angle) → [u_wind, v_wind].

    angle = arctan2(u_avg, v_avg) — direction of the velocity vector in wind
    frame.  The identity u = speed·sin(angle), v = speed·cos(angle) holds
    exactly, so only speed(V_ref, angle) needs interpolating.

    A single RegularGridInterpolator in (V_ref, angle) space replaces the
    old 500×500 Cartesian griddata pipeline.  PCHIP resampling per V_ref
    level gives smooth, monotone radii with natural zeros outside the valid
    arc.
    """
    def __init__(self, method: str = 'linear'):
        d = np.load(DATA / 'tacking_mirrored.npz')
        vrefs  = np.unique(d['V_ref'])
        angles = np.linspace(0, 2 * np.pi, _N_ANGLES, endpoint=False)

        speed_grid      = np.zeros((len(vrefs), _N_ANGLES))
        theta_forbidden = np.zeros(len(vrefs))

        for i, vr in enumerate(vrefs):
            mask = d['V_ref'] == vr
            a = d['angle'][mask]
            s = d['speed'][mask]
            idx = np.argsort(a)
            a_s, s_s = a[idx], s[idx]
            # Average speeds at duplicate angles (artefact of tacking-line gap fill)
            a_u, inv = np.unique(a_s, return_inverse=True)
            s_u = np.bincount(inv, weights=s_s) / np.bincount(inv)

            gap = a_u[0] + (2 * np.pi - a_u[-1])
            if gap < 2:
                # Small gap (~10°) straddles angle=0: a sampling artefact.  Wrap
                # one point from each end so PCHIP interpolates smoothly across
                # 0/2π.  Entire arc is valid — no forbidden zone.
                theta_forbidden[i] = 0.0
                a_ext = np.concatenate([[a_u[-1] - 2*np.pi], a_u, [a_u[0] + 2*np.pi]])
                s_ext = np.concatenate([[s_u[-1]], s_u, [s_u[0]]])
                row = PchipInterpolator(a_ext, s_ext)(angles)
            else:
                # Real forbidden zone near angle=0. Record its boundary.
                theta_forbidden[i] = a_u[0]
                row = PchipInterpolator(a_u, s_u, extrapolate=False)(angles)
                # Fill the forbidden zones with the boundary speed rather than
                # zero so that V_ref-direction interpolation is not contaminated
                # by artificial zeros. velocity() gates the result to 0 for
                # invalid angles, so these fill values are never returned.
                row[angles < a_u[0]] = s_u[0]
                row[angles > a_u[-1]] = s_u[-1]

            speed_grid[i] = np.nan_to_num(row, nan=0.0).clip(0)

        self._vrefs           = vrefs
        self._angles          = angles
        self._theta_forbidden = theta_forbidden  # decreases monotonically with V_ref
        self._u               = d['u_avg']   # raw data — used by plot helpers
        self._v               = d['v_avg']

        # Extend grid to 2π so angles just below 2π (e.g. from construct_arc with n>720)
        # don't fall outside the grid and get fill_value=0.
        angles_ext     = np.append(angles, 2 * np.pi)
        speed_grid_ext = np.hstack([speed_grid, speed_grid[:, [0]]])
        self._method = method
        self._interp = RegularGridInterpolator(
            (vrefs, angles_ext), speed_grid_ext,
            method=method, bounds_error=False, fill_value=0.0,
        )

    def _in_valid_domain(self, v_ref: float, angle: float) -> bool:
        """Return True if (v_ref, angle) lies inside the valid flight arc.

        The forbidden zone is symmetric around angle=0 (upwind) with a
        half-width that is linearly interpolated between V_ref levels.
        """
        v_ref_c = float(np.clip(v_ref, self._vrefs[0], self._vrefs[-1]))
        th_f    = float(np.interp(v_ref_c, self._vrefs, self._theta_forbidden))
        if th_f <= 0.0:
            return True
        angle_n = float(angle) % (2 * np.pi)
        return th_f <= angle_n <= 2 * np.pi - th_f

    def velocity(self, v_ref: float, angle: float) -> np.ndarray:
        angle = float(angle) % (2 * np.pi)
        if not self._in_valid_domain(v_ref, angle):
            return np.array([0.0, 0.0])
        speed = max(0.0, float(self._interp([[v_ref, angle]])[0]))
        return np.array([speed * np.sin(angle), speed * np.cos(angle)])

    def construct_arc(self, v_ref: float, n: int = 360) -> tuple:
        """Return (angles, u, v) for the iso-curve at v_ref.

        Only the valid arc is sampled; the forbidden zone near angle=0 is
        excluded entirely so the returned arrays contain no zero/NaN values.
        """
        v_ref_c = float(np.clip(v_ref, self._vrefs[0], self._vrefs[-1]))
        th_f    = float(np.interp(v_ref_c, self._vrefs, self._theta_forbidden))
        angles  = (np.linspace(0, 2 * np.pi, n, endpoint=False)
                   if th_f <= 0.0
                   else np.linspace(th_f, 2 * np.pi - th_f, n))
        pts    = np.column_stack([np.full(n, v_ref), angles])
        speeds = self._interp(pts).clip(0)
        return angles, speeds * np.sin(angles), speeds * np.cos(angles)

    def gradient(self, w: np.ndarray, theta: float, dw: float = 0.01) -> np.ndarray:
        def geo_vel(w_):
            w_mag = np.linalg.norm(w_)
            alpha = _alpha(w_)
            v_alpha = self.velocity(v_ref=w_mag, angle=theta)
            v_geo = _rotation(alpha) @ v_alpha
            return v_geo[0], v_geo[1]

        vxp0, vyp0 = geo_vel(w + np.array([dw, 0.0]))
        vxm0, vym0 = geo_vel(w - np.array([dw, 0.0]))
        vxp1, vyp1 = geo_vel(w + np.array([0.0, dw]))
        vxm1, vym1 = geo_vel(w - np.array([0.0, dw]))

        return np.array([
            [(vxp0 - vxm0) / (2*dw), (vxp1 - vxm1) / (2*dw)],
            [(vyp0 - vym0) / (2*dw), (vyp1 - vym1) / (2*dw)],
        ])


class SmoothHull:
    """
    C² tensor-product cubic spline replacement for Hull.

    Uses the same PCHIP-resampled speed_grid as Hull but replaces the bilinear
    RegularGridInterpolator with RectBivariateSpline(kx=3, ky=3), giving C²
    continuity in both V_ref and angle throughout the interior of the valid
    domain.  A 3-column periodic guard band is prepended/appended to the angle
    axis so the spline wraps smoothly at 0/2π.

    Public interface is identical to Hull: velocity(), construct_arc(),
    gradient(), _in_valid_domain().
    """

    def __init__(self):
        d = np.load(DATA / 'tacking_mirrored.npz')
        vrefs  = np.unique(d['V_ref'])
        angles = np.linspace(0, 2 * np.pi, _N_ANGLES, endpoint=False)

        speed_grid      = np.zeros((len(vrefs), _N_ANGLES))
        theta_forbidden = np.zeros(len(vrefs))

        for i, vr in enumerate(vrefs):
            mask = d['V_ref'] == vr
            a = d['angle'][mask]
            s = d['speed'][mask]
            idx = np.argsort(a)
            a_s, s_s = a[idx], s[idx]
            a_u, inv = np.unique(a_s, return_inverse=True)
            s_u = np.bincount(inv, weights=s_s) / np.bincount(inv)

            gap = a_u[0] + (2 * np.pi - a_u[-1])
            if gap < 2:
                theta_forbidden[i] = 0.0
                a_ext = np.concatenate([[a_u[-1] - 2*np.pi], a_u, [a_u[0] + 2*np.pi]])
                s_ext = np.concatenate([[s_u[-1]], s_u, [s_u[0]]])
                row = PchipInterpolator(a_ext, s_ext)(angles)
            else:
                theta_forbidden[i] = a_u[0]
                row = PchipInterpolator(a_u, s_u, extrapolate=False)(angles)
                row[angles < a_u[0]] = s_u[0]
                row[angles > a_u[-1]] = s_u[-1]

            speed_grid[i] = np.nan_to_num(row, nan=0.0).clip(0)

        self._vrefs           = vrefs
        self._angles          = angles
        self._theta_forbidden = theta_forbidden
        self._u               = d['u_avg']
        self._v               = d['v_avg']

        # 3-column periodic guard band so the cubic spline wraps smoothly at 0/2π.
        # A degree-3 basis has support over 4 consecutive knots; 3 guard columns on
        # each side ensure all valid-domain queries interpolate rather than extrapolate.
        _K = 3
        angles_ext = np.concatenate([
            angles[-_K:] - 2 * np.pi,
            angles,
            angles[:_K]  + 2 * np.pi,
        ])
        speed_grid_ext = np.hstack([
            speed_grid[:, -_K:],
            speed_grid,
            speed_grid[:, :_K],
        ])
        self._interp = RectBivariateSpline(
            vrefs, angles_ext, speed_grid_ext, kx=3, ky=3, s=0,
        )

    def _in_valid_domain(self, v_ref: float, angle: float) -> bool:
        v_ref_c = float(np.clip(v_ref, self._vrefs[0], self._vrefs[-1]))
        th_f    = float(np.interp(v_ref_c, self._vrefs, self._theta_forbidden))
        if th_f <= 0.0:
            return True
        angle_n = float(angle) % (2 * np.pi)
        return th_f <= angle_n <= 2 * np.pi - th_f

    def velocity(self, v_ref: float, angle: float) -> np.ndarray:
        angle = float(angle) % (2 * np.pi)
        if not self._in_valid_domain(v_ref, angle):
            return np.array([0.0, 0.0])
        speed = max(0.0, float(self._interp.ev(v_ref, angle)))
        return np.array([speed * np.sin(angle), speed * np.cos(angle)])

    def construct_arc(self, v_ref: float, n: int = 360) -> tuple:
        """Return (angles, u, v) for the iso-curve at v_ref."""
        v_ref_c = float(np.clip(v_ref, self._vrefs[0], self._vrefs[-1]))
        th_f    = float(np.interp(v_ref_c, self._vrefs, self._theta_forbidden))
        angles  = (np.linspace(0, 2 * np.pi, n, endpoint=False)
                   if th_f <= 0.0
                   else np.linspace(th_f, 2 * np.pi - th_f, n))
        speeds  = self._interp.ev(np.full(n, v_ref), angles).clip(0)
        return angles, speeds * np.sin(angles), speeds * np.cos(angles)

    def gradient(self, w: np.ndarray, theta: float, dw: float = 0.01) -> np.ndarray:
        def geo_vel(w_):
            w_mag = np.linalg.norm(w_)
            alpha = _alpha(w_)
            v_alpha = self.velocity(v_ref=w_mag, angle=theta)
            v_geo = _rotation(alpha) @ v_alpha
            return v_geo[0], v_geo[1]

        vxp0, vyp0 = geo_vel(w + np.array([dw, 0.0]))
        vxm0, vym0 = geo_vel(w - np.array([dw, 0.0]))
        vxp1, vyp1 = geo_vel(w + np.array([0.0, dw]))
        vxm1, vym1 = geo_vel(w - np.array([0.0, dw]))

        return np.array([
            [(vxp0 - vxm0) / (2*dw), (vxp1 - vxm1) / (2*dw)],
            [(vyp0 - vym0) / (2*dw), (vyp1 - vym1) / (2*dw)],
        ])


def _tacking_line(u_tip: float, v_const: float, n: int = 20):
    # Symmetric half-linspace ensures u=0 is always included, giving the true
    # directly-upwind speed (v_const) as an explicit data point rather than
    # having it inferred by PCHIP across a gap.
    half = np.linspace(0.0, abs(u_tip), n // 2 + 1)
    u = np.concatenate([-half[::-1][:-1], half])
    v = np.full(len(u), v_const)
    return u, v, np.sqrt(u**2 + v**2), np.arctan2(u, v) % (2 * np.pi)


def make_mirrored(src: Path = DATA / 'tacking_diagram.npz',
                  dst: Path = DATA / 'tacking_mirrored.npz') -> None:
    d = np.load(src)

    V_ref = d['V_ref']
    u_avg = d['u_avg']
    v_avg = d['v_avg']
    speed = d['speed']
    angle = d['angle']

    u_mirror     = -u_avg
    angle_mirror = np.arctan2(u_mirror, v_avg) % (2 * np.pi)

    fill_V, fill_u, fill_v, fill_s, fill_a = [], [], [], [], []

    for vr in np.unique(V_ref):
        mask = V_ref == vr
        vc, uc = v_avg[mask], u_avg[mask]

        bot = vc.argmin()
        u, v, s, a = _tacking_line(uc[bot], vc[bot])
        fill_V.append(np.full(len(u), vr))
        fill_u.append(u); fill_v.append(v); fill_s.append(s); fill_a.append(a)

        top = vc.argmax()
        if vc[top] > 0:
            u, v, s, a = _tacking_line(uc[top], vc[top])
            fill_V.append(np.full(len(u), vr))
            fill_u.append(u); fill_v.append(v); fill_s.append(s); fill_a.append(a)

    np.savez(
        dst,
        V_ref = np.concatenate([V_ref, V_ref,       *fill_V]),
        u_avg = np.concatenate([u_avg, u_mirror,     *fill_u]),
        v_avg = np.concatenate([v_avg, v_avg,        *fill_v]),
        speed = np.concatenate([speed, speed,        *fill_s]),
        angle = np.concatenate([angle, angle_mirror, *fill_a]),
    )
    n_fill = sum(len(x) for x in fill_u)
    print(f'Saved {len(V_ref)*2} arc + {n_fill} gap-fill rows = {len(V_ref)*2 + n_fill} total → {dst}')


def plot_mirrored(path: Path = DATA / 'tacking_mirrored.npz',
                  output: Path = Path(__file__).parents[1] / 'figures' / 'mesoscale' / 'tacking_mirrored.png') -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    d     = np.load(path)
    V_ref = d['V_ref']
    angle = d['angle']
    speed = d['speed']

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    vrefs = np.unique(V_ref)
    cmap  = plt.cm.plasma
    norm  = Normalize(vmin=vrefs.min(), vmax=vrefs.max())

    for vr in vrefs:
        mask  = V_ref == vr
        a     = angle[mask]
        s     = speed[mask]
        idx   = np.argsort(a)
        a_s, s_s = a[idx], s[idx]
        if d['v_avg'][mask].max() > 0:
            a_s = np.append(a_s, a_s[0] + 2 * np.pi)
            s_s = np.append(s_s, s_s[0])
        ax.plot(a_s, s_s, color=cmap(norm(vr)), lw=1.5, alpha=0.9)

    r_max = float(speed.max())
    ax.annotate('', xy=(0, r_max * 0.65), xytext=(0, r_max * 0.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
    ax.text(0, r_max * 0.97, 'Wind', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
    ax.set_title('Achievable velocity iso-curves by $V_{ref}$', pad=14)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.show()


def test_construct_arc(
    output: Path = Path(__file__).parents[1] / 'figures' / 'mesoscale' / 'tacking_construct_arcs.png',
    step: float = 0.3,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    h     = Hull()
    vrefs = np.arange(h._vrefs.min(), h._vrefs.max() + step, step)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    cmap = plt.cm.plasma
    norm = Normalize(vmin=h._vrefs.min(), vmax=h._vrefs.max())

    for vr in vrefs:
        angles, us, vs = h.construct_arc(vr)
        speed = np.sqrt(us**2 + vs**2)
        lw    = 1.8 if vr == round(vr) else 0.6
        ax.plot(angles, speed, color=cmap(norm(vr)), lw=lw, alpha=0.85)

    r_max = float(np.sqrt(h._u**2 + h._v**2).max())
    ax.annotate('', xy=(0, r_max * 0.65), xytext=(0, r_max * 0.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
    ax.text(0, r_max * 0.97, 'Wind', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
    ax.set_title(f'Interpolated iso-curves (step={step} m/s)', pad=14)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.show()


def test_velocity_vs_vref(
    output: Path = Path(__file__).parents[1] / 'figures' / 'mesoscale' / 'velocity_vs_vref.png',
) -> None:
    """V vs V_ref at theta=0 (upwind) and theta=180 (downwind)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    h = Hull()
    cases = [(0.0, 'upwind'), (180.0, 'downwind')]
    vrefs_fine = np.linspace(h._vrefs.min(), h._vrefs.max(), 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (theta_deg, label) in zip(axes, cases):
        angle = np.radians(theta_deg)
        speeds = np.array([np.linalg.norm(h.velocity(vr, angle)) for vr in vrefs_fine])
        ax.plot(vrefs_fine, speeds, '-', lw=1.5, label='interpolated')

        dots = np.array([np.linalg.norm(h.velocity(vr, angle)) for vr in h._vrefs])
        ax.plot(h._vrefs, dots, 'o', ms=6, label='data levels')

        ax.set_xlabel('$V_{ref}$ [m/s]')
        ax.set_ylabel('$V$ [m/s]')
        ax.set_title(f'$\\theta = {theta_deg:.0f}°$ ({label})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Achievable speed vs wind speed at fixed heading')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.show()

    # All headings coloured by theta
    thetas_deg = np.arange(0, 360, 10)
    cmap = plt.cm.hsv
    norm = Normalize(vmin=0, vmax=360)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for theta_deg in thetas_deg:
        angle  = np.radians(theta_deg)
        speeds = np.array([np.linalg.norm(h.velocity(vr, angle)) for vr in vrefs_fine])
        ax2.plot(vrefs_fine, speeds, '-', color=cmap(norm(theta_deg)), lw=1.2, alpha=0.85)

    ax2.set_xlabel('$V_{ref}$ [m/s]')
    ax2.set_ylabel('$V$ [m/s]')
    ax2.set_title('Achievable speed vs wind speed — all headings')
    ax2.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, label='$\\theta$ [°]')
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(['0° (upwind)', '90°', '180° (downwind)', '270°', '360°'])

    fig2.tight_layout()
    plt.show()


def test_spline_field(
    output: Path = Path(__file__).parents[1] / 'figures' / 'mesoscale' / 'spline_field.png',
    step: float = 0.1,
) -> None:
    """Polar visualisation of the interpolated iso-curves at fine V_ref steps."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    h = Hull()
    vrefs_fine = np.arange(h._vrefs.min(), h._vrefs.max() + step, step)
    cmap = plt.cm.plasma
    norm = Normalize(vmin=h._vrefs.min(), vmax=h._vrefs.max())

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    for vr in vrefs_fine:
        angles, us, vs = h.construct_arc(vr)
        speed = np.sqrt(us**2 + vs**2)
        lw = 1.8 if abs(vr - round(vr)) < 1e-9 else 0.6
        ax.plot(angles, speed, color=cmap(norm(vr)), lw=lw, alpha=0.85)

    r_max = float(np.sqrt(h._u**2 + h._v**2).max())
    ax.annotate('', xy=(0, r_max * 0.65), xytext=(0, r_max * 0.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
    ax.text(0, r_max * 0.97, 'Wind', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
    ax.set_title(f'Interpolated iso-curves (step={step} m/s)', pad=14)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    make_mirrored()
    plot_mirrored()
    #test_construct_arc()
    #test_velocity_vs_vref()
    test_spline_field()