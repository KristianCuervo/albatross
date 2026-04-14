"""
albatross.microscale.ensemble — collection of Container results.

Provides grouping and plotting utilities for polar sweeps.
"""

from .container import Container
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

LINESTYLES = ['-', '--', ':', '-.']

_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"


class Ensemble:
    """
    Holds all Containers across all (N, V_ref, theta, mode) combinations.
    """

    def __init__(self):
        self.containers: list[Container] = []

    def add_container(self, c: Container) -> None:
        self.containers.append(c)

    def __iter__(self):
        return iter(self.containers)

    def __len__(self):
        return len(self.containers)

    def __getitem__(self, i):
        return self.containers[i]

    def sorted_by_theta(self) -> list[Container]:
        return sorted(self.containers, key=lambda c: c.theta)

    def best(self) -> Container:
        """Container with the highest obj value (max VMG)."""
        return max(self.containers, key=lambda c: c.obj)

    def best_at(self, theta: float) -> Container:
        return min(self.containers, key=lambda c: abs(c.theta - theta))

    def save(self, path: str | Path) -> None:
        """
        Save key arrays to a compressed NPZ file.

        Stored keys: u_avg, v_avg, V_ref, theta, obj, T_cycle, N, scheme, mode.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            u_avg   = np.array([np.mean(c.u)   for c in self.containers]),
            v_avg   = np.array([np.mean(c.v)   for c in self.containers]),
            V_ref   = np.array([c.V_ref         for c in self.containers]),
            theta   = np.array([c.theta         for c in self.containers]),
            obj     = np.array([c.obj           for c in self.containers]),
            T_cycle = np.array([c.T_cycle       for c in self.containers]),
            N       = np.array([c.N             for c in self.containers]),
        )
        print(f"Saved Ensemble ({len(self.containers)} containers) → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Ensemble":
        """
        Load an Ensemble from an NPZ file.

        Accepts both the format produced by Ensemble.save() and the legacy
        tacking_diagram.npz format from the original refactor/ module (which
        uses key names 'N_v', 'speed', 'angle' instead of 'N', 'obj').

        Note: only per-cycle aggregate quantities are stored (no raw trajectories).
        """
        d   = np.load(path)
        ens = cls()
        n   = len(d['theta'])

        # Determine key mapping for both formats
        N_arr  = d['N']     if 'N'   in d.files else d['N_v']
        obj_arr = d['obj']  if 'obj' in d.files else d['speed']

        for i in range(n):
            N  = int(N_arr[i])
            tc = float(d['T_cycle'][i])
            dt = tc / N if N > 0 else 1.0
            c  = Container(
                u       = np.array([float(d['u_avg'][i])]),
                v       = np.array([float(d['v_avg'][i])]),
                w       = np.zeros(1),
                x       = np.zeros(1),
                y       = np.zeros(1),
                h       = np.zeros(1),
                cl      = np.zeros(1),
                mu      = np.zeros(1),
                obj     = float(obj_arr[i]),
                dt      = dt,
                T_cycle = tc,
                theta   = float(d['theta'][i]),
                V_ref   = float(d['V_ref'][i]),
                N       = N,
            )
            ens.add_container(c)
        return ens

    def to_hull(self):
        """Build a VelocityHull directly from this Ensemble."""
        from ..macroscale.hull import VelocityHull
        return VelocityHull.from_ensemble(self)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unique_V_refs(self) -> list[float]:
        return sorted(set(c.V_ref for c in self.containers))

    def _unique_Ns(self) -> list[int]:
        return sorted(set(c.N for c in self.containers))

    def _group(self) -> dict[tuple, list[Container]]:
        """Returns {(N, V_ref): [containers sorted by theta]}"""
        groups: dict[tuple, list[Container]] = {}
        for c in self.containers:
            key = (c.N, c.V_ref)
            groups.setdefault(key, []).append(c)
        for key in groups:
            groups[key].sort(key=lambda c: c.theta)
        return groups

    def _color_style_maps(self):
        V_refs = self._unique_V_refs()
        Ns     = self._unique_Ns()
        cmap   = plt.colormaps['viridis']
        colors = {v: cmap(i / max(len(V_refs) - 1, 1)) for i, v in enumerate(V_refs)}
        styles = {n: LINESTYLES[i % len(LINESTYLES)]   for i, n in enumerate(Ns)}
        return colors, styles

    def _save(self, fig, filename: str) -> None:
        _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_FIGURES_DIR / filename, dpi=150, bbox_inches='tight')

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_tacking_diagram(self, ax=None, save: bool = False, output=None):
        """
        Tacking diagram: achievable ground-velocity set in the wind frame.

        For each V_ref level, plots the actual mean travel velocity (u_avg, v_avg)
        and its left/right mirror (-u_avg, v_avg) as a connected arc on polar axes.

        Coordinate convention (matches VelocityHull.plot):
          θ = 0   → upwind  (top of polar)
          θ = π/2 → crosswind (right)
          θ = π   → downwind (bottom)

        The curves are open arcs — they do NOT close at θ = 0 (upwind), because
        the optimal tacking angle is always >45° away from dead upwind.
        Style (colormap, axes, labels) matches VelocityHull.plot() exactly.
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
        else:
            fig = ax.get_figure()

        V_refs = self._unique_V_refs()
        cmap   = plt.cm.plasma
        norm   = Normalize(vmin=min(V_refs), vmax=max(V_refs))

        all_radii = []

        for v_ref in V_refs:
            containers = sorted(
                [c for c in self.containers if c.V_ref == v_ref],
                key=lambda c: c.theta,
            )
            if not containers:
                continue

            u_arr = np.array([np.mean(c.u) for c in containers])
            v_arr = np.array([np.mean(c.v) for c in containers])

            # Two separate symmetric arcs — right half and left-mirror half.
            # Plotting separately avoids a false connecting segment at θ≈180°
            # (downwind), since the bird never travels exactly straight downwind.
            angles_r = np.arctan2( u_arr,          v_arr         ) % (2 * np.pi)
            angles_l = np.arctan2(-u_arr[::-1],    v_arr[::-1]   ) % (2 * np.pi)
            radii_r  = np.sqrt(u_arr**2        + v_arr**2)
            radii_l  = radii_r[::-1]
            all_radii.extend(radii_r.tolist())

            color = cmap(norm(v_ref))
            ax.plot(angles_r, radii_r, color=color, lw=1.5, alpha=0.9)
            ax.plot(angles_l, radii_l, color=color, lw=1.5, alpha=0.9)

        r_max = max(all_radii) if all_radii else 1.0

        # Wind arrow at θ=0 (top), pointing downward into the plot
        ax.annotate('', xy=(0, r_max * 0.65), xytext=(0, r_max * 0.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
        ax.text(0, r_max * 0.97, 'Wind', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
        ax.set_title(
            "Tacking diagram — wind frame\n"
            "(achievable ground velocity, coloured by $V_{ref}$)",
            pad=14,
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

        if output or save:
            out = Path(output) if output else _FIGURES_DIR / "tacking_diagram.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_ground_speed(self, ax=None, save: bool = False):
        """Polar plot of optimised ground speed (obj) vs heading theta."""
        colors, styles = self._color_style_maps()

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        else:
            fig = ax.get_figure()

        for (n, v), containers in self._group().items():
            thetas = np.array([c.theta for c in containers])
            speeds = np.array([c.obj   for c in containers])
            thetas = np.append(thetas, thetas[0])
            speeds = np.append(speeds, speeds[0])
            ax.plot(thetas, speeds,
                    color=colors[v], linestyle=styles[n],
                    label=f'N={n}, V_ref={v}')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("Ground speed [m/s]", pad=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        if save:
            self._save(fig, "ground_speed.png")
        return fig, ax

    def plot_cycle_times(self, ax=None, save: bool = False, output=None):
        """Polar plot of T_cycle [s] vs heading theta, styled like plot_obj_polar."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
        else:
            fig = ax.get_figure()

        V_refs = self._unique_V_refs()
        cmap   = plt.cm.plasma
        norm   = Normalize(vmin=min(V_refs), vmax=max(V_refs))

        all_tcycles = []

        for v_ref in V_refs:
            containers = sorted(
                [c for c in self.containers if c.V_ref == v_ref],
                key=lambda c: c.theta,
            )
            if not containers:
                continue

            thetas  = np.array([c.theta   for c in containers])
            tcycles = np.array([c.T_cycle for c in containers])

            # Right arc: θ ∈ [0, π] only
            mask_r  = thetas <= np.pi + 1e-9
            th_r    = thetas[mask_r]
            tc_r    = tcycles[mask_r]

            # Left (mirror) arc: 2π − θ, reversed to run from 2π → π
            th_l = (2 * np.pi - th_r)[::-1]
            tc_l = tc_r[::-1]

            color = cmap(norm(v_ref))
            ax.plot(th_r, tc_r, color=color, lw=1.5, alpha=0.9)
            ax.plot(th_l, tc_l, color=color, lw=1.5, alpha=0.9)

            all_tcycles.extend(tc_r.tolist())

        r_max = max(all_tcycles) if all_tcycles else 1.0

        # Wind arrow at θ=0 (top), pointing downward into the plot
        ax.annotate('', xy=(0, r_max * 0.55), xytext=(0, r_max * 0.85),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
        ax.text(0, r_max * 0.92, 'Wind', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
        ax.set_title(
            "Cycle time $T_{cycle}$ [s] — wind frame\n"
            "(coloured by $V_{ref}$)",
            pad=14,
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

        if output or save:
            out = Path(output) if output else _FIGURES_DIR / "cycle_times.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_obj_polar(self, ax=None, save: bool = False, output=None):
        """
        Objective polar diagram.

        Radius = obj (VMG along heading θ); angle = θ (the solver's heading angle).
        Produces closed curves — one per V_ref level — coloured by wind speed.
        The curve is enforced symmetric about the upwind/downwind axis by plotting
        the right half (θ ∈ [0, π]) and mirroring it to the left half (θ ∈ [π, 2π]).
        These curves do NOT represent actual physical translation direction or speed;
        they show the raw optimiser objective as a function of the requested heading.
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
        else:
            fig = ax.get_figure()

        V_refs = self._unique_V_refs()
        cmap   = plt.cm.plasma
        norm   = Normalize(vmin=min(V_refs), vmax=max(V_refs))

        all_objs = []

        for v_ref in V_refs:
            containers = sorted(
                [c for c in self.containers if c.V_ref == v_ref],
                key=lambda c: c.theta,
            )
            if not containers:
                continue

            thetas = np.array([c.theta for c in containers])
            objs   = np.array([c.obj   for c in containers])

            # Right arc: θ ∈ [0, π] only
            mask_r = thetas <= np.pi + 1e-9
            th_r   = thetas[mask_r]
            obj_r  = objs[mask_r]

            # Left (mirror) arc: 2π − θ for θ ∈ [0, π], reversed to run from 2π → π
            th_l  = (2 * np.pi - th_r)[::-1]
            obj_l = obj_r[::-1]

            color = cmap(norm(v_ref))
            ax.plot(th_r, obj_r, color=color, lw=1.5, alpha=0.9)
            ax.plot(th_l, obj_l, color=color, lw=1.5, alpha=0.9)

            all_objs.extend(obj_r.tolist())

        r_max = max(all_objs) if all_objs else 1.0

        # Wind arrow at θ=0 (top), pointing downward into the plot
        ax.annotate('', xy=(0, r_max * 0.55), xytext=(0, r_max * 0.85),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=20))
        ax.text(0, r_max * 0.92, 'Wind', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30), labels=[''] * 12)
        ax.set_title(
            "Objective polar — wind frame\n"
            r"(radius = VMG along $\theta$, coloured by $V_{ref}$)",
            pad=14,
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='$V_{ref}$ [m/s]', pad=0.12, shrink=0.75)

        if output or save:
            out = Path(output) if output else _FIGURES_DIR / "obj_polar.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_vmg_polar(self, ax=None, save: bool = False):
        """
        VMG-style polar diagram.

        Radius = mean ground speed magnitude; angle = actual travel direction.
        """
        colors, styles = self._color_style_maps()

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        else:
            fig = ax.get_figure()

        for (n, v), containers in self._group().items():
            u_means = np.array([np.mean(c.u) for c in containers])
            v_means = np.array([np.mean(c.v) for c in containers])

            angles = np.arctan2(u_means, v_means) % (2 * np.pi)
            radii  = np.sqrt(u_means**2 + v_means**2)

            ax.scatter(angles, radii,
                       color=colors[v], s=10,
                       label=f'N={n}, V_ref={v}')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        if save:
            self._save(fig, "vmg_polar.png")
        return fig, ax

    def plot_altitude(self, containers: 'Container | list[Container]',
                      ax=None, save: bool = False):
        """Altitude h vs time for one or more containers."""
        if isinstance(containers, Container):
            containers = [containers]

        colors, styles = self._color_style_maps()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for c in containers:
            t = np.arange(len(c.h)) * c.dt
            ax.plot(t, c.h,
                    color=colors[c.V_ref], linestyle=styles[c.N],
                    label=f'θ={c.theta:.2f}, V_ref={c.V_ref}, N={c.N}')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Altitude [m]')
        ax.legend()

        if save:
            self._save(fig, "altitude.png")
        return fig, ax

    def plot_3d_trajectories(self, save: bool = False):
        """Plot 3-D dynamic soaring orbits using Plotly, coloured by theta."""
        fig = go.Figure()
        for c in self.containers:
            u, v, h, dt = c.u, c.v, c.h, c.dt
            N = len(u)
            x, y = np.zeros(N), np.zeros(N)
            for i in range(1, N):
                x[i] = x[i-1] + 0.5 * (u[i] + u[i-1]) * dt
                y[i] = y[i-1] + 0.5 * (v[i] + v[i-1]) * dt
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=h, mode='lines',
                line=dict(width=5),
                name=f"\u03b8={c.theta:.2f}",
            ))

        fig.update_layout(
            title="Dynamic Soaring Orbits",
            scene=dict(
                xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='h [m]',
                aspectmode='cube',
            ),
        )

        if save:
            _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(_FIGURES_DIR / "trajectories.html"))

        fig.show()
