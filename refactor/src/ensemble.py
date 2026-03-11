from .container import Container
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

LINESTYLES = ['-', '--', ':', '-.']

# Figures directory sits one level above this file (refactor/figures/)
_FIGURES_DIR = Path(__file__).parent.parent / "figures"


class Ensemble:
    """
    Holds all Containers across all (N, V_ref, theta) combinations.
    Provides grouping and plotting utilities.
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
        return min(self.containers, key=lambda c: c.obj)

    def best_at(self, theta: float) -> Container:
        return min(self.containers, key=lambda c: abs(c.theta - theta))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unique_V_refs(self) -> list[float]:
        return sorted(set(c.V_ref for c in self.containers))

    def _unique_Ns(self) -> list[int]:
        return sorted(set(c.N for c in self.containers))

    def _group(self) -> dict[tuple, list[Container]]:
        """Returns {(N, V_ref): [containers sorted by theta]}"""
        groups = {}
        for c in self.containers:
            key = (c.N, c.V_ref)
            groups.setdefault(key, []).append(c)
        for key in groups:
            groups[key].sort(key=lambda c: c.theta)
        return groups

    def _color_style_maps(self):
        """Shared colour (V_ref) and linestyle (N) mappings for polar plots."""
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

    def plot_ground_speed(self, ax=None, save: bool = False):
        """
        Polar plot of optimised ground speed (−obj) vs heading theta.
        Colour encodes V_ref; linestyle encodes N.
        Returns (fig, ax).
        """
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

    def plot_cycle_times(self, ax=None, save: bool = False):
        """
        Polar plot of T_cycle [s] vs heading theta.
        Colour encodes V_ref; linestyle encodes N.
        Returns (fig, ax).
        """
        colors, styles = self._color_style_maps()

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        else:
            fig = ax.get_figure()

        for (n, v), containers in self._group().items():
            thetas  = np.array([c.theta   for c in containers])
            tcycles = np.array([c.T_cycle for c in containers])
            thetas  = np.append(thetas,  thetas[0])
            tcycles = np.append(tcycles, tcycles[0])
            ax.plot(thetas, tcycles,
                    color=colors[v], linestyle=styles[n],
                    label=f'N={n}, V_ref={v}')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("T_cycle [s]", pad=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        if save:
            self._save(fig, "cycle_times.png")

        return fig, ax

    def plot_vmg_polar(self, ax=None, save: bool = False):
        """
        VMG-style polar diagram (cf. sailboat polars).

        For each container the bird's mean ground velocity (u_mean, v_mean) is
        computed.  The point is plotted in polar coordinates:
          - radius  = sqrt(u_mean² + v_mean²)  — actual travel speed [m/s]
          - angle   = atan2(u_mean, v_mean)     — actual travel direction from
                      the +y axis (y=0 is 'straight downwind', angles increase
                      clockwise like a compass bearing)

        Unlike plot_ground_speed, the angle here is the direction the bird
        *actually travels*, not the optimisation heading theta.
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
        """
        Altitude h vs time for one or more containers.
        Colour encodes V_ref; linestyle encodes N.
        Returns (fig, ax).
        """
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
                xaxis_title='x [m]', yaxis_title='y [m]',
                zaxis_title='h [m]',
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=1),
            ),
        )

        if save:
            _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(_FIGURES_DIR / "trajectories.html"))

        fig.show()
