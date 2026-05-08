import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize


class Visualiser:
    """Animates a list of Trajectory objects over a wind-field background."""

    def __init__(self, trajectories: list, wind, figsize: tuple = (10, 8)):
        self.trajectories = trajectories
        self.wind = wind
        self.figsize = figsize

    def _extent(self, pad: float = 0.15):
        all_x = np.concatenate([[s.x[0] for s in t.states] for t in self.trajectories])
        all_y = np.concatenate([[s.x[1] for s in t.states] for t in self.trajectories])
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        rx = max((xmax - xmin) * pad, 1e5)
        ry = max((ymax - ymin) * pad, 1e5)
        return xmin - rx, xmax + rx, ymin - ry, ymax + ry

    def _draw_background(self, ax, xmin, xmax, ymin, ymax,
                         n_mesh: int = 120, n_quiver: int = 16):
        xs = np.linspace(xmin, xmax, n_mesh)
        ys = np.linspace(ymin, ymax, n_mesh)
        xq = np.linspace(xmin, xmax, n_quiver)
        yq = np.linspace(ymin, ymax, n_quiver)
        speed, U, V = self._background_fields(xs, ys, xq, yq)
        cmap = plt.cm.plasma.copy()
        cmap.set_under('white')
        im = ax.pcolormesh(xs / 1e3, ys / 1e3, speed,
                           cmap=cmap, shading='auto', alpha=0.4, vmin=9)
        Xq, Yq = np.meshgrid(xq, yq)

        x_range_km = (xmax - xmin) / 1e3
        arrow_scale = n_quiver / x_range_km * 2.0
        quiver = ax.quiver(Xq / 1e3, Yq / 1e3, U, V,
                           color='#1a1a1a', alpha=0.7, scale=arrow_scale,
                           scale_units='xy', headwidth=4, headlength=5)
        return im, quiver, xs, ys, xq, yq

    def _set_wind_time(self, t_secs: float) -> None:
        for name in ("set_time", "set_t", "set_unix_time", "set_timestamp"):
            method = getattr(self.wind, name, None)
            if callable(method):
                method(t_secs)
                return
        if hasattr(self.wind, "t"):
            try:
                self.wind.t = t_secs
            except Exception:
                pass

    def _background_fields(self, xs: np.ndarray, ys: np.ndarray,
                           xq: np.ndarray, yq: np.ndarray) -> tuple:
        speed = np.array([[self.wind.speed(np.array([x, y])) for x in xs] for y in ys])
        U = np.zeros((len(yq), len(xq)))
        V = np.zeros((len(yq), len(xq)))
        for i in range(len(yq)):
            for j in range(len(xq)):
                w = self.wind.velocity(np.array([xq[j], yq[i]]))
                mag = np.linalg.norm(w)
                if mag > 0:
                    U[i, j] = w[0] / mag
                    V[i, j] = w[1] / mag
        return speed, U, V

    def animate(self, interval: int = 60, repeat: bool = True) -> animation.FuncAnimation:
        xmin, xmax, ymin, ymax = self._extent()
        nframes = max(len(t.states) for t in self.trajectories)

        fig, ax = plt.subplots(figsize=self.figsize)
        im, quiver, xs, ys, xq, yq = self._draw_background(ax, xmin, xmax, ymin, ymax)
        plt.colorbar(im, ax=ax, label='Wind speed (m/s)')

        ax.set_xlim(xmin / 1e3, xmax / 1e3)
        ax.set_ylim(ymin / 1e3, ymax / 1e3)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('Macroscale trajectory animation')
        ax.plot(0, 0, 'k*', ms=12, zorder=6, label='Start')
        ax.legend(loc='lower right', fontsize=9)

        colors = plt.cm.tab10(np.linspace(0, 0.9, len(self.trajectories)))
        lines = [ax.plot([], [], '-', color=c, lw=2, alpha=0.9)[0] for c in colors]
        dots  = [ax.plot([], [], 'o', color=c, ms=7, zorder=5)[0] for c in colors]
        time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=11,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        xs_all = [[s.x[0] / 1e3 for s in t.states] for t in self.trajectories]
        ys_all = [[s.x[1] / 1e3 for s in t.states] for t in self.trajectories]
        lam_all = [[float(np.linalg.norm(s.lam)) for s in t.states] for t in self.trajectories]
        lam_vals = np.concatenate([np.array(l) for l in lam_all])
        lam_vmin = float(lam_vals.min())
        lam_vmax = float(lam_vals.max() or 1.0)
        if lam_vmin == lam_vmax:
            lam_vmax = lam_vmin + 1.0
        lam_norm = Normalize(vmin=lam_vmin, vmax=lam_vmax)
        lam_cmap = plt.cm.viridis
        lam_sm = plt.cm.ScalarMappable(norm=lam_norm, cmap=lam_cmap)
        lam_sm.set_array([])
        fig.colorbar(lam_sm, ax=ax, label='Co-state magnitude', location='left', pad=0.08)

        def init():
            for line, dot in zip(lines, dots):
                line.set_data([], [])
                dot.set_data([], [])
            time_text.set_text('')
            return (im, quiver, *lines, *dots, time_text)

        def update(frame):
            t_secs = self.trajectories[0].states[
                min(frame, len(self.trajectories[0].states) - 1)
            ].t
            self._set_wind_time(t_secs)
            speed, U, V = self._background_fields(xs, ys, xq, yq)
            im.set_array(speed.ravel())
            quiver.set_UVC(U, V)
            for i, (line, dot) in enumerate(zip(lines, dots)):
                end = min(frame + 1, len(xs_all[i]))
                line.set_data(xs_all[i][:end], ys_all[i][:end])
                dot.set_data([xs_all[i][end - 1]], [ys_all[i][end - 1]])
                lam_mag = lam_all[i][end - 1]
                color = lam_cmap(lam_norm(lam_mag))
                line.set_color(color)
                dot.set_color(color)
            time_text.set_text(f't = {t_secs / 3600:.1f} h')
            return (im, quiver, *lines, *dots, time_text)

        anim = animation.FuncAnimation(
            fig, update, frames=nframes, init_func=init,
            interval=interval, blit=True, repeat=repeat,
        )
        return anim