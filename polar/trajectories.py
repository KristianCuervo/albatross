

class PathFamily:
    def __init__(self, paths: list):
        self.paths = paths  # list of dicts from get_solution_dict()

    def plot_attribute(self, attr):
        """Plot path[attr] vs normalised cycle time for every path, coloured by theta."""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 4))
        all_thetas = [p['theta'] for p in self.paths]
        cmap = mpl.cm.get_cmap('twilight')
        norm = mpl.colors.Normalize(vmin=min(all_thetas), vmax=max(all_thetas))
        sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for path in self.paths:
            data   = np.asarray(path[attr])
            t_norm = np.linspace(0, 1, len(data))
            color  = mpl.colors.to_hex(sm.to_rgba(path['theta']))
            ax.plot(t_norm, data, color=color)

        fig.colorbar(sm, ax=ax, pad=0.02, label=r'$\theta$ [rad]')
        ax.set_xlabel(r'$t\,/\,t_\mathrm{cycle}$')
        ax.set_ylabel(attr)
        ax.set_title(attr)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_polar_attribute(self, attr):
        """Plot a scalar per path on polar axes (theta = wind heading, r = attribute)."""
        import matplotlib.pyplot as plt
        import numpy as np

        thetas = [p['theta'] for p in self.paths]
        values = []
        for p in self.paths:
            v = p[attr]
            values.append(float(np.mean(v)) if hasattr(v, '__len__') else float(v))

        thetas_c = thetas + [thetas[0]]
        values_c = values + [values[0]]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.plot(thetas_c, values_c)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(attr, va='bottom', fontsize=13)
        ax.set_rlabel_position(0)
        plt.tight_layout()
        return fig, ax

    def plot_3d_trajectories(self):
        """Plot 3-D dynamic soaring orbits using Plotly, coloured by theta."""
        import plotly.graph_objects as go
        import matplotlib as mpl
        import numpy as np

        all_thetas = [p['theta'] for p in self.paths]
        cmap = mpl.cm.get_cmap('twilight')
        norm = mpl.colors.Normalize(vmin=min(all_thetas), vmax=max(all_thetas))

        fig = go.Figure()
        for path in self.paths:
            u, v, h, dt = path['u'], path['v'], path['h'], path['dt']
            N = len(u)
            x, y = np.zeros(N), np.zeros(N)
            for i in range(1, N):
                x[i] = x[i-1] + 0.5 * (u[i] + u[i-1]) * dt
                y[i] = y[i-1] + 0.5 * (v[i] + v[i-1]) * dt
            color = mpl.colors.to_hex(cmap(norm(path['theta'])))
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=h, mode='lines',
                line=dict(color=color, width=5),
                name=f"\u03b8={path['theta']:.2f}",
            ))

        fig.update_layout(
            title=f"Dynamic Soaring Orbits (V_ref={self.paths[0]['V_ref']:.1f} m/s)",
            scene=dict(xaxis_title='x [m]', yaxis_title='y [m]',
                       zaxis_title='h [m]', aspectmode='cube'),
        )
        return fig
