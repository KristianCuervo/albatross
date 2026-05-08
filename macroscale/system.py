import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from macroscale.hull import Hull
from wind import Downwind, Wind
from storage import State, Control, Diagnostic, Record
from utils import _alpha, _rotation

class System:
    """
    Defines the dynamics of the hamiltonian system 
    """
    def __init__(self, hull: Hull, wind: Wind, debug: bool = False):
        self.hull = hull
        self.wind = wind
        self.debug = debug
        self._last_theta: float | None = None

    def optimal_control(self, state: State, n_sweep: int = 360) -> Control:
        """Sweep wind-frame headings and return the one maximising Λ · V."""
        w = self.wind.velocity(state.x)
        w_mag = np.linalg.norm(w)
        alpha = _alpha(w)

        thetas = np.linspace(0, 2 * np.pi, n_sweep, endpoint=False)
        h_vals = np.empty_like(thetas)
        for i, theta in enumerate(thetas):
            v_alpha = self.hull.velocity(v_ref=w_mag, angle=theta)
            v_geo = _rotation(alpha) @ v_alpha
            h_vals[i] = state.lam @ v_geo
        max_val = float(np.max(h_vals))
        candidates = np.where(np.isclose(h_vals, max_val, rtol=1e-9, atol=1e-12))[0]
        if self._last_theta is not None and len(candidates) > 1:
            deltas = np.abs(((thetas[candidates] - self._last_theta + np.pi) % (2 * np.pi)) - np.pi)
            idx_best = int(candidates[int(np.argmin(deltas))])
        else:
            idx_best = int(np.argmax(h_vals))
        theta_star = float(thetas[idx_best])
        v_star = _rotation(alpha) @ self.hull.velocity(v_ref=w_mag, angle=theta_star)
        self._last_theta = theta_star
        return Control(u=theta_star, v=v_star)

    def dx_dt(self, control: Control) -> np.ndarray:
        return control.v

    def dlam_dt(self, state: State, control: Control) -> np.ndarray:
        w = self.wind.velocity(state.x)
        dv_dw = self.hull.gradient(w=w, theta=control.u)
        dw_dx = self.wind.gradient(state.x)
        dlam_dt = - (dv_dw @ dw_dx).T @ state.lam
        alpha = float(np.arctan2(-w[0], -w[1]))
        theta = control.u 
        diagnostic = Diagnostic(hull_grad=dv_dw, wind_grad=dw_dx, v_ref=np.linalg.norm(w), alpha=alpha, theta=theta)
        return dlam_dt, diagnostic
    
    def rhs(self, state: State) -> tuple[np.ndarray, np.ndarray, Control]:
        control = self.optimal_control(state)
        dx = self.dx_dt(control)
        dlam, diagnostic = self.dlam_dt(state, control)
        return dx, dlam, control, diagnostic


def test_hamiltonian():
    """
    Test a single step of optimal control
    """
    hull = Hull()
    wind = Downwind(v0=15, decay=1e-6)
    system = System(hull, wind)
    state = State(x=np.array([0., 0.]), lam=np.array([0.0, 1.0]), t=0.0)
    control = system.optimal_control(state)
    dx = system.dx_dt(control)
    dlam, diagnostic = system.dlam_dt(state, control)
    print("Optimal heading (rad):", control.u)
    print("Optimal velocity (m/s):", control.v)
    print("State derivative dx/dt:", dx)
    print("Co-state derivative dlam/dt:", dlam)
    print(system.hull.velocity(v_ref=wind.v0, angle=0.0))

    w = wind.velocity(state.x)
    w_mag = np.linalg.norm(w)
    alpha = _alpha(w)
    _, u_wind, v_wind = hull.construct_arc(w_mag, n=360)
    arc_geo = _rotation(alpha) @ np.vstack([u_wind, v_wind])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(arc_geo[0], arc_geo[1], color='C0', lw=2, label='velocity arc')

    lam = state.lam
    lam_norm = np.linalg.norm(lam)
    arc_max = float(np.max(np.linalg.norm(arc_geo, axis=0)))
    if lam_norm > 0:
        lam_scale = 0.7 * arc_max / lam_norm
        lam_vec = lam * lam_scale
        ax.arrow(0.0, 0.0, lam_vec[0], lam_vec[1],
                 length_includes_head=True, head_width=0.05 * arc_max,
                 head_length=0.08 * arc_max, color='black', lw=1.5,
                 label='co-state')

    ax.scatter([control.v[0]], [control.v[1]], color='red', s=40,
               zorder=5, label='selected velocity')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x velocity (m/s)')
    ax.set_ylabel('y velocity (m/s)')
    ax.set_title('Optimal control: velocity arc and co-state')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    test_hamiltonian()