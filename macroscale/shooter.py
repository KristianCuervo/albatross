from system import System
from integrator import Integrator, Euler, Leapfrog
from storage import Diagnostic, State, Control
import numpy as np
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, system: System, integrator: Integrator, 
                 x0: np.ndarray, lam0:np.ndarray, dt: float, T: float, t0:float = 0.0):
        self.system = system
        self.integrator = integrator

        self.dt = dt
        self.T = T
        self.nt = int(T / dt)

        self.state = State(x=x0, lam=lam0, t=t0)

        self.states: list[State] = [self.state]
        self.controls: list[Control] = []
        self.diagnostics: list[Diagnostic] = []

    def simulate(self):
        for _ in range(self.nt):
            self.system.wind.set_time(self.state.t)
            new_state, control, diagnostic = self.integrator.step(self.system, self.state, self.dt)
            self.states.append(new_state)
            self.controls.append(control)
            self.diagnostics.append(diagnostic)
            self.state = new_state
            #if self.system.wind.speed(new_state.x) < 9.0:
            #    break # ds reaches border
    
    def debug(self) -> tuple[plt.Figure, plt.Figure]:
        t_states = np.array([s.t / 3600 for s in self.states])
        t_ctrl = t_states[:-1]

        # distance from origin
        r = np.array([np.linalg.norm(s.x) for s in self.states])
        # turning rate
        u = np.array([np.arctan2(c.v[0], c.v[1]) % (2*np.pi) for c in self.controls])
        du = np.gradient(u, axis=0)
        # co-state rate of change
        # co-state argument
        lam_arg = np.array([np.arctan2(s.lam[0], s.lam[1]) % (2*np.pi) for s in self.states])
        dlam_arg_dt = np.gradient([l for l in lam_arg]) / self.dt
        delta = (u - lam_arg[:-1] + np.pi) % (2 * np.pi) - np.pi
        if len(delta) > 0:
            mean_delta = float(np.mean(delta))
            mean_abs_delta = float(np.mean(np.abs(delta)))
            print("mean delta=", mean_delta, "mean |delta|=", mean_abs_delta)
        # graph them versus time
        fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))
        axs1[0, 0].plot(t_states, r / 1e3)
        axs1[0, 0].set_title('Distance from origin (km)')
        axs1[0, 1].plot(t_ctrl, u, label='Heading')
        axs1[0, 1].plot(t_states, lam_arg, label='Co-state')
        axs1[0, 1].legend()
        axs1[0, 1].set_title('Optimal heading (rad)')
        axs1[1, 0].plot(t_ctrl, du, label='Heading rate of change')
        axs1[1, 0].plot(t_states, dlam_arg_dt, label='Co-state rate of change')
        axs1[1, 0].set_title('Co-state rate of change')
        axs1[1, 0].legend()
        #axs1[1, 1].plot(t_ctrl, [np.linalg.matrix_norm(d.hull_grad) for d in self.diagnostics], label='hull gradient')
        #axs1[1, 1].plot(t_ctrl, [np.linalg.matrix_norm(d.wind_grad) for d in self.diagnostics], label='wind gradient')
        axs1[1, 1].plot(t_states, [np.linalg.norm(s.lam) for s in self.states], label='Co-state magnitude')
        axs1[1, 1].legend()
        axs1[1, 1].set_title('Co-state magntitude')
        plt.tight_layout()


class Shooter:
    """
    Shoots trajectories given an initial state, and varying co-states on the unit cirlce
    """
    def __init__(self, system: System, integrator: Integrator,
                 x0: np.ndarray, dt: float, T: float):
        self.system = system
        self.integrator = integrator
        self.x0 = x0
        self.dt = dt
        self.T = T

    
    def shoot(self, n: int) -> list[Trajectory]:
        trajectories = []
        for i in range(n):
            self.integrator.reset()
            self.system._last_theta = None

            angle = 2 * np.pi * i / n + np.pi / 2
            lam0 = np.array([np.cos(angle), np.sin(angle)])
            traj = Trajectory(self.system, self.integrator, self.x0, lam0, self.dt, self.T)
            traj.simulate()
            trajectories.append(traj)
        return trajectories
    



def test_boundaries():
    import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from macroscale.hull import Hull
    from wind import Vortex, Downwind, Waterfall, Rotation
    from visualise import Visualiser
    hull = Hull()
    #hull = SmoothHull(delta=0.1, p=1)
    #wind = Vortex(center=np.array([10e4, 10e4]), V_max=20., R_max=1e5, alpha=1.0, beta=0.0, U_inf=0.)
    wind = Downwind(w=np.array([0.0, -20.0]), decay=1e-6)
    #wind = Waterfall(v0=20., strip_half_width=100e3, decay=5e-6)
    #wind = Rotation(center=np.array([10e4, 10e4]), strength=20, radius=1e5)
    system = System(hull, wind)
    integrator = Leapfrog()
    trajectory = Trajectory(system, integrator, 
                            x0=np.array([0., 0.]), 
                            lam0=np.array([0., -1.0]), 
                            dt=100.0, T=2 * 24 * 3600.0)
    trajectory.simulate()
    anim = Visualiser(trajectories=[trajectory], wind=wind).animate(interval=100, repeat=True)
    trajectory.debug()
    plt.show()



def test_shooting():
    import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from hull import Hull, SmoothHull
    from wind import Waterfall, Downwind, Vortex, Rotation
    from visualise import Visualiser
    hull = Hull()
    #wind = Waterfall(v0=20., strip_half_width=100e3, decay=1e-6)
    wind = Downwind(w=np.array([0.0, -20.0]), decay=1e-6)
    #wind = Vortex(center=np.array([10e4, 10e4]), V_max=20., R_max=1e5, alpha=1.0, beta=0.0, U_inf=0.)
    #wind = Vortex(center=np.array([-9e5, 0.0]), V_max=20., R_max=1e6, alpha=1.0, beta=0.0, center_velocity=np.array([8.0, 0.0]))
    # Need to add dynamic visualisation of the wind fields
    #wind = Rotation(center=np.array([0, -50e4]), strength=20, radius=1e6)
    system = System(hull, wind)
    integrator = Euler()
    shooter = Shooter(system=system, integrator=integrator, x0=np.array([0., 0.]), dt=100.0, T = 24 * 1 * 3600.0)
    trajectories = shooter.shoot(n=12)
    vis = Visualiser(trajectories=trajectories, wind=wind)
    anim = vis.animate(interval=5, repeat=False)
    #iso = vis.isochrone(n_steps=12, background=True)
    plt.show()
    
if __name__ == '__main__':
    #test_boundaries()
    test_shooting()
    pass
