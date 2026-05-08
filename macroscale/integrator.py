from system import System
from storage import State, Control

class Integrator:
    def step(self, system: System, state: State, dt: float):
        raise NotImplementedError("Implement the step function for the integrator.")

    def reset(self) -> None:
        """Reset integrator state so the next trajectory starts cleanly."""
        pass


class Euler(Integrator):
    def step(self, system: System, state: State, dt: float) -> tuple[State, Control]:
        dx, dlam, control, diagnostic = system.rhs(state)
        x_next = state.x + dx * dt
        lam_next = state.lam + dlam * dt
        return State(x=x_next, lam=lam_next, t=state.t + dt), control, diagnostic


class Leapfrog(Integrator):
    def __init__(self):
        self.first_half_kick = True

    def reset(self) -> None:
        self.first_half_kick = True

    def half_kick(self, system: System, state: State, dt: float) -> State:
        dlam, _ = system.dlam_dt(state, system.optimal_control(state))
        lam_half = state.lam + dlam * (dt * 0.5)
        return State(x=state.x, lam=lam_half, t=state.t)

    def step(self, system: System, state: State, dt: float) -> tuple[State, Control]:
        if self.first_half_kick:
            state = self.half_kick(system, state, dt)
            self.first_half_kick = False
        dx, dlam, control, diagnostic = system.rhs(state)
        x_next = state.x + dx * dt
        lam_next = state.lam + dlam * dt
        return State(x=x_next, lam=lam_next, t=state.t + dt), control, diagnostic





