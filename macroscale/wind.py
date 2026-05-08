from abc import ABC, abstractmethod
import numpy as np


class Wind(ABC):
    @abstractmethod
    def velocity(self, x: np.ndarray) -> np.ndarray: ...  # 2D wind vector

    def speed(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.velocity(x))
    
    def set_time(self, t: float) -> None:
        """Optional method to set the time for time-varying winds."""
        pass

    def gradient(self, x: np.ndarray, h: float = 1e3) -> np.ndarray:
        w_xp = self.velocity(x + np.array([h, 0.0]))
        w_xm = self.velocity(x - np.array([h, 0.0]))
        w_yp = self.velocity(x + np.array([0.0, h]))
        w_ym = self.velocity(x - np.array([0.0, h]))

        dw_dx = (w_xp - w_xm) / (2 * h)
        dw_dy = (w_yp - w_ym) / (2 * h)

        return np.column_stack([dw_dx, dw_dy])


class Vortex(Wind):
    """Modified Rankine Vortex (Holland 1980 / Southern-Hemisphere convention)."""

    def __init__(
        self,
        center: np.ndarray,
        V_max: float = 25.0,       # m/s — peak tangential wind speed
        R_max: float = 300_000.0,  # m   — radius of maximum wind
        alpha: float = 0.8,        # —   — outer power-law decay exponent
        beta:  float = 0.0,        # rad — inflow angle (0 = pure tangential)
        U_inf: float = 0.0,        # m/s — uniform eastward background
        center_velocity: np.ndarray | None = None,  # m/s — center translation
        t0: float = 0.0,            # s — reference time for center position
    ):
        self.center = center
        self._center0 = np.array(center, dtype=float)
        self.center_velocity = (
            np.zeros(2, dtype=float)
            if center_velocity is None
            else np.array(center_velocity, dtype=float)
        )
        self._t0 = float(t0)
        self.V_max = V_max
        self.R_max = R_max
        self.alpha = alpha
        self.beta  = beta
        self.U_inf = U_inf

    def set_time(self, t: float) -> None:
        dt = float(t) - self._t0
        self.center = self._center0 + self.center_velocity * dt

    def velocity(self, x: np.ndarray) -> np.ndarray:
        dx, dy = x[0] - self.center[0], x[1] - self.center[1]
        r      = np.sqrt(dx**2 + dy**2)
        r_safe = max(r, 1.0)
        theta  = np.arctan2(dy, dx)

        V = (self.V_max * r / self.R_max if r <= self.R_max
             else self.V_max * (self.R_max / r_safe) ** self.alpha)

        # Clockwise rotation (Southern Hemisphere)
        u_v = +V * np.sin(theta)
        v_v = -V * np.cos(theta)

        cb, sb = np.cos(self.beta), np.sin(self.beta)
        u = self.U_inf + u_v * cb - np.cos(theta) * V * sb
        v =              v_v * cb - np.sin(theta) * V * sb

        return np.array([u, v])

class Rotation(Wind):
    """Uniform rotation around a center point, with speed uniform until a certain radius"""
    def __init__(self, center: np.ndarray, strength: float, radius: float = 1e5):
        self.center = center
        self.strength = strength
        self.radius = radius

    def velocity(self, x: np.ndarray) -> np.ndarray:
        # 20 m/s within a radius of 100 km, then nothing
        dx, dy = x[0] - self.center[0], x[1] - self.center[1]
        r = np.sqrt(dx**2 + dy**2)
        if r <= self.radius:
            theta = np.arctan2(dy, dx)
            u = +self.strength * np.sin(theta)
            v = -self.strength * np.cos(theta)
            return np.array([u, v])
        else:
            return np.array([0., 0.])

class Downwind(Wind):
    def __init__(self, w: np.ndarray, decay: float = 0.0):
        """
        Uniform wind with optional decay in a given direction, 
        """
        self.w = w
        self.decay = decay  # k

    def velocity(self, x: np.ndarray) -> np.ndarray:
        if self.decay == 0.0:
            return self.w
        else:
            # Exponential decay in the perpendicular direction to the wind, with length scale 1/k
            w_dir = self.w / np.linalg.norm(self.w)
            perp_dir = np.array([-w_dir[1], w_dir[0]])
            distance = abs(perp_dir @ x)
            decay_factor = np.exp(-self.decay * distance)
            return self.w * decay_factor
    



class Waterfall(Wind):
    """Uniform southward wind within a central strip, exponentially decaying outside.

    Speed profile in x:  v0         for |x| <= strip_half_width
                         v0·exp(-k·(|x| − strip_half_width))  otherwise
    """

    def __init__(
        self,
        v0: float,
        strip_half_width: float,  # m — half-width of the flat-top region
        decay: float = 2e-6,      # m⁻¹ — exponential decay rate outside the strip
    ):
        self.v0 = v0
        self.strip_half_width = strip_half_width
        self.decay = decay

    def velocity(self, x: np.ndarray) -> np.ndarray:
        excess = max(abs(x[0]) - self.strip_half_width, 0.0)
        speed = self.v0 * np.exp(-self.decay * excess)
        return np.array([0., -speed])


def _wind_speed_grid(wind: Wind, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.array([[wind.speed(np.array([x, y])) for x in xs] for y in ys])


def _jacobian_magnitude_grid(wind: Wind, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.array([
        [np.linalg.norm(wind.gradient(np.array([x, y]))) for x in xs]
        for y in ys
    ])


def _plot_wind_field(wind: Wind, title: str, ax, xs: np.ndarray, ys: np.ndarray):
    import matplotlib.pyplot as plt
    speed = _wind_speed_grid(wind, xs, ys)
    im = ax.pcolormesh(xs / 1e3, ys / 1e3, speed, cmap="plasma", shading="auto")
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    plt.colorbar(im, ax=ax, label="speed (m/s)")


def _plot_jacobian(wind: Wind, title: str, ax, xs: np.ndarray, ys: np.ndarray):
    import matplotlib.pyplot as plt
    mag = _jacobian_magnitude_grid(wind, xs, ys) * 1e3  # s⁻¹ → (m/s)/km
    im = ax.pcolormesh(xs / 1e3, ys / 1e3, mag, cmap="plasma", shading="auto")
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    plt.colorbar(im, ax=ax, label="|∇w| ((m/s)/km)")


def test_plotting() -> None:
    import matplotlib.pyplot as plt

    vortex   = Vortex(center=np.array([0., 0.]))
    downwind = Downwind(v0=10., decay=2e-6)
    waterfall = Waterfall(v0=10., strip_half_width=500_000., decay=3e-6)

    extent = 1_500_000.
    xs = np.linspace(-extent, extent, 200)
    ys = np.linspace(-extent, extent, 200)

    # Wind speed fields
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle("Wind speed")
    _plot_wind_field(vortex,    "Vortex (MRV)", axes1[0], xs, ys)
    _plot_wind_field(downwind,  "Downwind",     axes1[1], xs, ys)
    _plot_wind_field(waterfall, "Waterfall",    axes1[2], xs, ys)
    fig1.tight_layout()

    # Jacobian magnitude fields
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Jacobian magnitude |∇w|")
    _plot_jacobian(vortex,    "Vortex (MRV)", axes2[0], xs, ys)
    _plot_jacobian(downwind,  "Downwind",     axes2[1], xs, ys)
    _plot_jacobian(waterfall, "Waterfall",    axes2[2], xs, ys)
    fig2.tight_layout()

    plt.show()


def test_movement() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    extent = 1_500_000.
    xs = np.linspace(-extent, extent, 200)
    ys = np.linspace(-extent, extent, 200)
    xq = np.linspace(-extent, extent, 20)
    yq = np.linspace(-extent, extent, 20)
    Xq, Yq = np.meshgrid(xq, yq)

    vortex = Vortex(
        center=np.array([-6.0e5, 0.0]),
        center_velocity=np.array([8.0, 0.0]),
        V_max=25.0,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Moving vortex wind field")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    speed = _wind_speed_grid(vortex, xs, ys)
    im = ax.pcolormesh(xs / 1e3, ys / 1e3, speed, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax, label="speed (m/s)")

    U = np.zeros((len(yq), len(xq)))
    V = np.zeros((len(yq), len(xq)))
    for i in range(len(yq)):
        for j in range(len(xq)):
            w = vortex.velocity(np.array([xq[j], yq[i]]))
            mag = np.linalg.norm(w)
            if mag > 0:
                U[i, j] = w[0] / mag
                V[i, j] = w[1] / mag

    quiver = ax.quiver(Xq / 1e3, Yq / 1e3, U, V, color="#1a1a1a", alpha=0.7)
    center_dot, = ax.plot(vortex.center[0] / 1e3, vortex.center[1] / 1e3,
                          "ko", ms=6, zorder=5)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        t = frame * 900.0
        vortex.set_time(t)
        speed = _wind_speed_grid(vortex, xs, ys)
        im.set_array(speed.ravel())
        center_dot.set_data([vortex.center[0] / 1e3], [vortex.center[1] / 1e3])
        for i in range(len(yq)):
            for j in range(len(xq)):
                w = vortex.velocity(np.array([xq[j], yq[i]]))
                mag = np.linalg.norm(w)
                if mag > 0:
                    U[i, j] = w[0] / mag
                    V[i, j] = w[1] / mag
                else:
                    U[i, j] = 0.0
                    V[i, j] = 0.0
        quiver.set_UVC(U, V)
        time_text.set_text(f"t = {t / 3600:.1f} h")
        return im, quiver, center_dot, time_text

    anim = animation.FuncAnimation(fig, update, frames=80, interval=80, blit=False)
    fig._anim = anim
    plt.show()
    return anim


if __name__ == "__main__":
    #test_plotting()
    test_movement()
