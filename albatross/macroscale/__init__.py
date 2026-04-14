"""
albatross.macroscale — 2-D migration potential using ERA5 wind data.

Exports
-------
ERA5Interpolator    : spatiotemporal bilinear wind interpolator
download_era5       : CDS API downloader
plot_wind_map       : Cartopy wind vector field visualisation
VelocityHull        : convex hull of achievable velocities; micro→macro bridge
GreedyMigration     : RK4 greedy IVP migration isocurve simulator
configure_simulation: configure simulation.py module-level state
HamiltonianShooter  : Mayer-form Hamiltonian IVP shooter with full adjoint ODE
"""

from .wind import ERA5Interpolator, download_era5, plot_wind_map
from .hull import VelocityHull
from .migration import GreedyMigration
from .simulation import configure as configure_simulation
from .ivp_shooter import HamiltonianShooter
