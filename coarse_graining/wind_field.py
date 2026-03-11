"""
wind_field.py
=============
Wind field constructors for the coarse-graining analysis.

All functions return a dict with identical keys:
  'x_centers' : (grid_size,)            cell-centre x coordinates [m]
  'y_centers' : (grid_size,)            cell-centre y coordinates [m]
  'V_ref'     : (grid_size, grid_size)  local wind-shear reference speed [m/s]
  'alpha'     : (grid_size, grid_size)  direction wind blows FROM [rad, CCW from +x]
  'XX', 'YY'  : (grid_size, grid_size)  meshgrid of centres (for plotting)

Coordinate system (global):
  x  →  East
  y  →  North

Wind-frame → global-frame velocity conversion (used in potential_map.py):
  vx_global = v_wind * cos(alpha) − u_wind * sin(alpha)
  vy_global = v_wind * sin(alpha) + u_wind * cos(alpha)

Available field constructors
-----------------------------
  make_rotational_wind_field  — single CCW vortex (baseline)
  make_multi_vortex_field     — superposition of Rankine vortices
  make_hill_field             — stream-function hills and troughs
  add_noise                   — additive smooth spatial noise on any field
"""

import numpy as np


# =========================================================================== #
# Private helpers                                                              #
# =========================================================================== #

def _make_grid(grid_size: int, domain: float):
    """Return (xs, ys, XX, YY) for a uniform cell-centred grid."""
    cell = domain / grid_size
    xs = np.arange(grid_size) * cell + cell / 2
    ys = np.arange(grid_size) * cell + cell / 2
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    return xs, ys, XX, YY


def _velocity_to_field(
    ux: np.ndarray,
    uy: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    V_ref_min: float = 9.0,
    V_ref_max: float = 25.0,
) -> dict:
    """
    Convert wind velocity arrays (where wind blows TO) to field dict.

    V_ref  = clamp(|velocity|, V_ref_min, V_ref_max)
    alpha  = direction wind blows FROM = atan2(-uy, -ux)
    When speed is zero the direction defaults to 0 (East).
    """
    speed = np.hypot(ux, uy)
    V_ref = np.clip(speed, V_ref_min, V_ref_max)

    safe_s = np.where(speed > 0, speed, 1.0)
    alpha  = np.where(speed > 0, np.arctan2(-uy / safe_s, -ux / safe_s), 0.0)

    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    return {
        'x_centers': xs,
        'y_centers': ys,
        'V_ref':     V_ref,
        'alpha':     alpha,
        'XX':        XX,
        'YY':        YY,
    }


def _rankine_vortex_velocity(
    DX: np.ndarray,
    DY: np.ndarray,
    circulation: float,
    r_core: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Velocity field of a single Rankine vortex.

    circulation > 0  →  CCW,  < 0  →  CW
    Tangential speed:
      r < r_core  →  v_tan = Γ·r / (2π r_core²)   (solid-body core)
      r ≥ r_core  →  v_tan = Γ   / (2π r)          (irrotational outer field)
    """
    R      = np.hypot(DX, DY)
    R_safe = np.where(R > 0, R, 1e-10)

    v_tan = np.where(
        R < r_core,
        circulation * R       / (2.0 * np.pi * r_core ** 2),
        circulation / R_safe  /  (2.0 * np.pi),
    )
    # CCW tangential direction at (DX, DY): (−DY, DX) / R
    ux = v_tan * (-DY / R_safe)
    uy = v_tan * ( DX / R_safe)
    return ux, uy


def _smooth_noise_2d(
    shape: tuple,
    domain: float,
    amplitude: float,
    n_modes: int = 12,
    seed=None,
) -> np.ndarray:
    """
    Smooth 2-D noise via superposed random sinusoids.

    Frequencies span 0.5–3 cycles across the domain, giving spatial
    wavelengths of domain/3 to 2·domain (long-range correlations only).
    Output is unit-RMS, scaled to amplitude.
    """
    rng = np.random.default_rng(seed)
    grid_size = shape[0]
    xs = np.linspace(0, domain, grid_size)
    ys = np.linspace(0, domain, grid_size)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    noise = np.zeros(shape)
    for _ in range(n_modes):
        kx  = rng.uniform(0.5, 3.0) * 2.0 * np.pi / domain
        ky  = rng.uniform(0.5, 3.0) * 2.0 * np.pi / domain
        phi = rng.uniform(0.0, 2.0 * np.pi)
        noise += np.cos(kx * XX + ky * YY + phi)
    std = noise.std()
    if std > 0:
        noise /= std
    return amplitude * noise


# =========================================================================== #
# Public constructors                                                          #
# =========================================================================== #

def make_rotational_wind_field(
    grid_size: int  = 100,
    domain: float   = 1000.0,
    center: tuple   = (500.0, 500.0),
    V_min: float    = 10.0,
    V_max: float    = 20.0,
) -> dict:
    """
    Single CCW vortex with wind speed ramping linearly from centre to corner.

    alpha = atan2(−DX, DY) — verified:
      Right of centre (DX>0, DY=0): blows North, FROM South → alpha = −π/2 ✓
      Above centre  (DX=0, DY>0):  blows West,  FROM East  → alpha = 0    ✓
    """
    xs, ys, XX, YY = _make_grid(grid_size, domain)
    cx, cy = center
    DX = XX - cx
    DY = YY - cy
    R  = np.hypot(DX, DY)

    R_max = np.hypot(domain / 2.0, domain / 2.0)
    V_ref = np.clip(V_min + (V_max - V_min) * R / R_max, V_min, V_max)

    alpha = np.arctan2(-DX, DY)
    alpha[R == 0] = 0.0

    return {
        'x_centers': xs,
        'y_centers': ys,
        'V_ref':     V_ref,
        'alpha':     alpha,
        'XX':        XX,
        'YY':        YY,
    }


def make_multi_vortex_field(
    vortices: list,
    grid_size: int   = 100,
    domain: float    = 1000.0,
    bg_speed: float  = 12.0,        # background wind speed [m/s]
    bg_toward: float = 0.0,         # direction background wind blows TO [rad]
    r_core: float    = 150.0,       # Rankine core radius for all vortices [m]
    V_ref_min: float = 9.0,
    V_ref_max: float = 25.0,
) -> dict:
    """
    Superposition of Rankine vortices on a uniform background wind.

    Parameters
    ----------
    vortices : list of (cx, cy, Γ)
        cx, cy   — vortex centre [m]
        Γ        — circulation [m²/s]; positive = CCW, negative = CW.
        Typical magnitude: 4000–12000 m²/s for 5–13 m/s at r_core = 150 m.
    bg_speed : float
        Background uniform wind speed [m/s].
    bg_toward : float
        Direction the background wind blows TO [rad, CCW from East].
    """
    xs, ys, XX, YY = _make_grid(grid_size, domain)

    ux = np.full_like(XX, bg_speed * np.cos(bg_toward))
    uy = np.full_like(YY, bg_speed * np.sin(bg_toward))

    for (cx, cy, circulation) in vortices:
        du, dv = _rankine_vortex_velocity(XX - cx, YY - cy, circulation, r_core)
        ux += du
        uy += dv

    return _velocity_to_field(ux, uy, xs, ys, V_ref_min, V_ref_max)


def make_hill_field(
    hills: list,
    grid_size: int   = 100,
    domain: float    = 1000.0,
    bg_speed: float  = 14.0,
    bg_toward: float = 0.0,         # background blows East by default
    V_ref_min: float = 9.0,
    V_ref_max: float = 25.0,
) -> dict:
    """
    Stream-function wind field: hills (positive A) and troughs (negative A).

    Stream function:  Ψ = Σ A_k · exp(−r_k² / (2 σ_k²))
    Wind velocity:    u = ∂Ψ/∂y,   v = −∂Ψ/∂x
    This gives CCW flow around hills (A > 0) and CW flow around troughs
    (A < 0), analogous to atmospheric high/low pressure systems.

    Verified at (cx + σ, cy) for A > 0:
      u = 0,  v = A/σ·exp(−½) > 0  →  blows North  (CCW ✓)
    At (cx, cy + σ) for A > 0:
      u = −A/σ·exp(−½) < 0          →  blows West   (CCW ✓)

    Parameters
    ----------
    hills : list of (cx, cy, A, sigma)
        cx, cy  — hill/trough centre [m]
        A       — stream-function amplitude [m²/s]; positive = hill, negative = trough.
        sigma   — Gaussian width [m].
        Velocity scale at r = sigma: |A| / (sigma · √e)  ≈  |A| / (1.65 · sigma).
    bg_speed : float
        Background uniform wind speed [m/s].
    bg_toward : float
        Direction the background wind blows TO [rad].
    """
    xs, ys, XX, YY = _make_grid(grid_size, domain)

    ux = np.full_like(XX, bg_speed * np.cos(bg_toward))
    uy = np.full_like(YY, bg_speed * np.sin(bg_toward))

    for (cx, cy, A, sigma) in hills:
        DX  = XX - cx
        DY  = YY - cy
        exp = np.exp(-(DX ** 2 + DY ** 2) / (2.0 * sigma ** 2))
        ux += -A * DY / sigma ** 2 * exp    # u = ∂Ψ/∂y
        uy +=  A * DX / sigma ** 2 * exp    # v = −∂Ψ/∂x

    return _velocity_to_field(ux, uy, xs, ys, V_ref_min, V_ref_max)


def add_noise(
    wind_field: dict,
    amplitude: float = 3.0,
    n_modes: int     = 12,
    seed=None,
) -> dict:
    """
    Add smooth spatial noise to an existing wind field.

    Noise is applied to the wind velocity components before re-deriving
    V_ref and alpha, so the result is physically consistent.

    Parameters
    ----------
    wind_field : dict
        Output of any field constructor.
    amplitude : float
        RMS amplitude of noise added to each velocity component [m/s].
    n_modes : int
        Number of sinusoidal modes (more = richer texture, slower).
    seed : optional
        NumPy random seed for reproducibility.
    """
    xs    = wind_field['x_centers']
    ys    = wind_field['y_centers']
    alpha = wind_field['alpha']
    V_ref = wind_field['V_ref']

    # Reconstruct wind velocity vectors (where wind blows TO)
    ux = -V_ref * np.cos(alpha)
    uy = -V_ref * np.sin(alpha)

    cell   = xs[1] - xs[0]
    domain = float(len(xs)) * cell
    rng    = np.random.default_rng(seed)

    ux += _smooth_noise_2d(ux.shape, domain, amplitude, n_modes,
                           seed=int(rng.integers(1_000_000)))
    uy += _smooth_noise_2d(uy.shape, domain, amplitude, n_modes,
                           seed=int(rng.integers(1_000_000)))

    return _velocity_to_field(ux, uy, xs, ys, V_ref_min=9.0, V_ref_max=25.0)
