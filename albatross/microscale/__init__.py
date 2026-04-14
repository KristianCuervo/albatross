"""
albatross.microscale — 3-D dynamic soaring optimisation.

Exports
-------
Solver      : single-solve dynamic soaring optimiser (max_vmg or min_vref mode)
Container   : single-run result container
Ensemble    : collection of Container results with plotting utilities
Factory     : batch sweep + parallel execution + warm-starting
sweep_min_vref  : convenience function — polar V_ref_min(θ) sweep
sweep_max_vmg   : convenience function — VMG polar sweep at fixed V_ref levels
"""

from .solver import Solver
from .container import Container
from .ensemble import Ensemble
from .factory import Factory, sweep_min_vref, sweep_max_vmg
