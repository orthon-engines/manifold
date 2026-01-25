"""
PRISM Real Physics Engines

All engines compute REAL physics equations with proper units.

Philosophy:
- When physical constants are known → compute absolute values in Joules, etc.
- When physical constants are unknown → compute specific (per-unit) values
- Always honest about what was computed (is_specific flag, units, equation)

Engines:
1. kinetic_energy: T = ½mv²  [J or J/kg]
2. potential_energy: V = ½kx², V = mgh  [J or specific]
3. hamiltonian: H = T + V  [J]
4. lagrangian: L = T - V  [J]
5. momentum: p = mv, L = r × p  [kg·m/s, kg·m²/s]
6. gibbs_free_energy: G = H - TS  [J or J/mol]
7. work_energy: W = ∫F·dx, P = F·v  [J, W]

No fake physics. No assumed constants. No cover bands.
"""

from prism.engines.physics.kinetic_energy import (
    compute_kinetic_energy,
    compute_kinetic_energy_rotational,
    compute as compute_kinetic,
)

from prism.engines.physics.potential_energy import (
    compute_potential_energy_harmonic,
    compute_potential_energy_gravitational,
    estimate_spring_constant,
    compute as compute_potential,
)

from prism.engines.physics.hamiltonian import (
    compute_hamiltonian,
    compute_hamiltons_equations,
    compute as compute_hamilton,
)

from prism.engines.physics.lagrangian import (
    compute_lagrangian,
    compute_action,
    check_euler_lagrange,
    compute as compute_lagrange,
)

from prism.engines.physics.momentum import (
    compute_linear_momentum,
    compute_angular_momentum,
    check_momentum_conservation,
    compute_impulse,
    compute as compute_momentum,
)

from prism.engines.physics.gibbs_free_energy import (
    compute_gibbs_free_energy,
    compute_gibbs_ideal_gas,
    compute_gibbs_change,
    compute_chemical_potential,
    compute as compute_gibbs,
)

from prism.engines.physics.work_energy import (
    compute_work,
    compute_power,
    verify_work_energy_theorem,
    compute_conservative_force_test,
    compute_mechanical_energy,
    compute as compute_work_energy,
)


__all__ = [
    # Kinetic Energy
    'compute_kinetic_energy',
    'compute_kinetic_energy_rotational',
    'compute_kinetic',
    # Potential Energy
    'compute_potential_energy_harmonic',
    'compute_potential_energy_gravitational',
    'estimate_spring_constant',
    'compute_potential',
    # Hamiltonian
    'compute_hamiltonian',
    'compute_hamiltons_equations',
    'compute_hamilton',
    # Lagrangian
    'compute_lagrangian',
    'compute_action',
    'check_euler_lagrange',
    'compute_lagrange',
    # Momentum
    'compute_linear_momentum',
    'compute_angular_momentum',
    'check_momentum_conservation',
    'compute_impulse',
    'compute_momentum',
    # Gibbs Free Energy
    'compute_gibbs_free_energy',
    'compute_gibbs_ideal_gas',
    'compute_gibbs_change',
    'compute_chemical_potential',
    'compute_gibbs',
    # Work-Energy
    'compute_work',
    'compute_power',
    'verify_work_energy_theorem',
    'compute_conservative_force_test',
    'compute_mechanical_energy',
    'compute_work_energy',
]
