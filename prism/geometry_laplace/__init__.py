"""
PRISM Geometry Laplace — Stage 4
=================================

Dynamics computed on the geometry manifold.
Reads: state_geometry.parquet, signal_pairwise.parquet
Writes: geometry_dynamics.parquet, lyapunov.parquet,
        dynamics.parquet, information_flow.parquet

Modules:
    geometry_dynamics.py        — Derivatives on state evolution
                                   velocity = dx/dt
                                   acceleration = d²x/dt²
                                   jerk = d³x/dt³
    lyapunov_engine.py          — Lyapunov exponents (chaos measurement)
    dynamics_runner.py          — RQA, attractor reconstruction
    information_flow_runner.py  — Transfer entropy networks, causal structure

This is where eigenvalue trajectories become physics.
The eigenvalues from state_geometry (Stage 2) describe the SHAPE at each index.
The derivatives here describe how that shape is CHANGING.

A system losing coherence shows up as eigenvalue velocity increasing
while effective dimension decreases. PRISM computes these derivatives.
ORTHON interprets what they mean.

Credit: Avery Rudder — "Laplace transform IS the state engine"
"""

__all__ = [
    'geometry_dynamics',
    'lyapunov_engine',
    'dynamics_runner',
    'information_flow_runner',
]
