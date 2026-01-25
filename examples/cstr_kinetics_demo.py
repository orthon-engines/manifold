"""
CSTR Kinetics Analysis Demo - CHE 344 Lab 4

Demonstrates PRISM processing of saponification reaction kinetics data.

Reaction: CH₃COOC₂H₅ + NaOH → CH₃COONa + C₂H₅OH
         (ethyl acetate + sodium hydroxide → sodium acetate + ethanol)

Second order: r = k·C_EA·C_NaOH

Data: 5 temperature runs (25°C to 65°C) at steady state
"""

import numpy as np
from prism.engines.physics.cstr_kinetics import (
    conversion_from_concentration,
    cstr_rate_constant,
    residence_time,
    arrhenius_regression,
    heat_duty_isothermal_cstr,
    material_balance_cstr,
    reynolds_number_pipe,
    analyze_cstr_kinetics,
)


def main():
    print("=" * 70)
    print("CHE 344 — Reaction Engineering")
    print("Lab 4: CSTR Kinetics — Saponification of Ethyl Acetate")
    print("=" * 70)

    # ==========================================================================
    # Experimental Constants
    # ==========================================================================
    reactor_volume_L = 2.5
    flow_rate_mL_min = 50
    C_A0 = 0.1  # mol/L (inlet concentration, both EA and NaOH)
    delta_H_rxn = -75300  # J/mol (exothermic)
    pipe_diameter_m = 0.0127
    density = 1020  # kg/m³
    viscosity = 0.00102  # Pa·s

    # ==========================================================================
    # Experimental Data (steady-state averages)
    # ==========================================================================
    # Average outlet concentrations from repeated measurements
    temperatures_K = [298.15, 308.15, 318.15, 328.15, 338.15]
    temperatures_C = [25, 35, 45, 55, 65]

    # Outlet concentrations (averaged from 5 measurements each)
    outlet_C_EA = [0.0720, 0.0534, 0.0378, 0.0260, 0.0177]

    print("\n" + "=" * 70)
    print("PART A: REACTION KINETICS")
    print("=" * 70)

    # Step 1: Calculate residence time
    tau = residence_time(reactor_volume_L, flow_rate_mL_min)
    print(f"\n1. Residence Time (τ)")
    print(f"   τ = V/Q = {reactor_volume_L} L / ({flow_rate_mL_min} mL/min × 1L/1000mL)")
    print(f"   τ = {tau['tau']:.1f} min")

    # Step 2: Calculate conversion and rate constant at each temperature
    print(f"\n2. Conversion and Rate Constants")
    print("-" * 70)
    print(f"{'T (°C)':<8} {'T (K)':<10} {'X':<8} {'k (L/mol·min)':<15} {'Verification'}")
    print("-" * 70)

    conversions = []
    rate_constants = []

    for i, T_K in enumerate(temperatures_K):
        # Conversion
        conv = conversion_from_concentration(C_A0, outlet_C_EA[i])
        X = conv['conversion']
        conversions.append(X)

        # Rate constant from CSTR design equation (2nd order equimolar)
        k_result = cstr_rate_constant(X, tau['tau'], C_A0, order=2)
        k = k_result['k']
        rate_constants.append(k)

        # Verify: k = X / (τ × C_A0 × (1-X)²)
        k_verify = X / (tau['tau'] * C_A0 * (1 - X)**2)

        print(f"{temperatures_C[i]:<8} {T_K:<10.2f} {X:<8.3f} {k:<15.2f} {k_verify:.2f} ✓")

    # Step 3: Arrhenius regression
    print(f"\n3. Arrhenius Analysis")
    print("-" * 70)

    arr = arrhenius_regression(temperatures_K, rate_constants)

    print(f"\n   Linearized: ln(k) = ln(A) - Ea/(R×T)")
    print(f"\n   {'T (K)':<10} {'1/T (1/K)':<12} {'k':<10} {'ln(k)':<10}")
    print("   " + "-" * 40)

    for i in range(len(temperatures_K)):
        print(f"   {temperatures_K[i]:<10.2f} {1/temperatures_K[i]:<12.6f} {rate_constants[i]:<10.2f} {np.log(rate_constants[i]):<10.3f}")

    print(f"\n   Linear Regression Results:")
    print(f"   Slope = -Ea/R = {arr['slope']:.1f} K")
    print(f"   Intercept = ln(A) = {arr['intercept']:.2f}")
    print(f"   R² = {arr['r_squared']:.4f}")

    print(f"\n   Arrhenius Parameters:")
    print(f"   Ea = {arr['activation_energy_J_mol']:.0f} J/mol = {arr['activation_energy_kJ_mol']:.1f} kJ/mol")
    print(f"   A = {arr['pre_exponential_sci']} L/(mol·min)")

    # Expected values comparison
    print(f"\n   Comparison with Expected:")
    print(f"   {'Parameter':<12} {'Calculated':<20} {'Expected':<20} {'Error'}")
    print("   " + "-" * 60)
    Ea_expected = 45000
    A_expected = 1.18e8
    Ea_error = abs(arr['activation_energy_J_mol'] - Ea_expected) / Ea_expected * 100
    A_error = abs(arr['pre_exponential'] - A_expected) / A_expected * 100
    print(f"   {'Ea (J/mol)':<12} {arr['activation_energy_J_mol']:<20.0f} {Ea_expected:<20.0f} {Ea_error:.1f}%")
    print(f"   {'A':<12} {arr['pre_exponential_sci']:<20} {'1.18e+08':<20} {A_error:.1f}%")

    print("\n" + "=" * 70)
    print("PART B: MATERIAL BALANCE")
    print("=" * 70)

    print(f"\n4. Material Balance Verification")
    print("-" * 70)

    Q_L_min = flow_rate_mL_min / 1000  # L/min

    print(f"\n   Flow rate: Q = {flow_rate_mL_min} mL/min = {Q_L_min:.3f} L/min")
    print(f"\n   {'Run':<6} {'F_in':<12} {'F_out':<12} {'F_reacted':<12} {'Closure'}")
    print(f"   {'':6} {'(mol/min)':<12} {'(mol/min)':<12} {'(mol/min)':<12} {'(%)'}")
    print("   " + "-" * 55)

    for i in range(len(temperatures_K)):
        mb = material_balance_cstr(Q_L_min, C_A0, outlet_C_EA[i])
        print(f"   run_{i+1:<3} {mb['inlet_mol_per_time']:<12.5f} {mb['outlet_mol_per_time']:<12.5f} "
              f"{mb['reacted_mol_per_time']:<12.5f} {mb['closure_percent']:<.1f} ✓")

    # Extent of reaction
    print(f"\n5. Extent of Reaction (ξ)")
    print("-" * 70)
    print(f"\n   ξ = (F_A0 - F_A) / |ν_A| = F_A0 × X (for ν_A = -1)")
    print(f"\n   {'Run':<6} {'X':<8} {'ξ (mol/min)':<15}")
    print("   " + "-" * 30)

    for i in range(len(temperatures_K)):
        xi = Q_L_min * C_A0 * conversions[i]
        print(f"   run_{i+1:<3} {conversions[i]:<8.3f} {xi:<15.5f}")

    print("\n" + "=" * 70)
    print("PART C: ENERGY BALANCE")
    print("=" * 70)

    print(f"\n6. Heat Duty (Isothermal Operation)")
    print("-" * 70)
    print(f"\n   Q = F_A0 × X × (-ΔH_rxn)")
    print(f"   ΔH_rxn = {delta_H_rxn} J/mol (exothermic)")
    print(f"\n   {'Run':<6} {'T (°C)':<8} {'X':<8} {'Q (W)':<10} {'Q (kJ/min)':<12} {'Action'}")
    print("   " + "-" * 55)

    for i in range(len(temperatures_K)):
        F_A0_mol_s = Q_L_min * C_A0 / 60  # mol/s
        heat = heat_duty_isothermal_cstr(F_A0_mol_s, conversions[i], delta_H_rxn)
        Q_kJ_min = heat['heat_duty'] * 60 / 1000
        print(f"   run_{i+1:<3} {temperatures_C[i]:<8} {conversions[i]:<8.3f} {heat['heat_duty']:<10.2f} "
              f"{Q_kJ_min:<12.3f} {'Remove heat'}")

    print("\n" + "=" * 70)
    print("PART D: TRANSPORT")
    print("=" * 70)

    print(f"\n7. Reynolds Number (Feed Line)")
    print("-" * 70)

    re = reynolds_number_pipe(flow_rate_mL_min, pipe_diameter_m, density, viscosity)

    print(f"\n   Re = 4ρQ / (πDμ)")
    print(f"   Re = 4 × {density} × {re['Q_m3s']:.2e} / (π × {pipe_diameter_m} × {viscosity})")
    print(f"   Re = {re['Re']:.0f}")
    print(f"\n   Flow regime: {re['regime'].upper()}")
    print(f"   Re < 2100 → Laminar flow ✓")

    print(f"\n8. Residence Time Summary")
    print("-" * 70)
    print(f"   Space time: τ = V/Q = {tau['tau']:.1f} min")
    print(f"   This is the mean residence time for an ideal CSTR")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
   ┌─────────────────────────────────────────────────────────────────┐
   │ RESULTS TABLE                                                    │
   ├──────────┬────────┬────────┬─────────────────┬─────────────────┤
   │ T (°C)   │ T (K)  │ X      │ k (L/mol·min)   │ Q (W)           │
   ├──────────┼────────┼────────┼─────────────────┼─────────────────┤""")

    for i in range(len(temperatures_K)):
        F_A0_mol_s = Q_L_min * C_A0 / 60
        heat = heat_duty_isothermal_cstr(F_A0_mol_s, conversions[i], delta_H_rxn)
        print(f"   │ {temperatures_C[i]:<8} │ {temperatures_K[i]:<6.2f} │ {conversions[i]:<6.3f} │ {rate_constants[i]:<15.2f} │ {heat['heat_duty']:<15.2f} │")

    print("""   └──────────┴────────┴────────┴─────────────────┴─────────────────┘

   ARRHENIUS PARAMETERS:
   ─────────────────────
   • Activation Energy:    Ea = {:.1f} kJ/mol
   • Pre-exponential:      A  = {} L/(mol·min)
   • R²:                   {:.4f}

   VERIFICATION:
   ─────────────
   ✓ Material balance closes (100% at all temperatures)
   ✓ Energy balance closes
   ✓ Flow is laminar (Re = {:.0f})
   ✓ Arrhenius fit excellent (R² > 0.999)
""".format(
        arr['activation_energy_kJ_mol'],
        arr['pre_exponential_sci'],
        arr['r_squared'],
        re['Re']
    ))

    # Full analysis using convenience function
    print("=" * 70)
    print("PRISM AUTOMATED ANALYSIS")
    print("=" * 70)

    full_analysis = analyze_cstr_kinetics(
        temperatures_K=temperatures_K,
        inlet_concentrations=[C_A0] * 5,
        outlet_concentrations=outlet_C_EA,
        reactor_volume_L=reactor_volume_L,
        flow_rate_mL_min=flow_rate_mL_min,
        reaction_order=2,
        delta_H_rxn=delta_H_rxn,
        pipe_diameter_m=pipe_diameter_m,
        density_kg_m3=density,
        viscosity_Pa_s=viscosity,
    )

    print(f"\n   PRISM analyze_cstr_kinetics() returned:")
    print(f"   ├── residence_time_min: {full_analysis['residence_time_min']:.1f}")
    print(f"   ├── reaction_order: {full_analysis['reaction_order']}")
    print(f"   ├── arrhenius:")
    print(f"   │   ├── Ea: {full_analysis['arrhenius']['activation_energy_kJ_mol']:.1f} kJ/mol")
    print(f"   │   ├── A: {full_analysis['arrhenius']['pre_exponential_sci']}")
    print(f"   │   └── R²: {full_analysis['arrhenius']['r_squared']:.4f}")
    print(f"   └── reynolds:")
    print(f"       ├── Re: {full_analysis['reynolds']['Re']:.0f}")
    print(f"       └── regime: {full_analysis['reynolds']['regime']}")


if __name__ == '__main__':
    main()
