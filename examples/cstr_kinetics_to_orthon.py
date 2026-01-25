"""
CSTR Kinetics → ORTHON Pipeline Demo

Shows PRISM calculating pure numerical results and outputting to parquet
files that ORTHON can read for interpretation and display.

PRISM outputs:
- reaction_kinetics.parquet: Per-run kinetics (k, X, r)
- arrhenius.parquet: Temperature-dependent parameters (Ea, A)
- material_balance.parquet: Mass balance closure
- energy_balance.parquet: Heat duties
- transport.parquet: Re and flow regime
- metadata.parquet: Experiment context

ORTHON then:
- Reads parquet files
- Adds labels/classifications
- Generates reports and visualizations
"""

import polars as pl
from pathlib import Path
import tempfile


def main():
    # Import PRISM engines
    from prism.engines.physics.reaction_output import analyze_and_output

    print("=" * 70)
    print("PRISM → ORTHON Pipeline: CSTR Kinetics")
    print("=" * 70)

    # ==========================================================================
    # Experimental Data
    # ==========================================================================
    entity_ids = ['run_1', 'run_2', 'run_3', 'run_4', 'run_5']
    temperatures_K = [298.15, 308.15, 318.15, 328.15, 338.15]
    inlet_concentrations = [0.1] * 5  # mol/L
    outlet_concentrations = [0.0720, 0.0534, 0.0378, 0.0260, 0.0177]  # mol/L

    # Experiment constants
    reactor_volume_L = 2.5
    flow_rate_mL_min = 50
    heat_of_reaction = -75300  # J/mol (exothermic)
    pipe_diameter_m = 0.0127
    density_kg_m3 = 1020
    viscosity_Pa_s = 0.00102

    # ==========================================================================
    # Run PRISM Analysis
    # ==========================================================================
    print("\n[PRISM] Running calculations...")

    # Output to temp directory (in production, this would be ~/prism-mac/data/)
    output_dir = Path(tempfile.mkdtemp(prefix='prism_cstr_'))

    result = analyze_and_output(
        output_dir=str(output_dir),
        entity_ids=entity_ids,
        temperatures_K=temperatures_K,
        inlet_concentrations=inlet_concentrations,
        outlet_concentrations=outlet_concentrations,
        reactor_volume_L=reactor_volume_L,
        flow_rate_mL_min=flow_rate_mL_min,
        reaction_order=2,
        heat_of_reaction=heat_of_reaction,
        pipe_diameter_m=pipe_diameter_m,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
        metadata={
            'experiment': 'cstr_kinetics',
            'reaction': 'saponification',
            'course': 'CHE 344',
        }
    )

    print(f"\n[PRISM] Output written to: {output_dir}")
    print(f"\n[PRISM] Files created:")
    for name, path in result['files'].items():
        print(f"   • {name}: {Path(path).name}")

    # ==========================================================================
    # Show Parquet Contents (what ORTHON would read)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("[ORTHON] Reading parquet files...")
    print("=" * 70)

    # 1. Reaction kinetics
    print("\n📊 reaction_kinetics.parquet:")
    print("-" * 60)
    df = pl.read_parquet(result['files']['reaction_kinetics'])
    print(df.select(['entity_id', 'temperature_C', 'conversion', 'rate_constant',
                     'reaction_rate', 'heat_duty_W']))

    # 2. Arrhenius parameters
    print("\n📊 arrhenius.parquet:")
    print("-" * 60)
    df = pl.read_parquet(result['files']['arrhenius'])
    print(df)

    # 3. Material balance
    print("\n📊 material_balance.parquet:")
    print("-" * 60)
    df = pl.read_parquet(result['files']['material_balance'])
    print(df)

    # 4. Energy balance
    print("\n📊 energy_balance.parquet:")
    print("-" * 60)
    df = pl.read_parquet(result['files']['energy_balance'])
    print(df)

    # 5. Transport
    print("\n📊 transport.parquet:")
    print("-" * 60)
    df = pl.read_parquet(result['files']['transport'])
    print(df)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    s = result['summary']
    print(f"""
   PRISM Calculated (pure numbers):
   ├── Residence time: {s['residence_time_min']:.1f} min
   ├── Runs analyzed: {s['n_runs']}
   ├── Arrhenius parameters:
   │   ├── Ea = {s['activation_energy_kJ_mol']:.1f} kJ/mol
   │   ├── A = {s['pre_exponential']}
   │   └── R² = {s['r_squared']:.4f}
   └── Reynolds: {s['reynolds']:.0f}

   ORTHON would add:
   ├── Flow regime label: {'Laminar' if s['reynolds'] < 2100 else 'Turbulent'}
   ├── Data quality assessment
   ├── Comparison to literature values
   ├── Thesis-ready figures
   └── CHE 344 Lab Report template
""")

    print("=" * 70)
    print("Pipeline complete.")
    print(f"Parquet files at: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
