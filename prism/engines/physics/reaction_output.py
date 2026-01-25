"""
Reaction Engineering Output Schema

PRISM outputs pure numerical results to parquet.
ORTHON interprets and adds labels/classifications.

Output files for reaction discipline:
- reaction.parquet - per-run kinetics data
- arrhenius.parquet - temperature-dependent parameters

Schema follows PRISM pure calculation philosophy:
- All columns are numeric (except identifiers)
- No interpretation, no labels
- ORTHON handles classification and reporting
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def create_reaction_dataframe(
    entity_ids: List[str],
    temperatures_K: List[float],
    inlet_concentrations: List[float],
    outlet_concentrations: List[float],
    conversions: List[float],
    rate_constants: List[float],
    reaction_rates: List[float] = None,
    heat_duties: List[float] = None,
    residence_time: float = None,
    reaction_order: int = 2,
) -> pl.DataFrame:
    """
    Create reaction kinetics DataFrame.

    Parameters
    ----------
    entity_ids : list
        Run identifiers (e.g., ['run_1', 'run_2', ...])
    temperatures_K : list
        Temperatures [K]
    inlet_concentrations : list
        Inlet concentrations [mol/L]
    outlet_concentrations : list
        Outlet concentrations [mol/L]
    conversions : list
        Conversions (0-1)
    rate_constants : list
        Rate constants [L/(mol·time) or 1/time]
    reaction_rates : list, optional
        Reaction rates [mol/(L·time)]
    heat_duties : list, optional
        Heat duties [W]
    residence_time : float, optional
        Residence time [min]
    reaction_order : int
        Reaction order

    Returns
    -------
    pl.DataFrame
        Reaction kinetics data
    """
    n = len(entity_ids)

    data = {
        'entity_id': entity_ids,
        'temperature_K': temperatures_K,
        'temperature_C': [T - 273.15 for T in temperatures_K],
        'C_inlet': inlet_concentrations,
        'C_outlet': outlet_concentrations,
        'conversion': conversions,
        'rate_constant': rate_constants,
        'ln_rate_constant': [np.log(k) if k > 0 else np.nan for k in rate_constants],
        'inv_temperature': [1/T for T in temperatures_K],
    }

    if reaction_rates is not None:
        data['reaction_rate'] = reaction_rates

    if heat_duties is not None:
        data['heat_duty_W'] = heat_duties

    if residence_time is not None:
        data['residence_time'] = [residence_time] * n

    data['reaction_order'] = [reaction_order] * n

    return pl.DataFrame(data)


def create_arrhenius_dataframe(
    activation_energy_J_mol: float,
    pre_exponential: float,
    r_squared: float,
    n_points: int,
    slope: float,
    intercept: float,
    std_error: float = None,
) -> pl.DataFrame:
    """
    Create Arrhenius parameters DataFrame.

    Parameters
    ----------
    activation_energy_J_mol : float
        Activation energy [J/mol]
    pre_exponential : float
        Pre-exponential factor
    r_squared : float
        Coefficient of determination
    n_points : int
        Number of data points
    slope : float
        Arrhenius plot slope (-Ea/R)
    intercept : float
        Arrhenius plot intercept (ln(A))
    std_error : float, optional
        Standard error of slope

    Returns
    -------
    pl.DataFrame
        Arrhenius parameters
    """
    data = {
        'parameter': ['activation_energy_J_mol', 'activation_energy_kJ_mol',
                      'pre_exponential', 'ln_pre_exponential',
                      'r_squared', 'n_points', 'slope', 'intercept'],
        'value': [
            activation_energy_J_mol,
            activation_energy_J_mol / 1000,
            pre_exponential,
            np.log(pre_exponential) if pre_exponential > 0 else np.nan,
            r_squared,
            float(n_points),
            slope,
            intercept,
        ]
    }

    if std_error is not None:
        data['parameter'].append('std_error_slope')
        data['value'].append(std_error)

    return pl.DataFrame(data)


def create_material_balance_dataframe(
    entity_ids: List[str],
    inlet_molar_flow: List[float],
    outlet_molar_flow: List[float],
    reacted_molar_flow: List[float],
    product_molar_flow: List[float] = None,
) -> pl.DataFrame:
    """
    Create material balance DataFrame.

    Parameters
    ----------
    entity_ids : list
        Run identifiers
    inlet_molar_flow : list
        Inlet molar flow [mol/time]
    outlet_molar_flow : list
        Outlet molar flow [mol/time]
    reacted_molar_flow : list
        Reacted molar flow [mol/time]
    product_molar_flow : list, optional
        Product molar flow [mol/time]

    Returns
    -------
    pl.DataFrame
        Material balance data
    """
    data = {
        'entity_id': entity_ids,
        'inlet_molar_flow': inlet_molar_flow,
        'outlet_molar_flow': outlet_molar_flow,
        'reacted_molar_flow': reacted_molar_flow,
        'balance_error': [i - o - r for i, o, r in
                         zip(inlet_molar_flow, outlet_molar_flow, reacted_molar_flow)],
    }

    # Closure percentage
    closures = []
    for i, o, r in zip(inlet_molar_flow, outlet_molar_flow, reacted_molar_flow):
        if i > 0:
            closures.append((o + r) / i * 100)
        else:
            closures.append(100.0)
    data['closure_percent'] = closures

    if product_molar_flow is not None:
        data['product_molar_flow'] = product_molar_flow

    return pl.DataFrame(data)


def create_energy_balance_dataframe(
    entity_ids: List[str],
    heat_duties_W: List[float],
    temperatures_K: List[float],
    conversions: List[float],
    heat_of_reaction: float,
) -> pl.DataFrame:
    """
    Create energy balance DataFrame.

    Parameters
    ----------
    entity_ids : list
        Run identifiers
    heat_duties_W : list
        Heat duties [W]
    temperatures_K : list
        Temperatures [K]
    conversions : list
        Conversions (0-1)
    heat_of_reaction : float
        Heat of reaction [J/mol]

    Returns
    -------
    pl.DataFrame
        Energy balance data
    """
    data = {
        'entity_id': entity_ids,
        'temperature_K': temperatures_K,
        'conversion': conversions,
        'heat_duty_W': heat_duties_W,
        'heat_duty_kJ_min': [Q * 60 / 1000 for Q in heat_duties_W],
        'heat_of_reaction_J_mol': [heat_of_reaction] * len(entity_ids),
        'is_exothermic': [heat_of_reaction < 0] * len(entity_ids),
    }

    return pl.DataFrame(data)


def create_transport_dataframe(
    entity_ids: List[str],
    reynolds: List[float],
    flow_rate_m3s: List[float] = None,
    velocity: List[float] = None,
) -> pl.DataFrame:
    """
    Create transport properties DataFrame.

    Parameters
    ----------
    entity_ids : list
        Run identifiers
    reynolds : list
        Reynolds numbers
    flow_rate_m3s : list, optional
        Volumetric flow rates [m³/s]
    velocity : list, optional
        Velocities [m/s]

    Returns
    -------
    pl.DataFrame
        Transport properties
    """
    data = {
        'entity_id': entity_ids,
        'reynolds': reynolds,
        'is_laminar': [Re < 2100 for Re in reynolds],
        'is_turbulent': [Re > 4000 for Re in reynolds],
    }

    if flow_rate_m3s is not None:
        data['flow_rate_m3s'] = flow_rate_m3s

    if velocity is not None:
        data['velocity'] = velocity

    return pl.DataFrame(data)


def write_reaction_outputs(
    output_dir: str,
    reaction_df: pl.DataFrame,
    arrhenius_df: pl.DataFrame = None,
    material_balance_df: pl.DataFrame = None,
    energy_balance_df: pl.DataFrame = None,
    transport_df: pl.DataFrame = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, str]:
    """
    Write all reaction discipline outputs to parquet.

    Parameters
    ----------
    output_dir : str
        Output directory
    reaction_df : pl.DataFrame
        Main reaction kinetics data
    arrhenius_df : pl.DataFrame, optional
        Arrhenius parameters
    material_balance_df : pl.DataFrame, optional
        Material balance data
    energy_balance_df : pl.DataFrame, optional
        Energy balance data
    transport_df : pl.DataFrame, optional
        Transport properties
    metadata : dict, optional
        Experiment metadata

    Returns
    -------
    dict
        Paths to written files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Main reaction data
    reaction_path = output_dir / 'reaction_kinetics.parquet'
    reaction_df.write_parquet(reaction_path)
    files['reaction_kinetics'] = str(reaction_path)

    # Arrhenius parameters
    if arrhenius_df is not None:
        arr_path = output_dir / 'arrhenius.parquet'
        arrhenius_df.write_parquet(arr_path)
        files['arrhenius'] = str(arr_path)

    # Material balance
    if material_balance_df is not None:
        mb_path = output_dir / 'material_balance.parquet'
        material_balance_df.write_parquet(mb_path)
        files['material_balance'] = str(mb_path)

    # Energy balance
    if energy_balance_df is not None:
        eb_path = output_dir / 'energy_balance.parquet'
        energy_balance_df.write_parquet(eb_path)
        files['energy_balance'] = str(eb_path)

    # Transport
    if transport_df is not None:
        tr_path = output_dir / 'transport.parquet'
        transport_df.write_parquet(tr_path)
        files['transport'] = str(tr_path)

    # Metadata
    if metadata is not None:
        meta_df = pl.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'discipline': 'reaction',
            **{k: str(v) if not isinstance(v, (int, float)) else v
               for k, v in metadata.items()}
        }])
        meta_path = output_dir / 'metadata.parquet'
        meta_df.write_parquet(meta_path)
        files['metadata'] = str(meta_path)

    return files


# =============================================================================
# Convenience function for complete analysis
# =============================================================================

def analyze_and_output(
    output_dir: str,
    entity_ids: List[str],
    temperatures_K: List[float],
    inlet_concentrations: List[float],
    outlet_concentrations: List[float],
    reactor_volume_L: float,
    flow_rate_mL_min: float,
    reaction_order: int = 2,
    heat_of_reaction: float = None,
    pipe_diameter_m: float = None,
    density_kg_m3: float = None,
    viscosity_Pa_s: float = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Complete analysis and output for ORTHON.

    Runs all calculations and writes parquet files.

    Parameters
    ----------
    output_dir : str
        Where to write parquet files
    entity_ids : list
        Run identifiers
    temperatures_K : list
        Temperatures [K]
    inlet_concentrations : list
        Inlet concentrations [mol/L]
    outlet_concentrations : list
        Outlet concentrations [mol/L]
    reactor_volume_L : float
        Reactor volume [L]
    flow_rate_mL_min : float
        Flow rate [mL/min]
    reaction_order : int
        Reaction order (1 or 2)
    heat_of_reaction : float, optional
        Heat of reaction [J/mol]
    pipe_diameter_m : float, optional
        Feed pipe diameter [m]
    density_kg_m3 : float, optional
        Fluid density [kg/m³]
    viscosity_Pa_s : float, optional
        Fluid viscosity [Pa·s]
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    dict
        Summary of results and file paths
    """
    from prism.engines.physics.cstr_kinetics import (
        conversion_from_concentration,
        cstr_rate_constant,
        residence_time,
        arrhenius_regression,
        heat_duty_isothermal_cstr,
        material_balance_cstr,
        reynolds_number_pipe,
    )

    n = len(entity_ids)

    # Residence time
    tau_result = residence_time(reactor_volume_L, flow_rate_mL_min)
    tau = tau_result['tau']

    # Conversions and rate constants
    conversions = []
    rate_constants = []
    reaction_rates = []

    for i in range(n):
        conv = conversion_from_concentration(inlet_concentrations[i], outlet_concentrations[i])
        conversions.append(conv['conversion'])

        k_result = cstr_rate_constant(conv['conversion'], tau, inlet_concentrations[i],
                                       order=reaction_order)
        rate_constants.append(k_result['k'])
        reaction_rates.append(k_result['reaction_rate'])

    # Arrhenius regression
    arr_result = arrhenius_regression(temperatures_K, rate_constants)

    # Heat duties
    heat_duties = None
    if heat_of_reaction is not None:
        heat_duties = []
        Q_L_min = flow_rate_mL_min / 1000
        for i in range(n):
            F_A0_mol_s = Q_L_min * inlet_concentrations[i] / 60
            heat = heat_duty_isothermal_cstr(F_A0_mol_s, conversions[i], heat_of_reaction)
            heat_duties.append(heat['heat_duty'])

    # Material balance
    Q_L_min = flow_rate_mL_min / 1000
    inlet_flows = [Q_L_min * c for c in inlet_concentrations]
    outlet_flows = [Q_L_min * c for c in outlet_concentrations]
    reacted_flows = [i - o for i, o in zip(inlet_flows, outlet_flows)]

    # Reynolds number
    reynolds_nums = None
    if all(x is not None for x in [pipe_diameter_m, density_kg_m3, viscosity_Pa_s]):
        re_result = reynolds_number_pipe(flow_rate_mL_min, pipe_diameter_m,
                                          density_kg_m3, viscosity_Pa_s)
        reynolds_nums = [re_result['Re']] * n

    # Create DataFrames
    reaction_df = create_reaction_dataframe(
        entity_ids, temperatures_K, inlet_concentrations, outlet_concentrations,
        conversions, rate_constants, reaction_rates, heat_duties, tau, reaction_order
    )

    arrhenius_df = create_arrhenius_dataframe(
        arr_result['activation_energy_J_mol'],
        arr_result['pre_exponential'],
        arr_result['r_squared'],
        n,
        arr_result['slope'],
        arr_result['intercept'],
        arr_result.get('std_err_slope')
    )

    material_balance_df = create_material_balance_dataframe(
        entity_ids, inlet_flows, outlet_flows, reacted_flows
    )

    energy_balance_df = None
    if heat_of_reaction is not None:
        energy_balance_df = create_energy_balance_dataframe(
            entity_ids, heat_duties, temperatures_K, conversions, heat_of_reaction
        )

    transport_df = None
    if reynolds_nums is not None:
        transport_df = create_transport_dataframe(entity_ids, reynolds_nums)

    # Write outputs
    all_metadata = {
        'reactor_volume_L': reactor_volume_L,
        'flow_rate_mL_min': flow_rate_mL_min,
        'residence_time_min': tau,
        'reaction_order': reaction_order,
        **(metadata or {})
    }

    files = write_reaction_outputs(
        output_dir, reaction_df, arrhenius_df, material_balance_df,
        energy_balance_df, transport_df, all_metadata
    )

    return {
        'files': files,
        'summary': {
            'residence_time_min': tau,
            'n_runs': n,
            'activation_energy_kJ_mol': arr_result['activation_energy_kJ_mol'],
            'pre_exponential': arr_result['pre_exponential_sci'],
            'r_squared': arr_result['r_squared'],
            'reynolds': reynolds_nums[0] if reynolds_nums else None,
        }
    }
