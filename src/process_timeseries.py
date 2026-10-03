#!/usr/bin/env python3
"""
Build basal shear stress and sliding velocity timeseries at each stake.

Workflow:
1. Read Elmer/Ice outputs at DEM dates and average them around each stake.
2. Read in-situ observations (surface elevation and velocity).
3. Fit empirical relationships tau_b ~ H and u_def ~ H^4 at DEM dates.
4. Apply these relationships to all observation dates.
5. Compute basal sliding velocity u_bed = u_surf - u_def.
6. Save the final timeseries as CSV.

Output: {glacier}_all_data_{stake}.csv with columns:
    date, u_bed_elmer, u_surf_elmer, tau_d_elmer, tau_b_elmer,
    sigma_elmer, u_def_elmer, slope, ...,
    altitude, velocity, thickness, obs_tau_b, obs_u_def, obs_u_bed
"""
from utils import GLACIERS, geom_data_dir, proc_data_dir
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import re

script_dir = Path(__file__).resolve().parent

# ============================================================================
# ELMER/ICE OUTPUTS
# ============================================================================

def read_elmer_data_file(glacier_name, year, m, C, Arg_simu=None):
    """
    Read the Elmer/Ice output file for a given glacier and year.

    If Arg_simu is given, read the corresponding sensitivity experiment
    for Argentière (data/uncertainties/{Arg_simu}/).
    """
    if Arg_simu is not None:
        elmer_file = script_dir / '..' / 'data' / 'uncertainties' / f'{Arg_simu}' / f"Arg_{Arg_simu}_{year}.csv"
    else:
        elmer_file = script_dir / '..' / 'data' / 'elmer_raw' / f'mw{m:.0f}' / f'{glacier_name}_{year}.csv'

    if not elmer_file.exists():
        print(f"[WARNING] missing Elmer file: {glacier_name} {year}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(elmer_file)
    except Exception as e:
        print(f"[WARNING] error reading {elmer_file}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"[WARNING] empty file: {glacier_name} {year}")
        return pd.DataFrame()

    return df


# ============================================================================
# STRESS CALCULATIONS
# ============================================================================

def calc_tau_b(u_bed, C, m=3):
    """Basal shear stress from the Weertman friction law used in Elmer/Ice."""
    return C * (u_bed ** (1/m))


def calc_tau_d(thickness, xgrad, ygrad):
    """Driving stress tau_d = rho * g * H * sin(alpha), in MPa."""
    angle = np.arctan(np.sqrt(xgrad**2 + ygrad**2))
    return 1e-6 * 917 * 9.81 * thickness * np.sin(angle)


# ============================================================================
# SPATIAL AVERAGING
# ============================================================================

def average_in_radius(glacier_name, stake_name, df, x0, y0, radius, m, C, Hmin=20, spatial_as=False):
    """
    Average Elmer/Ice outputs within a circle around a stake.

    Parameters
    ----------
    glacier_name, stake_name : str
        Glacier and stake identifiers.
    df : pd.DataFrame
        Elmer/Ice nodal outputs for one date.
    x0, y0 : float
        Stake coordinates [m].
    radius : float
        Averaging radius [m].
    m, C : float
        Exponent and coefficient of the Weertman law used in Elmer/Ice.
    Hmin : float
        Minimum ice thickness [m].
    spatial_as : bool
        If True, use the spatially variable friction coefficient (column 'cw').

    Returns
    -------
    dict or None
        Averaged variables, or None if no node lies within the circle.
    """
    df['distance'] = np.sqrt((df['xcoord'] - x0)**2 + (df['ycoord'] - y0)**2)

    mask = (df['distance'] <= radius) & (df['thicksurf'] >= Hmin)
    neighbourhood = df[mask].copy()

    if len(neighbourhood) == 0:
        return None

    # Close neighbourhood used for the local slope
    mask_close = (df['distance'] <= 50) & (df['thicksurf'] >= Hmin)
    neighbourhood_close = df[mask_close].copy()

    # Mean flow direction
    slopex = -neighbourhood['xgrad'].mean(skipna=True)
    slopey = -neighbourhood['ygrad'].mean(skipna=True)

    neighbourhood['zgrad_dirmean'] = (slopex * neighbourhood['xgrad'] +
                                      slopey * neighbourhood['ygrad'])
    slopez = neighbourhood['zgrad_dirmean'].mean(skipna=True)
    normslope = np.sqrt(slopex**2 + slopey**2 + slopez**2)

    if normslope == 0:
        normslope = 1e-9

    # Projection of the bed normal onto the mean flow direction
    neighbourhood['projvector'] = (
        (slopex / normslope) * neighbourhood['normalbed1'] +
        (slopey / normslope) * neighbourhood['normalbed2'] +
        (slopez / normslope) * neighbourhood['normalbed3']
    )

    # Area-weighted averages
    total_area = neighbourhood['nodearea'].sum(skipna=True)

    if total_area == 0:
        return None

    vel_h_bed = neighbourhood['vel_h_bed'].mean(skipna=True)
    vel_h_surf = neighbourhood['vel_h_surf'].mean(skipna=True)
    thick_elmer = neighbourhood['thicksurf'].mean(skipna=True)

    sigma = (neighbourhood['normalstress'] * neighbourhood['projvector'] *
             neighbourhood['nodearea']).sum(skipna=True) / total_area

    # Driving stress
    neighbourhood['tau_d'] = calc_tau_d(
        neighbourhood['thicksurf'],
        neighbourhood['xgrad'],
        neighbourhood['ygrad'],
    )
    tau_d = (neighbourhood['tau_d'] * neighbourhood['nodearea']).sum(skipna=True) / total_area

    # Basal shear stress
    if spatial_as:
        neighbourhood['tau_b'] = calc_tau_b(neighbourhood['vel_h_bed'], neighbourhood['cw'], m)
    else:
        neighbourhood['tau_b'] = calc_tau_b(neighbourhood['vel_h_bed'], C, m)

    tau_b = (neighbourhood['tau_b'] * neighbourhood['nodearea']).sum(skipna=True) / total_area

    # Surface slopes
    slope = np.sqrt(
        neighbourhood_close['xgrad']**2 + neighbourhood_close['ygrad']**2
    ).mean(skipna=True)

    averaged_slope = np.sqrt(
        neighbourhood['xgrad']**2 + neighbourhood['ygrad']**2
    ).mean(skipna=True)

    df_slopes = pd.read_csv(geom_data_dir / 'slopes/mean_slopes.csv', sep=",")
    row = df_slopes[(df_slopes['glacier'] == glacier_name) & (df_slopes['stake'] == stake_name)]
    slope_rad = row['mean_slope_rad_full'].values[0]
    slope_dem = np.tan(slope_rad)

    return {
        'thick_elmer': thick_elmer,
        'u_bed_elmer': vel_h_bed,
        'u_surf_elmer': vel_h_surf,
        'tau_d_elmer': tau_d,
        'tau_b_elmer': tau_b,
        'sigma_elmer': sigma,
        'u_def_elmer': vel_h_surf - vel_h_bed,
        'slope': slope,
        'averaged_slope': averaged_slope,
        'slope_dem': slope_dem,
        'gradxmean': slopex,
        'gradymean': slopey,
        'gradzmean': slopez,
        'normalstress': neighbourhood['normalstress'].mean(skipna=True),
        'projvector': neighbourhood['projvector'].mean(skipna=True)
    }


def process_elmer_timeseries(glacier_name, stake_name, years_DEM, x0, y0, radius, m, C,
                             Hmin=20, Arg_simu=None, spatial_as=False):
    """
    Build the Elmer/Ice timeseries at a stake over all DEM dates.

    Parameters
    ----------
    years_DEM : list
        Years with an available surface DEM.
    x0, y0 : float
        Stake coordinates [m].
    radius : float
        Averaging radius [m].
    m, C : float
        Exponent and coefficient of the Weertman law used in Elmer/Ice.
    Hmin : float
        Minimum ice thickness [m].
    Arg_simu : str, optional
        Name of an Argentière sensitivity experiment.
    spatial_as : bool
        If True, use the spatially variable friction coefficient.

    Returns
    -------
    pd.DataFrame
        One row per DEM date.
    """
    records = []

    for year in years_DEM:
        df_elmer = read_elmer_data_file(glacier_name, year, m, C, Arg_simu=Arg_simu)

        if df_elmer.empty:
            continue

        result = average_in_radius(glacier_name, stake_name, df_elmer, x0, y0, radius,
                                   m, C, Hmin, spatial_as=spatial_as)

        if result is None:
            continue

        result['date'] = year
        result['sigma_plus_tau_b'] = result['sigma_elmer'] + result['tau_b_elmer']

        records.append(result)

    if len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)

    return df


# ============================================================================
# OBSERVATIONS
# ============================================================================

def read_observations(glacier_name, stake_name):
    """
    Read in-situ observations of surface elevation and velocity at a stake.

    Returns
    -------
    dict
        Keys 'altitude' and 'velocity', each a DataFrame with a 'date' column.
    """
    obs = {}

    alt_file = script_dir / '..' / 'data' / 'obs_raw' / f'{glacier_name}_alt_{stake_name}.csv'
    if alt_file.exists():
        df = pd.read_csv(alt_file)
        if 'year' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'year': 'date'})
        obs['altitude'] = df

    vel_file = script_dir / '..' / 'data' / 'obs_raw' / f'{glacier_name}_vel_{stake_name}.csv'
    if vel_file.exists():
        df = pd.read_csv(vel_file)
        if 'year' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'year': 'date'})
        obs['velocity'] = df

    return obs


def interp_zdem(mnt_bed, xx, yy):
    """Interpolate the bedrock elevation at (xx, yy)."""
    x = mnt_bed.iloc[:, 0].values
    y = mnt_bed.iloc[:, 1].values
    z = mnt_bed.iloc[:, 2].values

    zi = griddata((x, y), z, (xx, yy), method='linear')

    return round(float(zi), 2)


# ============================================================================
# EMPIRICAL RELATIONSHIPS
# ============================================================================

def fit_empirical_relation(x_obs, y_elmer, degree=1):
    """
    Fit a polynomial relationship between an observed variable and an
    Elmer/Ice output (e.g. tau_b as a function of thickness).

    Returns
    -------
    np.ndarray or None
        Polynomial coefficients, or None if there are not enough points.
    """
    mask = ~np.isnan(x_obs) & ~np.isnan(y_elmer)

    if mask.sum() < degree + 1:
        return None

    coeffs = np.polyfit(x_obs[mask], y_elmer[mask], degree)

    return coeffs


def apply_empirical_relation(x_continuous, coeffs):
    """Apply a fitted polynomial relationship."""
    if coeffs is None:
        return np.full_like(x_continuous, np.nan)

    poly = np.poly1d(coeffs)
    return poly(x_continuous)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_glacier_stake(glacier_name, stake_name, config, m, C,
                          Arg_simu=None, output_file=None, spatial_as=False):
    """
    Build and save the full timeseries for one stake.

    Parameters
    ----------
    glacier_name, stake_name : str
        Glacier and stake identifiers.
    config : dict
        Glacier entry of the GLACIERS dictionary.
    m, C : float
        Exponent and coefficient of the Weertman law used in Elmer/Ice.
    Arg_simu : str, optional
        Name of an Argentière sensitivity experiment.
    output_file : Path, optional
        Output path. Defaults to processed_timeseries/mw{1/m}/{glacier}_all_data_{stake}.csv.
    spatial_as : bool
        If True, use the spatially variable friction coefficient.

    Returns
    -------
    pd.DataFrame or None
    """
    print(f"\nProcessing {glacier_name} - {stake_name}")

    years_DEM = config['years_DEM']
    x0, y0 = config['xy_coords'][stake_name]
    Hmin = 20
    radius = config['avg_dist'][stake_name]

    print(f"  Coordinates: ({x0}, {y0}), averaging radius: {radius} m, C={C}, m={m}")

    # 1. Elmer/Ice outputs
    df_elmer = process_elmer_timeseries(
        glacier_name, stake_name, years_DEM, x0, y0, radius, m, C, Hmin,
        Arg_simu=Arg_simu, spatial_as=spatial_as
    )
    print(f"  {len(df_elmer)} Elmer/Ice dates")

    # 2. Observations
    obs = read_observations(glacier_name, stake_name)

    df_altitude = obs.get('altitude', pd.DataFrame())
    df_velocity = obs.get('velocity', pd.DataFrame())

    print(f"  {len(df_altitude)} elevation and {len(df_velocity)} velocity observations")

    # 3. Merge Elmer/Ice outputs and observations at DEM dates
    if not df_velocity.empty:
        df_obs = df_velocity.copy()
        col = [c for c in df_obs.columns if c != 'date'][0]
        df_obs = df_obs.rename(columns={col: 'velocity'})
    else:
        df_obs = pd.DataFrame()

    if not df_altitude.empty:
        df_alt = df_altitude.copy()
        col = [c for c in df_alt.columns if c != 'date'][0]
        df_alt = df_alt.rename(columns={col: 'altitude'})

        df_obs = pd.merge(df_obs, df_alt, on='date', how='outer')

    # Ice thickness from observed surface elevation and bedrock DEM
    if 'altitude' in df_obs.columns:
        mnt_bed_path = geom_data_dir / 'bedrocks' / f'DEM_bedrock_{glacier_name}.dat'
        mnt_bed = pd.read_csv(mnt_bed_path, delimiter=r'\s+', header=None)
        zbedrock = interp_zdem(mnt_bed, x0, y0)
        df_obs['thickness'] = df_obs['altitude'] - zbedrock

    df_elmer = df_elmer.sort_values("date")
    df_obs = df_obs.sort_values("date")
    df_elmer["date"] = df_elmer["date"].astype(float)
    df_obs["date"] = df_obs["date"].astype(float)

    # Match each DEM date with the nearest observation date (within 4 years)
    df_merged_dem = pd.merge_asof(
        df_elmer,
        df_obs,
        on="date",
        direction="nearest",
        tolerance=4
    )

    # Fill missing values column by column with the nearest available observation
    cols_obs = ['velocity', 'altitude', 'thickness']
    for col in cols_obs:
        if col not in df_obs.columns:
            continue
        df_temp = df_obs[['date', col]].dropna().sort_values('date')
        df_merged_dem = pd.merge_asof(
            df_merged_dem,
            df_temp.rename(columns={col: f'{col}_fill'}),
            on='date',
            direction='nearest',
            tolerance=4
        )
        df_merged_dem[col] = df_merged_dem[col].fillna(df_merged_dem[f'{col}_fill'])
        df_merged_dem.drop(columns=f'{col}_fill', inplace=True)

    cols = ['date'] + [col for col in df_merged_dem.columns if col != 'date']
    df_merged_dem = df_merged_dem[cols]

    print(f"  {len(df_merged_dem)} dates with both Elmer/Ice outputs and observations")

    if len(df_merged_dem) < 3:
        print("  Not enough points to fit the empirical relationships")
        return None

    # 4. Fit empirical relationships
    if glacier_name == "GB":
        # Glacier Blanc: relationships also depend on surface slope
        # tau_b ~ H * slope
        HS = (df_merged_dem['thickness'].values) * (np.arctan(df_merged_dem['slope']).values)
        coeffs_tau = fit_empirical_relation(
            HS, df_merged_dem['tau_b_elmer'].values, degree=1)

        # u_def ~ H^4 * slope^3
        H4S3 = (df_merged_dem['thickness'].values ** 4) * (np.arctan(df_merged_dem['slope']).values ** 3)
        coeffs_udef = fit_empirical_relation(
            H4S3, df_merged_dem['u_def_elmer'].values, degree=1)

    else:
        # tau_b ~ H
        coeffs_tau = fit_empirical_relation(
            df_merged_dem['thickness'].values,
            df_merged_dem['tau_b_elmer'].values, degree=1)

        # u_def ~ H^4
        H4 = df_merged_dem['thickness'].values ** 4
        coeffs_udef = fit_empirical_relation(
            H4, df_merged_dem['u_def_elmer'].values, degree=1)

        if coeffs_tau is not None:
            print(f"  tau_b = {coeffs_tau[0]:.2e} * H + {coeffs_tau[1]:.2e}")
        if coeffs_udef is not None:
            print(f"  u_def = {coeffs_udef[0]:.2e} * H^4 + {coeffs_udef[1]:.2e}")

    # 5. Apply the relationships to all observation dates
    if 'thickness' in df_obs.columns:

        if glacier_name == "GB":
            df_slope = df_merged_dem[['date', 'slope']].drop_duplicates('date')
            df_obs['slope'] = np.interp(df_obs['date'].values, df_slope['date'].values, df_slope['slope'].values)

            df_obs['obs_tau_b'] = apply_empirical_relation(
                (df_obs['thickness'].values) * (np.arctan(df_obs['slope'])), coeffs_tau
            )

            df_obs['obs_u_def'] = apply_empirical_relation(
                (df_obs['thickness'].values) ** 4 * (np.arctan(df_obs['slope'].values)) ** 3, coeffs_udef
            )

            df_obs = df_obs.drop(columns=['slope'])  # avoid duplicate slope columns when merging

        elif glacier_name == "StSo":
            # Saint-Sorlin: linear interpolation in time of the Elmer/Ice outputs
            df_obs['obs_tau_b'] = np.interp(
                df_obs['date'].values,
                df_merged_dem['date'].values,
                df_merged_dem['tau_b_elmer'].values
            )

            df_obs['obs_u_def'] = np.interp(
                df_obs['date'].values,
                df_merged_dem['date'].values,
                df_merged_dem['u_def_elmer'].values
            )

            # Thickness-based estimates, kept for comparison
            df_obs['obs_tau_b_reglin'] = apply_empirical_relation(
                df_obs['thickness'].values, coeffs_tau
            )

            df_obs['obs_u_def_reglin'] = apply_empirical_relation(
                df_obs['thickness'].values ** 4, coeffs_udef
            )

        else:
            df_obs['obs_tau_b'] = apply_empirical_relation(
                df_obs['thickness'].values, coeffs_tau
            )

            df_obs['obs_u_def'] = apply_empirical_relation(
                df_obs['thickness'].values ** 4, coeffs_udef
            )

        # Basal sliding velocity
        if 'velocity' in df_obs.columns:
            df_obs['obs_u_bed'] = df_obs['velocity'] - df_obs['obs_u_def']

    # 6. Build the final dataset
    # Add the reconstructed variables at DEM dates
    cols_calculated = [c for c in ['obs_tau_b', 'obs_u_def', 'obs_u_bed'] if c in df_obs.columns]

    for col in cols_calculated:
        df_temp = df_obs[['date', col]].dropna().sort_values('date')
        df_merged_dem = pd.merge_asof(
            df_merged_dem,
            df_temp.rename(columns={col: f'{col}_fill'}),
            on='date',
            direction='nearest',
            tolerance=4
        )
        df_merged_dem[col] = df_merged_dem.get(col, np.nan)
        df_merged_dem[col] = df_merged_dem[col].fillna(df_merged_dem[f'{col}_fill'])
        df_merged_dem.drop(columns=f'{col}_fill', inplace=True)

    # Add observation-only dates
    df_final = pd.merge(
        df_merged_dem,
        df_obs,
        on='date',
        how='outer',
        suffixes=('', '_obs')
    )

    obs_cols = [col for col in df_final.columns if col.endswith('_obs')]
    for col in obs_cols:
        original = col.replace('_obs', '')
        df_final[original] = df_final[original].fillna(df_final[col])
        df_final.drop(columns=col, inplace=True)

    df_final = df_final.sort_values('date').reset_index(drop=True)

    # 7. Save
    output_dir = Path(script_dir / '..' / 'data' / 'processed_timeseries' / f'mw{1/m:.3f}')
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        output_file = output_dir / f'{glacier_name}_all_data_{stake_name}.csv'
    df_final.to_csv(output_file, index=False)

    print(f"  Saved {output_file} ({len(df_final)} rows)")

    return df_final


def process_all_glaciers(m_index):
    """
    Process all stakes of all glaciers.

    Parameters
    ----------
    m_index : int
        Index of the (m, C) pair used in Elmer/Ice: 0 for m=1, 1 for m=3, 2 for m=6.
    """
    print(f"\nProcessing all glaciers (m_index={m_index})")

    for glacier_name, config in GLACIERS.items():
        m, C = config['mval_Cval'][m_index]
        for stake_name in config['xy_coords'].keys():
            try:
                process_glacier_stake(glacier_name, stake_name, config, m, C)
            except Exception as e:
                print(f"\n[ERROR] {glacier_name} - {stake_name}: {e}")
                import traceback
                traceback.print_exc()
                continue


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    for m_index in range(3):
        process_all_glaciers(m_index)