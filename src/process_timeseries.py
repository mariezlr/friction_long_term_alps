#!/usr/bin/env python3
"""
Traitement des séries temporelles glaciaires.

Reproduit le workflow du notebook Timeseries_Alpine_glaciers.ipynb :
1. Lit les sorties Elmer aux dates DEM
2. Lit les observations in-situ (altitude, vitesse)
3. Établit relations empiriques τ_b ~ H et u_def ~ H^4
4. Interpole sur toutes les dates d'observation
5. Calcule u_bed = u_surf - u_def
6. Sauvegarde CSV final

Output: glacier_all_data_stake.csv avec colonnes:
    date, u_bed_elmer, u_surf_elmer, tau_d_elmer, tau_b_elmer, 
    sigma_elmer, u_def_elmer, slope, ..., 
    altitude, velocity, thickness, obs_tau_b, obs_u_def, obs_u_bed
"""
from utils import GLACIERS, geom_data_dir, proc_data_dir
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

script_dir = Path(__file__).resolve().parent

# ============================================================================
# LECTURE DES SORTIES ELMER
# ============================================================================

def read_elmer_data_file(glacier_name, year, m_index):
    """
    Lit le fichier Elmer_data_{year}.dat pré-calculé.

    """
    # Essayer de lire le fichier pré-calculé
    m, C = GLACIERS[f'{glacier_name}']['mval_Cval'][m_index]
    elmer_file = script_dir / '..' / 'data'/ 'elmer_raw' / f'mw{m:.0f}' / f'{glacier_name}_{year}.csv'

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
# CALCULS DE FRICTION
# ============================================================================

def calc_tau_b(u_bed, C, m=3):
    """Calcule τ_b avec loi de Weertman."""
    return C * (u_bed ** (1/m))


def calc_tau_d(thickness, xgrad, ygrad, zgrad):
    """Calcule driving stress τ_d = ρ*g*H*sin(α)."""
    norm_grad = np.sqrt(xgrad**2 + ygrad**2)
    slope = np.where(norm_grad != 0, zgrad / norm_grad, 0)
    angle = np.arctan(slope)
    return 1e-6 * 917 * 9.81 * thickness * np.sin(angle)


# ============================================================================
# MOYENNAGE SPATIAL
# ============================================================================

def average_in_radius(glacier_name, df, x0, y0, radius, m_index, Hmin=20):
    """
    Calcule les variables moyennées dans un rayon autour d'un point.
    
    Parameters
    ----------
    x0, y0 : float
        Coordonnées du centre
    radius : float
        Rayon de moyennage [m]
    C : float
        Coefficient de friction
    m : float
        Exposant
    Hmin : float
        Épaisseur minimale [m]
        
    Returns
    -------
    dict
        Variables moyennées
    """
    # Calculer distances
    df['distance'] = np.sqrt((df['xcoord'] - x0)**2 + (df['ycoord'] - y0)**2)
    
    # Sélectionner voisinage
    mask = (df['distance'] <= radius) & (df['thicksurf'] >= Hmin)
    voisinage = df[mask].copy()
    
    if len(voisinage) == 0:
        return None
    
    # Voisinage proche pour la pente
    mask_proche = (df['distance'] <= 50) & (df['thicksurf'] >= Hmin)
    voisinage_proche = df[mask_proche].copy()
    
    # Calculs de pente
    slopex = -voisinage['xgrad'].mean(skipna=True)
    slopey = -voisinage['ygrad'].mean(skipna=True)
    
    voisinage['zgrad_dirmean'] = (slopex * voisinage['xgrad'] + 
                                  slopey * voisinage['ygrad'])
    slopez = voisinage['zgrad_dirmean'].mean(skipna=True)
    normslope = np.sqrt(slopex**2 + slopey**2 + slopez**2)
    
    if normslope == 0:
        normslope = 1e-9
    
    # Projection du vecteur normal
    voisinage['projvector'] = (
        (slopex / normslope) * voisinage['normalbed1'] +
        (slopey / normslope) * voisinage['normalbed2'] +
        (slopez / normslope) * voisinage['normalbed3']
    )
    
    # Moyennes pondérées par surface
    total_area = voisinage['nodearea'].sum(skipna=True)
    
    if total_area == 0:
        return None
    
    vel_h_bed = voisinage['vel_h_bed'].mean(skipna=True)
    vel_h_surf = voisinage['vel_h_surf'].mean(skipna=True)
    
    sigma = (voisinage['normalstress'] * voisinage['projvector'] * 
             voisinage['nodearea']).sum(skipna=True) / total_area
    
    # Driving stress
    voisinage['tau_d'] = calc_tau_d(
        voisinage['thicksurf'], 
        voisinage['xgrad'], 
        voisinage['ygrad'],
        voisinage['xgrad']**2 + voisinage['ygrad']**2
    )
    tau_d = (voisinage['tau_d'] * voisinage['nodearea']).sum(skipna=True) / total_area
    
    # Basal stress
    m, C = GLACIERS[glacier_name]['mval_Cval'][m_index]
    voisinage['tau_b'] = calc_tau_b(voisinage['vel_h_bed'], C, m)
    tau_b = (voisinage['tau_b'] * voisinage['nodearea']).sum(skipna=True) / total_area
    
    # Pentes
    slope = np.arctan(np.sqrt(
        voisinage_proche['xgrad']**2 + voisinage_proche['ygrad']**2
    )).mean(skipna=True)
    
    averaged_slope = np.arctan(np.sqrt(
        voisinage['xgrad']**2 + voisinage['ygrad']**2
    )).mean(skipna=True)
    
    return {
        'u_bed_elmer': vel_h_bed,
        'u_surf_elmer': vel_h_surf,
        'tau_d_elmer': tau_d,
        'tau_b_elmer': tau_b,
        'sigma_elmer': sigma,
        'u_def_elmer': vel_h_surf - vel_h_bed,
        'slope': slope,
        'averaged_slope': averaged_slope,
        'gradxmean': slopex,
        'gradymean': slopey,
        'gradzmean': slopez,
        'normalstress': voisinage['normalstress'].mean(skipna=True),
        'projvector': voisinage['projvector'].mean(skipna=True)
    }


def process_elmer_timeseries(glacier_name, years_DEM, x0, y0, radius, m_index=1, Hmin=20):
    """
    Traite toutes les années Elmer pour un stake.
    
    Parameters
    ----------
    years_DEM : list
        Années avec DEM
    x0, y0 : float
        Coordonnées du stake
    radius : float
        Rayon de moyennage [m]
    m_index : int
        Indice des valeurs (C,m) pour Weertman
    Hmin : float
        Épaisseur minimale [m]
        
    Returns
    -------
    df : pd.DataFrame
        Série temporelle Elmer
    """
    records = []
    
    for year in years_DEM:
        # Lire données Elmer
        df_elmer = read_elmer_data_file(glacier_name, year, m_index)
        
        if df_elmer.empty:
            continue
        
        # Moyenner dans rayon
        result = average_in_radius(glacier_name, df_elmer, x0, y0, radius, m_index, Hmin)
        
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
# LECTURE DES OBSERVATIONS
# ============================================================================

def read_observations(glacier_name, stake_name):
    """
    Lit les observations in-situ (altitude, vitesse).
    
    Parameters
    ----------
    glacier_name : str
        Nom du glacier
    stake_name : str
        Nom du stake
        
    Returns
    -------
    dict
        Dictionnaire avec 'altitude', 'velocity', 'thickness' DataFrames
    """
    obs = {}
    
    # Altitude (= thickness)
    alt_file = script_dir / '..' / 'data' / 'obs_raw' / f'{glacier_name}_alt_{stake_name}.csv'
    if alt_file.exists():
        df = pd.read_csv(alt_file)
        # Renommer colonnes si nécessaire
        if 'year' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'year': 'date'})
        obs['altitude'] = df
    
    # Vitesse
    vel_file = script_dir / '..' / 'data' / 'obs_raw' / f'{glacier_name}_vel_{stake_name}.csv'
    if vel_file.exists():
        df = pd.read_csv(vel_file)
        if 'year' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'year': 'date'})
        obs['velocity'] = df
    
    # Thickness (peut être calculé depuis altitude si nécessaire)
    # À adapter selon votre structure de données
    
    return obs

def interp_zdem(mnt_bed, xx, yy):
    # Charger les données du lit rocheux
    
    # Extraire les colonnes x, y, z
    x = mnt_bed.iloc[:, 0].values
    y = mnt_bed.iloc[:, 1].values
    z = mnt_bed.iloc[:, 2].values

    # Utiliser griddata pour l'interpolation
    zi = griddata((x, y), z, (xx, yy), method='linear')
    
    return round(float(zi), 2)

# ============================================================================
# INTERPOLATION TEMPORELLE
# ============================================================================

def fit_empirical_relation(x_obs, y_elmer, degree=1):
    """
    Ajuste une relation empirique entre observations et Elmer.
    
    Parameters
    ----------
    x_obs : array
        Variable observée (ex: thickness)
    y_elmer : array
        Variable Elmer (ex: tau_b)
    degree : int
        Degré du polynôme
        
    Returns
    -------
    coeffs : array
        Coefficients du polynôme
    """
    # Retirer NaN
    mask = ~np.isnan(x_obs) & ~np.isnan(y_elmer)
    
    if mask.sum() < degree + 1:
        return None
    
    x_clean = x_obs[mask]
    y_clean = y_elmer[mask]
    
    # Ajuster
    coeffs = np.polyfit(x_clean, y_clean, degree)
    
    return coeffs


def apply_empirical_relation(x_continuous, coeffs):
    """Applique la relation empirique."""
    if coeffs is None:
        return np.full_like(x_continuous, np.nan)
    
    poly = np.poly1d(coeffs)
    return poly(x_continuous)


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def process_glacier_stake(glacier_name, stake_name, config, m_index):
    """
    Traite un glacier/stake complet.
    
    Parameters
    ----------
    glacier_name : str
        Nom du glacier
    stake_name : str
        Nom du stake
    config : dict
        Configuration du glacier

        
    Returns
    -------
    df_final : pd.DataFrame
        Dataset final
    """
    print(f"\n{'='*60}")
    print(f"Traitement: {glacier_name} - {stake_name}")
    print(f"{'='*60}")

    years_DEM = config['years_DEM']
    m, C = config['mval_Cval'][m_index]
    x0, y0 = config['xy_coords'][stake_name]
    Hmin = 20  # ou config['Hmin'][stake_name] si ajout plus tard
    radius = config['avg_dist'][stake_name]
    
    print(f"  Coordonnées: ({x0}, {y0})")
    print(f"  Rayon moyennage: {radius} m")
    print(f"  Friction: C={C}, m={m}")
    
    # 1. Traiter données Elmer
    print("\n→ Traitement données Elmer...")
    df_elmer = process_elmer_timeseries(
        glacier_name, years_DEM, x0, y0, radius, m_index, Hmin
    )
    print(f"  ✓ {len(df_elmer)} dates Elmer")
    
    # 2. Lire observations
    print("\n→ Lecture observations...")
    obs = read_observations(glacier_name, stake_name)
    
    df_altitude = obs.get('altitude', pd.DataFrame())
    df_velocity = obs.get('velocity', pd.DataFrame())
    
    print(f"  ✓ {len(df_altitude)} obs altitude")
    print(f"  ✓ {len(df_velocity)} obs vitesse")
    
    # 3. Fusionner Elmer + observations aux dates DEM
    print("\n→ Fusion Elmer + observations...")
    
    # Créer DataFrame avec observations continues
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
    
    # Calculer thickness si disponible
    if 'altitude' in df_obs.columns:
        mnt_bed_path = geom_data_dir / 'bedrocks' / f'DEM_bedrock_{glacier_name}.dat'
        mnt_bed = pd.read_csv(mnt_bed_path, delimiter=r'\s+', header=None)
        zbedrock = interp_zdem(mnt_bed, x0, y0)
        df_obs['thickness'] = df_obs['altitude'] - zbedrock
    
    # Fusionner avec Elmer aux dates DEM
    print(df_elmer.columns)
    print(df_obs.columns)

    df_elmer = df_elmer.sort_values("date")
    df_obs   = df_obs.sort_values("date")
    df_elmer["date"] = df_elmer["date"].astype(float)
    df_obs["date"]   = df_obs["date"].astype(float)

    df_merged_dem = pd.merge_asof(
        df_elmer,
        df_obs,
        on="date",
        direction="nearest",   # prend la date la plus proche
        tolerance=4            # tolérance de ±4 ans
    )
    
    print(f"  ✓ {len(df_merged_dem)} dates avec Elmer ET observations")
    
    if len(df_merged_dem) < 3:
        print("  ✗ Pas assez de points pour ajuster les relations")
        return None
    
    # 4. Ajuster relations empiriques
    print("\n→ Ajustement relations empiriques...")
    
    if glacier_name == "GB": # Exception glacier blanc où on utilise la pente
        # τ_b ~ H \times slope
        HS = (df_merged_dem['thickness'].values) * (df_merged_dem['thickness'].values)
        coeffs_tau = fit_empirical_relation(
            HS, df_merged_dem['tau_b_elmer'].values, degree=1)
        
        # u_def ~ H^4 \times slope^3
        H4S3 = (df_merged_dem['thickness'].values ** 4) * (df_merged_dem['slope'].values ** 3)
        coeffs_udef = fit_empirical_relation(
            H4S3, df_merged_dem['u_def_elmer'].values, degree=1)
    
    else:
        # τ_b ~ H (linéaire)
        coeffs_tau = fit_empirical_relation(
            df_merged_dem['thickness'].values,
            df_merged_dem['tau_b_elmer'].values, degree=1)
        
        # u_def ~ H^4 (linéaire en H^4)
        H4 = df_merged_dem['thickness'].values ** 4
        coeffs_udef = fit_empirical_relation(
            H4, df_merged_dem['u_def_elmer'].values, degree=1)

    if coeffs_tau is not None:
        print(f"  ✓ τ_b = {coeffs_tau[0]:.2e} * H + {coeffs_tau[1]:.2e}")
    if coeffs_udef is not None:
        print(f"  ✓ u_def = {coeffs_udef[0]:.2e} * H^4 + {coeffs_udef[1]:.2e}")
    
    # 5. Appliquer aux observations continues
    print("\n→ Interpolation temporelle...")
    
    if 'thickness' in df_obs.columns:
        df_obs['obs_tau_b'] = apply_empirical_relation(
            df_obs['thickness'].values, coeffs_tau
        )
        
        df_obs['obs_u_def'] = apply_empirical_relation(
            df_obs['thickness'].values ** 4, coeffs_udef
        )
        
        # Calculer u_bed
        if 'velocity' in df_obs.columns:
            df_obs['obs_u_bed'] = df_obs['velocity'] - df_obs['obs_u_def']
    
    # 6. Fusionner tout
    print("\n→ Création dataset final...")
    df_final = pd.merge(
        df_elmer,
        df_obs,
        on='date',
        how='outer'
    )
    
    df_final = df_final.sort_values('date').reset_index(drop=True)
    
    # 7. Sauvegarder
    output_dir = Path(script_dir / '..' / 'data' / 'processed_timeseries' / f'mw{1/m:.3f}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{glacier_name}_all_data_{stake_name}.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Sauvegardé: {output_file}")
    print(f"  {len(df_final)} lignes au total")
    print(f"  Colonnes: {list(df_final.columns)}")
    
    return df_final


def process_all_glaciers(m_index):
    """
    Traite tous les glaciers.
    
    Parameters
    ----------
    m_index : int
        0 pour m=1, 1 pour m=3, 2 pour m=6
    output_dir : str or Path
        Répertoire de sortie
    """
    print("\n" + "="*80)
    print(f"TRAITEMENT SÉRIES TEMPORELLES - m_index={m_index}")
    print("="*80)
    
    for glacier_name, config in GLACIERS.items():
        for stake_name in config['xy_coords'].keys():
            try:
                process_glacier_stake(
                    glacier_name, stake_name, config, m_index
                )
            except Exception as e:
                print(f"\n✗ Erreur {glacier_name} - {stake_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*80)
    print("TRAITEMENT TERMINÉ")
    print("="*80)


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == '__main__':

# Only one stake (example)

    glacier_name = 'Geb'
    stake_name = 'ss'
    m_index = 0

    config = GLACIERS[glacier_name]

    process_glacier_stake(
        glacier_name,
        stake_name,
        config,
        m_index
    )

    glacier_name = 'Geb'
    stake_name = 'ss'
    m_index = 1

    config = GLACIERS[glacier_name]

    process_glacier_stake(
        glacier_name,
        stake_name,
        config,
        m_index
    )

    glacier_name = 'Geb'
    stake_name = 'ss'
    m_index = 2

    config = GLACIERS[glacier_name]

    process_glacier_stake(
        glacier_name,
        stake_name,
        config,
        m_index
    )


# ## Toutes les stakes
#     for m_index in range(3):

#         process_all_glaciers(m_index)

