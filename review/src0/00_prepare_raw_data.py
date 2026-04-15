#!/usr/bin/env python3
"""
Préparation des données brutes pour l'analyse de friction.

Ce script est l'intermédiaire entre:
- Sorties brutes d'Elmer/Ice (profils .dat, grilles .vtu)
- Observations in-situ (CSV vitesse et épaisseur)
→ Dataset consolidé exploitable pour l'analyse

Structure du dataset final:
- glacier: nom du glacier
- stake: nom du stake/profil
- date_obs: dates avec observations in-situ
- date_elmer: dates avec sorties Elmer (DEM)
- H_obs: épaisseur observée [m]
- u_surf_obs: vitesse de surface observée [m/an]
- u_def_elmer: vitesse de déformation modélisée [m/an]
- tau_b_elmer: contrainte basale modélisée [Pa]
- slope_elmer: pente locale [rad]
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration des glaciers et leurs profils
GLACIER_CONFIG = {
    'Arg': {
        'stakes': {
            '4': {'x': 958350, 'y': 117550},
            '5': {'x': 958450, 'y': 117750},
            'wheel': {'x': 958652, 'y': 118002}
        },
        'years_elmer': [1905, 1935, 1952, 1969, 1979, 1987, 1994, 1998, 
                        2000, 2003, 2008, 2012, 2018],
        'data_dir': 'data_raw/Arg'
    },
    'Geb': {
        'stakes': {
            'sup': {'x': 330560, 'y': 103100},
            'ss': {'x': 330650, 'y': 102950}
        },
        'years_elmer': [1952, 1967, 1979, 1998, 2003, 2009, 2015, 2021],
        'data_dir': 'data_raw/Geb'
    },
    'MDG': {
        'stakes': {
            'tac': {'x': 956100, 'y': 110500},
            'trel': {'x': 956200, 'y': 112000},
            'ech': {'x': 955800, 'y': 113500}
        },
        'years_elmer': [1905, 1937, 1949, 1958, 1960, 1963, 1968, 1970, 
                        1971, 1972, 1973, 1979, 1984, 1988, 1994, 1998, 
                        2000, 2003, 2008, 2012, 2018],
        'data_dir': 'data_raw/MDG'
    },
    'Cor': {
        'stakes': {
            'A4': {'x': 2588450, 'y': 1094257},
            'B4': {'x': 2589392, 'y': 1093101}
        },
        'years_elmer': [1952, 1964, 1973, 1983, 1993, 1999, 2009, 2012, 2017, 2020],
        'data_dir': 'data_raw/Cor'
    },
    # Ajouter autres glaciers...
}


# ============================================================================
# LECTURE DES SORTIES ELMER
# ============================================================================

def read_elmer_profile(filepath):
    """
    Lit un fichier de profil Elmer (Solver 7 SaveLine).
    
    Format attendu: colonnes séparées par espaces
    - x, y, z (coordonnées)
    - velocity 1, 2, 3 (composantes de vitesse)
    - depth
    
    Parameters
    ----------
    filepath : Path
        Chemin du fichier .dat
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame avec les données du profil
    """
    try:
        # Lire le fichier (adapter selon le format exact)
        df = pd.read_csv(filepath, sep=r'\s+', header=None,
                        names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'depth'])
        
        # Calculer vitesse horizontale
        df['u_horiz'] = np.sqrt(df['vx']**2 + df['vy']**2)
        
        # Calculer épaisseur si bed connu
        # H = z_surf - z_bed (à adapter selon vos données)
        
        return df
        
    except Exception as e:
        warnings.warn(f"Erreur lecture {filepath}: {e}")
        return None


def extract_elmer_at_point(df_profile, x_stake, y_stake, radius=50):
    """
    Extrait les valeurs Elmer au point le plus proche d'un stake.
    
    Parameters
    ----------
    df_profile : pd.DataFrame
        DataFrame du profil Elmer
    x_stake, y_stake : float
        Coordonnées du stake
    radius : float, optional
        Rayon de recherche [m]
        
    Returns
    -------
    values : dict
        Dictionnaire avec les valeurs extraites
    """
    if df_profile is None or len(df_profile) == 0:
        return None
    
    # Calculer distance au stake
    dx = df_profile['x'] - x_stake
    dy = df_profile['y'] - y_stake
    dist = np.sqrt(dx**2 + dy**2)
    
    # Point le plus proche
    idx_min = dist.argmin()
    
    if dist[idx_min] > radius:
        warnings.warn(f"Point le plus proche à {dist[idx_min]:.1f}m > {radius}m")
        return None
    
    # Extraire valeurs
    row = df_profile.iloc[idx_min]
    
    return {
        'x': row['x'],
        'y': row['y'],
        'vx': row['vx'],
        'vy': row['vy'],
        'vz': row['vz'],
        'u_horiz': row['u_horiz'],
        'depth': row['depth'],
        'distance_to_stake': dist[idx_min]
    }


def load_elmer_timeseries(glacier_name, stake_name, config):
    """
    Charge toutes les sorties Elmer pour un stake donné.
    
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
    df_elmer : pd.DataFrame
        DataFrame avec colonnes: date_elmer, u_def, tau_b, slope, H
    """
    data_dir = Path(config['data_dir'])
    years = config['years_elmer']
    stake_coords = config['stakes'][stake_name]
    
    records = []
    
    for year in years:
        # Construire nom de fichier (adapter selon votre convention)
        # Exemple: Profil_ARG_init_1905.dat
        profile_file = data_dir / 'Elmer_outputs' / f'Profil_{glacier_name}_init_{year}.dat'
        
        if not profile_file.exists():
            warnings.warn(f"Fichier non trouvé: {profile_file}")
            continue
        
        # Lire profil
        df_profile = read_elmer_profile(profile_file)
        
        if df_profile is None:
            continue
        
        # Extraire au stake
        values = extract_elmer_at_point(
            df_profile, 
            stake_coords['x'], 
            stake_coords['y']
        )
        
        if values is None:
            continue
        
        # Stocker
        record = {
            'date_elmer': year,
            'u_def': values['u_horiz'],  # Vitesse de déformation
            'x': values['x'],
            'y': values['y'],
            'depth': values['depth']
        }
        
        # Ajouter tau_b et slope si disponibles
        # (nécessite lecture d'autres fichiers Elmer)
        
        records.append(record)
    
    if len(records) == 0:
        return pd.DataFrame()
    
    df_elmer = pd.DataFrame(records)
    
    return df_elmer


# ============================================================================
# LECTURE DES OBSERVATIONS IN-SITU
# ============================================================================

def load_observations(glacier_name, stake_name, data_dir, obs_type='velocity'):
    """
    Charge les observations in-situ (vitesse ou épaisseur).
    
    Format attendu: CSV avec colonnes 'date' et 'value'
    
    Parameters
    ----------
    glacier_name : str
        Nom du glacier
    stake_name : str
        Nom du stake
    data_dir : Path
        Répertoire des données
    obs_type : str
        'velocity' ou 'thickness'
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame avec colonnes: date_obs, value
    """
    data_dir = Path(data_dir)
    
    # Construire nom de fichier
    if obs_type == 'velocity':
        filename = f'{glacier_name}_{stake_name}_velocity.csv'
    elif obs_type == 'thickness':
        filename = f'{glacier_name}_{stake_name}_thickness.csv'
    else:
        raise ValueError(f"obs_type inconnu: {obs_type}")
    
    filepath = data_dir / 'observations' / filename
    
    if not filepath.exists():
        warnings.warn(f"Fichier non trouvé: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        
        # Standardiser noms de colonnes
        if 'date' not in df.columns and 'year' in df.columns:
            df = df.rename(columns={'year': 'date'})
        
        # Identifier colonne de valeur
        value_cols = [c for c in df.columns if c != 'date']
        if len(value_cols) == 0:
            warnings.warn(f"Aucune colonne de valeur dans {filepath}")
            return pd.DataFrame()
        
        value_col = value_cols[0]
        
        # Renommer
        df = df.rename(columns={value_col: 'value'})
        df = df[['date', 'value']]
        df = df.rename(columns={'date': 'date_obs'})
        
        # Nettoyer
        df['date_obs'] = pd.to_numeric(df['date_obs'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        
        return df
        
    except Exception as e:
        warnings.warn(f"Erreur lecture {filepath}: {e}")
        return pd.DataFrame()


# ============================================================================
# FUSION DES DONNÉES
# ============================================================================

def merge_elmer_and_observations(df_elmer, df_velocity, df_thickness):
    """
    Fusionne les données Elmer et observations.
    
    Stratégie:
    - Garder toutes les dates d'observation (outer join sur date_obs)
    - Ajouter les dates Elmer qui n'ont pas d'observations
    - Marquer d'où vient chaque donnée
    
    Parameters
    ----------
    df_elmer : pd.DataFrame
        Données Elmer (date_elmer, u_def, tau_b, ...)
    df_velocity : pd.DataFrame
        Observations de vitesse (date_obs, value)
    df_thickness : pd.DataFrame
        Observations d'épaisseur (date_obs, value)
        
    Returns
    -------
    df_merged : pd.DataFrame
        Dataset fusionné
    """
    # Créer dataset de base avec observations
    if not df_velocity.empty and not df_thickness.empty:
        df_base = pd.merge(
            df_velocity.rename(columns={'value': 'u_surf_obs'}),
            df_thickness.rename(columns={'value': 'H_obs'}),
            on='date_obs',
            how='outer'
        )
    elif not df_velocity.empty:
        df_base = df_velocity.rename(columns={'value': 'u_surf_obs'})
    elif not df_thickness.empty:
        df_base = df_thickness.rename(columns={'value': 'H_obs'})
    else:
        df_base = pd.DataFrame()
    
    # Ajouter dates Elmer
    if not df_elmer.empty:
        # Créer une colonne 'date' unifiée
        if not df_base.empty:
            df_base['date'] = df_base['date_obs']
        
        df_elmer_for_merge = df_elmer.copy()
        df_elmer_for_merge['date'] = df_elmer_for_merge['date_elmer']
        
        # Fusionner
        df_merged = pd.merge(
            df_base,
            df_elmer_for_merge,
            on='date',
            how='outer'
        )
    else:
        df_merged = df_base
        if 'date' not in df_merged.columns and 'date_obs' in df_merged.columns:
            df_merged['date'] = df_merged['date_obs']
    
    # Trier par date
    if 'date' in df_merged.columns:
        df_merged = df_merged.sort_values('date').reset_index(drop=True)
    
    return df_merged


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def prepare_glacier_stake_data(glacier_name, stake_name, config, output_dir):
    """
    Prépare le dataset complet pour un glacier/stake.
    
    Parameters
    ----------
    glacier_name : str
        Nom du glacier
    stake_name : str
        Nom du stake
    config : dict
        Configuration du glacier
    output_dir : Path
        Répertoire de sortie
        
    Returns
    -------
    df_final : pd.DataFrame
        Dataset préparé
    """
    print(f"\n{'='*60}")
    print(f"{glacier_name} - {stake_name}")
    print(f"{'='*60}")
    
    data_dir = Path(config['data_dir'])
    
    # 1. Charger données Elmer
    print("\n1. Chargement données Elmer...")
    df_elmer = load_elmer_timeseries(glacier_name, stake_name, config)
    print(f"   → {len(df_elmer)} dates avec sorties Elmer")
    
    # 2. Charger observations
    print("\n2. Chargement observations in-situ...")
    df_velocity = load_observations(glacier_name, stake_name, data_dir, 'velocity')
    print(f"   → {len(df_velocity)} observations de vitesse")
    
    df_thickness = load_observations(glacier_name, stake_name, data_dir, 'thickness')
    print(f"   → {len(df_thickness)} observations d'épaisseur")
    
    # 3. Fusionner
    print("\n3. Fusion des données...")
    df_merged = merge_elmer_and_observations(df_elmer, df_velocity, df_thickness)
    print(f"   → {len(df_merged)} dates au total")
    
    # 4. Ajouter métadonnées
    df_merged['glacier'] = glacier_name
    df_merged['stake'] = stake_name
    
    # 5. Réorganiser colonnes
    cols_order = ['glacier', 'stake', 'date', 'date_obs', 'date_elmer',
                  'H_obs', 'u_surf_obs', 'u_def', 'tau_b', 'slope']
    existing_cols = [c for c in cols_order if c in df_merged.columns]
    other_cols = [c for c in df_merged.columns if c not in existing_cols]
    df_final = df_merged[existing_cols + other_cols]
    
    # 6. Sauvegarder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{glacier_name}_{stake_name}_prepared.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✓ Sauvegardé: {output_file}")
    
    return df_final


def prepare_all_data(output_dir='data_prepared'):
    """
    Prépare les données pour tous les glaciers.
    
    Parameters
    ----------
    output_dir : str or Path
        Répertoire de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PRÉPARATION DES DONNÉES BRUTES")
    print("="*80)
    
    all_data = []
    
    for glacier_name, config in GLACIER_CONFIG.items():
        for stake_name in config['stakes'].keys():
            try:
                df = prepare_glacier_stake_data(
                    glacier_name, stake_name, config, output_dir
                )
                all_data.append(df)
            except Exception as e:
                print(f"\n✗ Erreur {glacier_name} - {stake_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Créer fichier consolidé
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        output_file = output_dir / 'all_data_prepared.csv'
        df_all.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Fichier consolidé: {output_file}")
        print(f"  Total: {len(df_all)} enregistrements")
        print(f"  Glaciers: {df_all['glacier'].nunique()}")
        print(f"{'='*80}")


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == '__main__':
    prepare_all_data()
