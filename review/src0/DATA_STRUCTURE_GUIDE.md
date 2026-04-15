# Structure et flux des données

## Vue d'ensemble

```
DONNÉES BRUTES                 PRÉPARATION              DONNÉES PRÊTES
===============                ===========              ==============

Elmer/Ice outputs          →                         →  Dataset consolidé
├─ Profils .dat                                         ├─ glacier
├─ Grilles .vtu             00_prepare_raw_data.py     ├─ stake  
└─ Multiple années DEM                                  ├─ date
                                                        ├─ date_obs (observations)
Observations in-situ       →                            ├─ date_elmer (Elmer)
├─ velocity.csv                                         ├─ H_obs (épaisseur)
├─ thickness.csv                                        ├─ u_surf_obs (vitesse surf)
└─ Par profil/stake                                     ├─ u_def (déformation Elmer)
                                                        ├─ tau_b (contrainte Elmer)
                                                        └─ slope (pente Elmer)
```

## 1. Données brutes Elmer/Ice

### Structure des sorties

Elmer/Ice est exécuté pour chaque année où un DEM est disponible. Le fichier `.sif` définit plusieurs solveurs qui produisent différentes sorties :

#### Solver 7: SaveLine (Profils)
- **Fichier**: `Profil_{glacier}_init_{year}.dat`
- **Contenu**: Le long de lignes de profils prédéfinies
  - Coordonnées (x, y, z)
  - Vitesse (vx, vy, vz) - vitesse 3D totale
  - Profondeur (depth)
  
**Format typique** (colonnes séparées par espaces):
```
x          y          z        vx      vy      vz    depth
958350.0  117550.0  2450.5   12.5    8.3    -2.1   120.3
958360.0  117560.0  2449.2   12.7    8.4    -2.0   119.8
...
```

#### Solver 9: ExtractNodeBed (Lit glaciaire)
- **Fichier**: Grille au lit
- **Contenu**: 
  - NormalStress (contrainte normale au lit)
  - Composantes du vecteur normal

#### Solver 10-14: Export vertical et surface
- **Fichier**: Grilles 3D
- **Contenu**:
  - Contraintes projetées verticalement
  - Valeurs à la surface

### Années disponibles par glacier

```python
ANNÉES_ELMER = {
    'Arg': [1905, 1935, 1952, 1969, 1979, 1987, 1994, 1998, 
            2000, 2003, 2008, 2012, 2018],
    'Geb': [1952, 1967, 1979, 1998, 2003, 2009, 2015, 2021],
    'MDG': [1905, 1937, 1949, ..., 2018],  # 21 dates
    # etc.
}
```

**Point clé**: Elmer donne une **couverture spatiale complète** mais seulement à **quelques dates** (5-20 selon le glacier).

## 2. Données brutes observations in-situ

### Structure des fichiers

Pour chaque profil/stake, deux types de fichiers CSV :

#### Fichier de vitesse
- **Nom**: `{glacier}_{stake}_velocity.csv`
- **Colonnes**: `date, velocity`
- **Exemple**:
```csv
date,velocity
1950.5,45.2
1951.3,46.8
1952.1,44.5
...
```

#### Fichier d'épaisseur
- **Nom**: `{glacier}_{stake}_thickness.csv`
- **Colonnes**: `date, thickness`
- **Exemple**:
```csv
date,thickness
1950,152.3
1951,151.8
1952,150.5
...
```

**Point clé**: Les observations donnent une **résolution temporelle élevée** (annuelle ou sub-annuelle) mais seulement à **quelques points** (stakes).

## 3. Coordonnées des stakes

Définies dans la configuration :

```python
STAKES = {
    'Arg': {
        '4': {'x': 958350, 'y': 117550},
        '5': {'x': 958450, 'y': 117750},
        'wheel': {'x': 958652, 'y': 118002}
    },
    # etc.
}
```

Ces coordonnées permettent d'extraire les valeurs Elmer au point le plus proche de chaque stake.

## 4. Dataset préparé (sortie de 00_prepare_raw_data.py)

### Format du fichier

Un fichier par glacier/stake: `{glacier}_{stake}_prepared.csv`

### Colonnes du dataset

| Colonne | Type | Source | Description |
|---------|------|--------|-------------|
| `glacier` | str | Méta | Nom du glacier |
| `stake` | str | Méta | Nom du stake/profil |
| `date` | float | Unifiée | Date (année décimale) |
| `date_obs` | float | In-situ | Date avec observation (NaN si Elmer seul) |
| `date_elmer` | float | Elmer | Date avec sortie Elmer (NaN si obs seule) |
| `H_obs` | float | In-situ | Épaisseur observée [m] |
| `u_surf_obs` | float | In-situ | Vitesse de surface observée [m/an] |
| `u_def` | float | Elmer | Vitesse de déformation modélisée [m/an] |
| `tau_b` | float | Elmer | Contrainte basale modélisée [Pa] |
| `slope` | float | Elmer | Pente locale [rad] |
| `x`, `y` | float | Géo | Coordonnées du point |

### Exemple de dataset

```csv
glacier,stake,date,date_obs,date_elmer,H_obs,u_surf_obs,u_def,tau_b,slope
Arg,4,1950.5,1950.5,,152.3,45.2,,,,
Arg,4,1952.0,1952.0,1952.0,150.5,44.5,12.3,8.5e4,0.125
Arg,4,1953.2,1953.2,,149.8,43.1,,,,
Arg,4,1969.0,,1969.0,,,11.8,8.2e4,0.122
...
```

**Remarques** :
- Certaines dates ont **seulement observations** (date_elmer = NaN)
- Certaines dates ont **seulement Elmer** (date_obs = NaN)
- Certaines dates ont **les deux** (dates DEM avec observations)

## 5. Stratégie d'interpolation temporelle

Le script `01_prepare_data.py` traite ensuite ces données :

### Problème à résoudre
- Elmer : u_def, tau_b, slope à dates_elmer (sparse)
- Observations : H_obs, u_surf_obs à dates_obs (dense)
- Besoin : u_def(t), tau_b(t) pour toutes les dates_obs

### Solution : Relations empiriques

À partir des dates où Elmer ET observations coexistent :

1. **Ajuster τ_b vs H** :
   ```
   τ_b ≈ c₁ × H    (Eq. 7a dans le papier)
   ```

2. **Ajuster u_def vs H** :
   ```
   u_def ≈ c₂ × H^(n+1)    (Eq. 7b dans le papier)
   ```

3. **Interpoler** :
   - Utiliser H_obs(t) (continu)
   - Calculer τ_b(t) = c₁ × H_obs(t)
   - Calculer u_def(t) = c₂ × H_obs(t)^4

### Résultat
Série temporelle dense avec :
- `date` : toutes les dates d'observation
- `H` : observé
- `u_surf` : observé
- `u_def` : interpolé via relation empirique
- `tau_b` : interpolé via relation empirique
- `u_bed = u_surf - u_def` : calculé
- **Loi de friction** : ajuster τ_b vs u_bed

## 6. Fichier consolidé

`all_data_prepared.csv` contient tous les glaciers/stakes empilés :

```
Total: ~2000-5000 lignes
├─ Arg_4: 150 lignes
├─ Arg_5: 120 lignes
├─ Geb_sup: 180 lignes
└─ ...
```

## 7. Dimensions du problème

### Dim 1: Glacier
- 7-8 glaciers alpins

### Dim 2: Stake/Profil
- 1-3 stakes par glacier
- Total: ~15-20 stakes

### Dim 3: Date
- **dates_obs**: 20-200 par stake (variable selon glacier)
- **dates_elmer**: 5-20 par glacier
- **dates_interp**: toutes les dates_obs

### Structure proposée (xarray)

```python
import xarray as xr

ds = xr.Dataset(
    {
        'H_obs': (['glacier', 'stake', 'date_obs'], ...),
        'u_surf_obs': (['glacier', 'stake', 'date_obs'], ...),
        'u_def_elmer': (['glacier', 'stake', 'date_elmer'], ...),
        'tau_b_elmer': (['glacier', 'stake', 'date_elmer'], ...),
        'u_def_interp': (['glacier', 'stake', 'date_obs'], ...),
        'tau_b_interp': (['glacier', 'stake', 'date_obs'], ...),
    },
    coords={
        'glacier': ['Arg', 'Geb', 'MDG', ...],
        'stake': ['4', '5', 'sup', ...],
        'date_obs': [1950, 1950.5, 1951, ...],
        'date_elmer': [1952, 1969, 1987, ...],
    }
)
```

**Mais** : dates différentes pour chaque glacier/stake → structure irrégulière

**Solution actuelle** : CSV avec (glacier, stake, date) = format long
- Plus flexible
- Facile à manipuler avec pandas
- Pas de NaN inutiles

## 8. Workflow complet

```
┌──────────────────┐
│  Données brutes  │
├──────────────────┤
│ Elmer: .dat      │ → 00_prepare_raw_data.py → glacier_stake_prepared.csv
│ Obs: .csv        │                             ├─ date_obs (dense)
└──────────────────┘                             ├─ date_elmer (sparse)
                                                 ├─ H_obs, u_surf_obs
                                                 └─ u_def, tau_b (sparse)
                    ↓
┌──────────────────┐
│  Données prêtes  │
├──────────────────┤
│ prepared.csv     │ → 01_prepare_data.py    → glacier_stake_clean.csv
└──────────────────┘   (standardisation)         (colonnes standardisées)
                    ↓
┌──────────────────┐
│  Interpolation   │
├──────────────────┤
│ clean.csv        │ → 02_analyze_timeseries.py → glacier_stake_timeseries.csv
└──────────────────┘   (relations empiriques)     ├─ u_def_interp
                                                   ├─ tau_b_interp
                                                   └─ u_bed = u_surf - u_def
                    ↓
┌──────────────────┐
│  Loi de friction │
├──────────────────┤
│ timeseries.csv   │ → Ajustement Weertman    → Paramètres m, As
└──────────────────┘   τ_b vs u_bed
```

## 9. Avantages de cette approche

1. **Séparation claire** : données brutes → préparées → analysées
2. **Traçabilité** : chaque étape produit un CSV intermédiaire
3. **Flexibilité** : facile d'ajouter un glacier/stake
4. **Robustesse** : chaque fonction gère les erreurs
5. **Réutilisable** : les CSV peuvent être utilisés par d'autres outils

## 10. Fichiers produits

```
data_prepared/
├── Arg_4_prepared.csv
├── Arg_5_prepared.csv
├── Geb_sup_prepared.csv
├── all_data_prepared.csv
└── README_data_structure.txt

data_processed/
├── Arg_4_clean.csv
├── all_glaciers_timeseries.csv
└── ...

outputs/
├── Arg_4_timeseries.csv
├── timeseries_analysis_summary.csv
└── *.png
```
