# Glacier Friction Analysis

Package Python pour l'analyse des lois de friction des glaciers alpins.

## 📋 Description

Ce package fournit des outils pour :
- Modéliser les lois de friction glaciaire (Weertman, cavitation)
- Ajuster ces lois aux données observées de vitesse et contrainte basales
- Visualiser les résultats et comparer différents glaciers
- Gérer les flowlines (lignes d'écoulement) glaciaires

## 🚀 Installation

### Installation standard

```bash
cd glacier_friction_analysis
pip install -e .
```

### Installation avec dépendances de développement

```bash
pip install -e ".[dev]"
```

### Installation avec support Jupyter

```bash
pip install -e ".[notebooks]"
```

## 📦 Structure du projet

```
glacier_friction_analysis/
├── src/                          # Code source du package
│   ├── __init__.py              # Point d'entrée du package
│   ├── constants.py             # Constantes physiques
│   ├── friction_laws.py         # Lois de friction
│   ├── fitting.py               # Ajustement de courbes
│   ├── visualization.py         # Outils de visualisation
│   └── flowlines.py             # Gestion des flowlines
├── examples/                     # Scripts d'exemple
│   └── example_usage.py         # Exemples d'utilisation
├── notebooks/                    # Jupyter notebooks (vos analyses)
├── data/                         # Données d'entrée
├── outputs/                      # Résultats (figures, CSV)
├── tests/                        # Tests unitaires
├── config/                       # Fichiers de configuration
├── requirements.txt             # Dépendances
├── setup.py                     # Script d'installation
└── README.md                    # Ce fichier
```

## 🔧 Utilisation

### Exemple 1 : Ajustement d'une loi de Weertman

```python
import numpy as np
from glacier_friction_analysis import power_law, FrictionFitter

# Données observées
velocities = np.array([10, 20, 50, 100])  # m/an
stresses = np.array([0.05, 0.08, 0.15, 0.25])  # MPa

# Créer un fitter
fitter = FrictionFitter()

# Ajuster la loi de puissance
m, As, rmse = fitter.fit_weertman(
    velocities, 
    stresses,
    initial_guess=(3.0, 1e5),
    friclaw=power_law
)

print(f"m = {m:.2f}, As = {As:.2e}, RMSE = {rmse:.2e}")
```

### Exemple 2 : Visualisation

```python
from glacier_friction_analysis import GlacierVisualizer
import matplotlib.pyplot as plt

# Créer un visualizer
vis = GlacierVisualizer()

# Tracer les données
fig, ax = plt.subplots()
vis.plot_friction_data(
    velocities, stresses,
    glacier_name="Argentière",
    color=vis.get_glacier_color('Arg'),
    marker=vis.get_glacier_marker('Arg'),
    ax=ax
)
plt.show()
```

### Exemple 3 : Calcul de k_c

```python
from glacier_friction_analysis import compute_kc
import numpy as np

# Calculer k_c pour différents débits
Q_values = np.logspace(-3, 1, 100)  # 0.001 à 10 m³/s
kc_values = [compute_kc(Q) for Q in Q_values]

# Visualiser
from glacier_friction_analysis import GlacierVisualizer
vis = GlacierVisualizer()
fig = vis.plot_kc_vs_discharge(Q_values, kc_values)
```

### Exemple 4 : Gestion des flowlines

```python
from glacier_friction_analysis import FlowlineManager
from pathlib import Path

# Créer un gestionnaire
manager = FlowlineManager(data_dir=Path("./data"))

# Définir des points de contrôle
points = [[x1, y1], [x2, y2], [x3, y3]]

# Calculer la flowline interpolée
x_flow, y_flow = manager.compute_flowline(points, num_points=100)

# Sauvegarder
manager.save_flowline("MyGlacier", x_flow, y_flow)
```

## 📊 Lois de friction disponibles

### Loi de puissance (Weertman)

```
τ_b = (u_bed / As)^(1/m)
```

où :
- `τ_b` : contrainte basale de cisaillement [Pa]
- `u_bed` : vitesse de glissement basal [m/an]
- `m` : exposant de la loi
- `As` : paramètre de friction

### Loi de cavitation

```
τ_b = CN * (χ / (1 + α * χ^q))^(1/m)
```

où :
- `χ = u_bed / (As * CN^m)`
- `α = ((q-1)^(q-1)) / (q^q)`
- `q` : exposant de cavitation
- `CN` : pression de contact effective [Pa]

## 🧪 Tests

Pour exécuter les tests unitaires :

```bash
pytest tests/
```

Avec couverture de code :

```bash
pytest --cov=src tests/
```

## 📝 Constantes physiques

Le package inclut les constantes suivantes (valeurs SI) :

| Constante | Symbole | Valeur | Unité |
|-----------|---------|--------|-------|
| Densité de la glace | `RHO_I` | 910.0 | kg/m³ |
| Gravité | `G` | 9.81 | m/s² |
| Chaleur latente | `L` | 3.35×10⁵ | J/kg |
| Exposant de Glen | `N` | 3.0 | - |
| Exposant cavitation | `ALPHA_C` | 1.25 | - |
| Paramètre de fluage | `A` | 2.4×10⁻²⁴ | Pa⁻³ s⁻¹ |

## 🎨 Palettes de couleurs par glacier

Le package inclut des palettes prédéfinies pour chaque glacier :

- **Argentière (Arg)** : Rouge/Brun
- **Mer de Glace (MDG)** : Orange/Jaune
- **Giétroz (Gie)** : Vert
- **Gébroulaz (Geb)** : Gris/Olive
- **Saint-Sorlin (StSo)** : Violet/Magenta
- **Corbassière (Cor)** : Jaune vif
- **Glacier Blanc (GB)** : Bleu

## 📚 Documentation

Pour plus de détails sur chaque module, consultez les docstrings :

```python
from glacier_friction_analysis import power_law
help(power_law)
```