"""
Constants physiques et rhéologiques pour l'analyse des lois de friction glaciaire.
"""

import numpy as np

# ===== Constantes physiques (SI) =====
RHO_I = 910.0          # Densité de la glace [kg/m³]
G = 9.81               # Accélération gravitationnelle [m/s²]
L = 3.35e5             # Chaleur latente de fusion [J/kg]

# ===== Paramètres rhéologiques =====
N = 3.0                # Exposant de la loi de Glen
ALPHA_C = 1.25         # Exposant pour la loi de cavitation
A = 2.4e-24            # Paramètre de fluage de Glen [Pa⁻³ s⁻¹]
A_C = 2 * A * N**(-N)  # Paramètre ajusté selon la définition

# ===== Paramètres par défaut =====
DEFAULT_Q = 0.1        # Débit par défaut [m³/s]
DEFAULT_C = 0.4        # Coefficient de friction par défaut
DEFAULT_CN = 0.29      # Coefficient CN par défaut [MPa]

# ===== Paramètres temporels =====
DATE_MIN = 1900
DATE_MAX = 2020
