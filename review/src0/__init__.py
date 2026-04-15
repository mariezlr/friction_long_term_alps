"""
Glacier Friction Analysis Package
==================================

Package pour l'analyse des lois de friction des glaciers alpins.

Modules principaux:
- constants: Constantes physiques et rhéologiques
- friction_laws: Implémentations des lois de friction
- fitting: Ajustement des lois aux données
- visualization: Outils de visualisation
- flowlines: Gestion des flowlines glaciaires
"""

__version__ = "1.0.0"
__author__ = "Glacier Friction Analysis Team"

from .constants import (
    RHO_I, G, L, N, ALPHA_C, A, A_C,
    DEFAULT_Q, DEFAULT_C, DEFAULT_CN,
    DATE_MIN, DATE_MAX
)

from .friction_laws import (
    power_law,
    power_law_m3,
    inverse_power_law,
    cavitation_law,
    inverse_cavitation_law,
    compute_kc
)

from .fitting import FrictionFitter

from .visualization import GlacierVisualizer

from .flowlines import FlowlineManager, create_all_flowlines

__all__ = [
    # Constants
    'RHO_I', 'G', 'L', 'N', 'ALPHA_C', 'A', 'A_C',
    'DEFAULT_Q', 'DEFAULT_C', 'DEFAULT_CN',
    'DATE_MIN', 'DATE_MAX',
    
    # Friction laws
    'power_law',
    'power_law_m3',
    'inverse_power_law',
    'cavitation_law',
    'inverse_cavitation_law',
    'compute_kc',
    
    # Fitting and visualization
    'FrictionFitter',
    'GlacierVisualizer',
    
    # Flowlines
    'FlowlineManager',
    'create_all_flowlines'
]
