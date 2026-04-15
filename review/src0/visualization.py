"""
Module de visualisation pour l'analyse des lois de friction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
from typing import Optional, List, Dict


class GlacierVisualizer:
    """
    Classe pour créer des visualisations pour l'analyse glaciaire.
    """
    
    # Palettes de couleurs par glacier
    COLORS = {
        'All': ['#4682B4', '#5F9EA0', '#B0C4DE'],
        'Arg': ['#661100', '#AA4466', '#CC79A7'],
        'Geb': ['#999999', '#669966'],
        'StSo': ['#332288', '#882255', '#AA4499', '#CC6677'],
        'MDG': ['#D55E00', '#E69F00', '#F0E442'],
        'Gie': ['#117733', '#009E73', '#44AA99', '#66CC99'],
        'Cor': ['#FFC300', '#FF5733'],
        'GB': ['#0072B2', '#56B4E9'],
        'GN': ['#888888', '#BBBBBB']
    }
    
    # Marqueurs par glacier
    MARKERS = {
        'All': ['P', '8'],
        'Arg': ['.', 'v', '^'],
        'Geb': ['2', 's'],
        'StSo': ['x', '*', 'd', '|'],
        'MDG': ['o', '*', 'D'],
        'Gie': ['<', '>'],
        'Cor': ['^', 'p'],
        'GB': ['h', 'H'],
        'GN': ['4', '8']
    }
    
    def __init__(self, style: str = 'default'):
        """
        Initialise le visualizer avec un style matplotlib.
        
        Parameters
        ----------
        style : str, optional
            Style matplotlib à utiliser
        """
        self.setup_style()
        self.all_colors = self._flatten_colors()
        self.all_markers = self._flatten_markers()
    
    @staticmethod
    def setup_style():
        """Configure le style matplotlib par défaut."""
        plt.rcParams['lines.linewidth'] = 0.9
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
    
    def _flatten_colors(self) -> List[str]:
        """Aplatit toutes les couleurs en une liste."""
        return [color for colors in self.COLORS.values() for color in colors]
    
    def _flatten_markers(self) -> List[str]:
        """Aplatit tous les marqueurs en une liste."""
        return [marker for markers in self.MARKERS.values() for marker in markers]
    
    def get_glacier_color(self, glacier_name: str, index: int = 0) -> str:
        """
        Retourne une couleur pour un glacier donné.
        
        Parameters
        ----------
        glacier_name : str
            Nom du glacier
        index : int, optional
            Index de la couleur dans la palette du glacier
            
        Returns
        -------
        str
            Code couleur hexadécimal
        """
        colors = self.COLORS.get(glacier_name, self.COLORS['All'])
        return colors[index % len(colors)]
    
    def get_glacier_marker(self, glacier_name: str, index: int = 0) -> str:
        """
        Retourne un marqueur pour un glacier donné.
        
        Parameters
        ----------
        glacier_name : str
            Nom du glacier
        index : int, optional
            Index du marqueur dans la liste du glacier
            
        Returns
        -------
        str
            Caractère de marqueur matplotlib
        """
        markers = self.MARKERS.get(glacier_name, self.MARKERS['All'])
        return markers[index % len(markers)]
    
    @staticmethod
    def create_date_colormap(date_min: int = 1900, 
                            date_max: int = 2020,
                            num_colors: Optional[int] = None) -> tuple:
        """
        Crée une colormap discrète basée sur les dates.
        
        Parameters
        ----------
        date_min : int, optional
            Date minimale
        date_max : int, optional
            Date maximale
        num_colors : int, optional
            Nombre de couleurs. Si None, calculé automatiquement
            
        Returns
        -------
        cmap : ListedColormap
            Colormap discrète
        norm : Normalize
            Normalisation pour les dates
        """
        if num_colors is None:
            dates = np.arange(date_min, date_max, 10)
            num_colors = len(dates)
        
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        norm = Normalize(vmin=date_min, vmax=date_max)
        cmap = ListedColormap(colors)
        
        return cmap, norm
    
    @staticmethod
    def plot_kc_vs_discharge(Q_vals: np.ndarray, 
                            kc_vals: np.ndarray,
                            figsize: tuple = (6, 4)) -> plt.Figure:
        """
        Trace k_c en fonction du débit Q.
        
        Parameters
        ----------
        Q_vals : np.ndarray
            Valeurs de débit [m³/s]
        kc_vals : np.ndarray
            Valeurs de k_c correspondantes
        figsize : tuple, optional
            Taille de la figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.loglog(Q_vals, kc_vals, 'b-', lw=2)
        
        ax.set_xlabel("Débit Q [m³/s]")
        ax.set_ylabel(r"$k_c$")
        ax.set_title(r"Variation de $k_c$ en fonction du débit Q")
        
        # Configuration des graduations (1, 2, 5 × 10^k)
        min_exp = int(np.floor(np.log10(kc_vals.min())))
        max_exp = int(np.ceil(np.log10(kc_vals.max())))
        mantissas = np.array([1.0, 2.0, 5.0])
        major_ticks = np.sort(np.concatenate([
            mantissas * 10.0**e for e in range(min_exp-1, max_exp+2)
        ]))
        major_ticks = major_ticks[
            (major_ticks >= kc_vals.min()*0.9) & 
            (major_ticks <= kc_vals.max()*1.1)
        ]
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))
        
        # Graduations mineures (1-9 × 10^k)
        minor_ticks = np.sort(np.concatenate([
            np.arange(1, 10) * 10.0**e for e in range(min_exp-1, max_exp+2)
        ]))
        minor_ticks = minor_ticks[
            (minor_ticks >= kc_vals.min()*0.9) & 
            (minor_ticks <= kc_vals.max()*1.1)
        ]
        ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))
        
        # Formatter pour les graduations majeures
        def yfmt(y, _pos):
            e = int(np.floor(np.log10(y)))
            mant = y / (10.0**e)
            if np.isclose(mant, 1.0):
                return r"$10^{%d}$" % e
            else:
                return r"$%g\times10^{%d}$" % (mant, e)
        
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_friction_data(vel: np.ndarray,
                          tau: np.ndarray,
                          glacier_name: str = None,
                          color: str = 'blue',
                          marker: str = 'o',
                          ax: Optional[plt.Axes] = None,
                          label: Optional[str] = None) -> plt.Axes:
        """
        Trace les données de friction (vitesse vs contrainte).
        
        Parameters
        ----------
        vel : np.ndarray
            Vitesses de glissement [m/an]
        tau : np.ndarray
            Contraintes basales [Pa]
        glacier_name : str, optional
            Nom du glacier pour le titre
        color : str, optional
            Couleur des points
        marker : str, optional
            Style de marqueur
        ax : matplotlib.axes.Axes, optional
            Axes matplotlib. Si None, crée une nouvelle figure
        label : str, optional
            Label pour la légende
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes du graphique
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(vel, tau, color=color, marker=marker, 
                  s=50, alpha=0.7, label=label)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"Basal sliding velocity $(m \cdot yr^{-1})$")
        ax.set_ylabel(r"Basal shear stress (MPa)")
        
        if glacier_name:
            ax.set_title(f"Friction law - {glacier_name}")
        
        ax.grid(True, which='both', linestyle='dotted', alpha=0.5)
        
        if label:
            ax.legend()
        
        return ax
