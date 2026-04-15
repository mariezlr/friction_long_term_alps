"""
Module pour l'ajustement des lois de friction aux données observées.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from typing import Tuple, Optional, Callable

from .friction_laws import power_law, power_law_m3


class FrictionFitter:
    """
    Classe pour ajuster les lois de friction aux données de vitesse et contrainte.
    """
    
    def __init__(self, x_ticks: Optional[list] = None, y_ticks: Optional[list] = None):
        """
        Initialise le fitter avec des graduations personnalisées pour les graphiques.
        
        Parameters
        ----------
        x_ticks : list, optional
            Graduations de l'axe x
        y_ticks : list, optional
            Graduations de l'axe y
        """
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
    
    def fit_weertman(self, 
                     vel: np.ndarray, 
                     tau: np.ndarray, 
                     initial_guess: Tuple[float, float],
                     friclaw: Callable = power_law,
                     fix_m: Optional[float] = None,
                     fix_As: Optional[float] = None,
                     velmin: Optional[float] = None,
                     velmax: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Ajuste une loi de Weertman aux données.
        
        Parameters
        ----------
        vel : np.ndarray
            Vitesses de glissement basal observées [m/an]
        tau : np.ndarray
            Contraintes basales observées [Pa]
        initial_guess : tuple
            Estimation initiale (m, As)
        friclaw : callable, optional
            Fonction de la loi de friction, par défaut power_law
        fix_m : float, optional
            Si fourni, fixe la valeur de m
        fix_As : float, optional
            Si fourni, fixe la valeur de As
        velmin : float, optional
            Vitesse minimale pour le tracé
        velmax : float, optional
            Vitesse maximale pour le tracé
            
        Returns
        -------
        m_fit : float
            Exposant ajusté
        As_fit : float
            Paramètre de friction ajusté
        rmse : float
            Erreur quadratique moyenne
        """
        guess_m, guess_As = initial_guess
        
        # Ajustement selon les contraintes
        if friclaw == power_law:
            if fix_m is not None and fix_As is None:
                best_fit_params, _ = curve_fit(
                    lambda u, As: friclaw(u, fix_m, As), 
                    vel, tau, p0=[guess_As], maxfev=10000
                )
                As_fit = best_fit_params[0]
                m_fit = fix_m
            elif fix_As is not None and fix_m is None:
                best_fit_params, _ = curve_fit(
                    lambda u, m: friclaw(u, m, fix_As), 
                    vel, tau, p0=[guess_m], maxfev=10000
                )
                m_fit = best_fit_params[0]
                As_fit = fix_As
            else:
                best_fit_params, _ = curve_fit(
                    friclaw, vel, tau, p0=initial_guess, maxfev=10000
                )
                m_fit, As_fit = best_fit_params
        elif friclaw == power_law_m3:
            best_fit_params, _ = curve_fit(
                friclaw, vel, tau, p0=[guess_As], maxfev=10000
            )
            m_fit, As_fit = 3, best_fit_params[0]
        else:
            raise ValueError("Loi de friction non reconnue")
        
        # Calcul de la RMSE
        if friclaw == power_law:
            tau_pred = friclaw(vel, m_fit, As_fit)
        elif friclaw == power_law_m3:
            tau_pred = friclaw(vel, As_fit)
        
        rmse = np.sqrt(mean_squared_error(tau, tau_pred))
        
        return m_fit, As_fit, rmse
    
    def plot_fit(self,
                 vel: np.ndarray,
                 tau: np.ndarray,
                 m_fit: float,
                 As_fit: float,
                 friclaw: Callable = power_law,
                 ax: Optional[plt.Axes] = None,
                 color: str = 'blue',
                 alpha: float = 1.0,
                 linestyle: str = '-',
                 linewidth: float = 1,
                 label: Optional[str] = None,
                 velmin: Optional[float] = None,
                 velmax: Optional[float] = None,
                 show_legend: bool = True) -> plt.Axes:
        """
        Trace l'ajustement de la loi de friction.
        
        Parameters
        ----------
        vel : np.ndarray
            Vitesses observées
        tau : np.ndarray
            Contraintes observées
        m_fit : float
            Exposant ajusté
        As_fit : float
            Paramètre de friction ajusté
        friclaw : callable, optional
            Loi de friction, par défaut power_law
        ax : matplotlib.axes.Axes, optional
            Axes matplotlib. Si None, crée une nouvelle figure
        color : str, optional
            Couleur de la courbe
        alpha : float, optional
            Transparence
        linestyle : str, optional
            Style de ligne
        linewidth : float, optional
            Épaisseur de ligne
        label : str, optional
            Label pour la légende
        velmin : float, optional
            Vitesse minimale pour le tracé
        velmax : float, optional
            Vitesse maximale pour le tracé
        show_legend : bool, optional
            Afficher la légende
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes du graphique
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plage de vitesses pour le tracé
        if velmin is None:
            velmin = min(vel)
        if velmax is None:
            velmax = max(vel)
        
        vel_fit = np.linspace(velmin, velmax, 100)
        
        if friclaw == power_law:
            tau_fit = friclaw(vel_fit, m_fit, As_fit)
        elif friclaw == power_law_m3:
            tau_fit = friclaw(vel_fit, As_fit)
        
        # Tracé
        if label is None:
            label = f'm={m_fit:.2f}, As={As_fit:.1e}'
        
        ax.plot(vel_fit, tau_fit, color=color, alpha=alpha, 
                linestyle=linestyle, linewidth=linewidth, label=label)
        
        # Configuration des axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if self.x_ticks is not None:
            ax.set_xticks([x for x in self.x_ticks 
                          if ax.get_xlim()[0] <= x <= ax.get_xlim()[-1]])
        if self.y_ticks is not None:
            ax.set_yticks([y for y in self.y_ticks 
                          if ax.get_ylim()[0] <= y <= ax.get_ylim()[-1]])
        
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax.grid(which='both', linestyle='dotted')
        
        if show_legend:
            ax.legend()
            ax.set_xlabel(r"Basal sliding velocity $(m \cdot yr^{-1})$")
            ax.set_ylabel(r"Basal shear stress (MPa)")
        
        return ax
