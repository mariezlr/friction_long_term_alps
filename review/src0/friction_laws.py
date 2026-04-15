"""
Lois de friction pour l'analyse glaciaire.
Contient les implémentations des lois de Weertman et de cavitation.
"""

import numpy as np
from scipy.optimize import newton


def power_law(u_bed: np.ndarray, m: float, As: float) -> np.ndarray:
    """
    Loi de puissance générale (Weertman).
    
    Parameters
    ----------
    u_bed : np.ndarray
        Vitesse de glissement basal [m/an]
    m : float
        Exposant de la loi
    As : float
        Paramètre de friction
        
    Returns
    -------
    np.ndarray
        Contrainte basale de cisaillement [Pa]
    """
    return (u_bed / As) ** (1 / m)


def power_law_m3(u_bed: np.ndarray, As: float) -> np.ndarray:
    """
    Loi de puissance avec m=3 (cas particulier de Weertman).
    
    Parameters
    ----------
    u_bed : np.ndarray
        Vitesse de glissement basal [m/an]
    As : float
        Paramètre de friction
        
    Returns
    -------
    np.ndarray
        Contrainte basale de cisaillement [Pa]
    """
    return (u_bed / As) ** (1 / 3)


def inverse_power_law(tau_b: np.ndarray, m: float, As: float) -> np.ndarray:
    """
    Inverse de la loi de puissance.
    
    Parameters
    ----------
    tau_b : np.ndarray
        Contrainte basale de cisaillement [Pa]
    m : float
        Exposant de la loi
    As : float
        Paramètre de friction
        
    Returns
    -------
    np.ndarray
        Vitesse de glissement basal [m/an]
    """
    return As * (tau_b) ** m


def cavitation_law(u_bed: np.ndarray, CN: float, q: float, 
                   As: float, m: float = 3) -> np.ndarray:
    """
    Loi de cavitation modifiée sans affaiblissement (weakening).
    
    Parameters
    ----------
    u_bed : np.ndarray
        Vitesse de glissement basal [m/an]
    CN : float
        Pression de contact effective [Pa]
    q : float
        Exposant de cavitation
    As : float
        Paramètre de friction
    m : float, optional
        Exposant de la loi de puissance, par défaut 3
        
    Returns
    -------
    np.ndarray
        Contrainte basale de cisaillement [Pa]
    """
    alpha = ((q - 1) ** (q - 1)) / (q ** q)
    chi = u_bed / (As * (CN) ** m)
    tau_b = (CN) * (chi / (1 + alpha * chi ** q)) ** (1 / m)
    
    return tau_b


def inverse_cavitation_law(tau_b: float, CN: float, q: float, 
                          As: float, m: float, initial_guess: float = 1) -> float:
    """
    Inverse de la loi de cavitation (méthode de Newton-Raphson).
    
    Parameters
    ----------
    tau_b : float
        Contrainte basale de cisaillement [Pa]
    CN : float
        Pression de contact effective [Pa]
    q : float
        Exposant de cavitation
    As : float
        Paramètre de friction
    m : float
        Exposant de la loi de puissance
    initial_guess : float, optional
        Estimation initiale pour la méthode de Newton, par défaut 1
        
    Returns
    -------
    float
        Vitesse de glissement basal [m/an]
    """
    def fct_to_inverse(u_bed):
        return cavitation_law(u_bed, CN, q, As, m) - tau_b
    
    u_bed = newton(fct_to_inverse, initial_guess)
    return u_bed


def compute_kc(Q: float, CN: float = 0.29, C: float = 0.4, 
               rho_i: float = 910.0, L: float = 3.35e5, 
               A_c: float = None, g: float = 9.81,
               alpha_c: float = 1.25, n: float = 3.0) -> float:
    """
    Calcule le coefficient k_c pour une configuration donnée.
    
    Cette fonction utilise la relation :
    k_c = ((1e6 * CN / C)^(alpha_c * n)) * ((rho_i * L * A_c)^alpha_c) 
          * ((rho_i * g)^(-(2*alpha_c + 1)/2)) * (Q^(-(alpha_c - 1)))
    
    Parameters
    ----------
    Q : float
        Débit [m³/s]
    CN : float, optional
        Coefficient CN [MPa], par défaut 0.29
    C : float, optional
        Coefficient de friction, par défaut 0.4
    rho_i : float, optional
        Densité de la glace [kg/m³], par défaut 910.0
    L : float, optional
        Chaleur latente [J/kg], par défaut 3.35e5
    A_c : float, optional
        Paramètre rhéologique. Si None, calculé automatiquement
    g : float, optional
        Gravité [m/s²], par défaut 9.81
    alpha_c : float, optional
        Exposant, par défaut 1.25
    n : float, optional
        Exposant de Glen, par défaut 3.0
        
    Returns
    -------
    float
        Coefficient k_c
    """
    if A_c is None:
        A = 2.4e-24
        A_c = 2 * A * n ** (-n)
    
    return (((1e6 * CN / C) ** (alpha_c * n)) 
            * ((rho_i * L * A_c) ** alpha_c) 
            * ((rho_i * g) ** (-(2 * alpha_c + 1) / 2)) 
            * (Q ** (-(alpha_c - 1))))
