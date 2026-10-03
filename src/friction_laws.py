"""
Friction laws and fitting functions.

- Weertman-type law: tau_b = (u_b / As)^(1/m)
- Lliboutry-type (regularised Coulomb) law, Gagliardini et al. (2007)
- Tsai-type law: tau_b = min((u_b / As)^(1/m), CN)
"""
import numpy as np
from scipy.optimize import curve_fit


def rmse(obs, pred):
    """Root mean square error."""
    return np.sqrt(np.mean((np.asarray(obs) - np.asarray(pred))**2))


# ============================================================================
# WEERTMAN-TYPE LAW
# ============================================================================

def power_law(u_bed, As, m=3):
    """Weertman-type friction law."""
    tau_b = (u_bed/As)**(1/m)
    return tau_b


def fit_weertman_law(vel, tau, initial_guess, velmin=0.01, velmax=220,
                     fix_m=None, fix_As=None):
    """
    Fit a Weertman-type friction law.

    Parameters
    ----------
    vel, tau : array
        Basal sliding velocity [m/yr] and basal shear stress [MPa].
    initial_guess : tuple
        Initial guess for (As, m).
    velmin, velmax : float
        Velocity range of the returned fitted curve [m/yr].
    fix_m, fix_As : float or None
        Fixed parameter value; fitted if None.

    Returns
    -------
    dict
        As, m: fitted values; vel_fit, tau_fit: fitted curve; rmse: fit error.
    """
    guess_As, guess_m = initial_guess

    if fix_m is not None and fix_As is None:
        popt, cov = curve_fit(lambda u, As: power_law(u, As, fix_m),
                              vel, tau, p0=[guess_As], maxfev=10000)
        m_fit = fix_m
        As_fit = popt[0]

    elif fix_m is None and fix_As is not None:
        popt, cov = curve_fit(lambda u, m: power_law(u, fix_As, m),
                              vel, tau, p0=[guess_m], maxfev=10000)
        m_fit = popt[0]
        As_fit = fix_As

    else:
        popt, cov = curve_fit(power_law, vel, tau, p0=initial_guess, maxfev=10000)
        As_fit, m_fit = popt

    tau_pred = power_law(vel, As_fit, m_fit)

    vel_fit = np.linspace(velmin, velmax, 1000)
    tau_fit = power_law(vel_fit, As_fit, m_fit)

    return {"As": As_fit, "m": m_fit,
            "rmse": rmse(tau, tau_pred), "vel_fit": vel_fit, "tau_fit": tau_fit}


# ============================================================================
# LLIBOUTRY-TYPE LAW
# ============================================================================

def cavitation_law(u_bed, CN, q, As, m=3):
    """
    Lliboutry-type friction law (Gagliardini et al., 2007).
    For q > 1, tau_b is capped at CN beyond its maximum (no rate weakening).
    """
    alpha = ((q-1)**(q-1))/(q**q)
    chi = u_bed / (As*(CN)**m)
    tau_b = (CN)*(chi/(1+alpha*chi**q))**(1/m)

    if q != 1:
        try:
            # Velocity at which tau_b reaches its maximum
            u_bed_max = (As * CN**m) * (1 / (alpha * (q-1)))**(1/q)
            tau_b = np.where(u_bed > u_bed_max, CN, tau_b)
        except ZeroDivisionError:
            pass

    return tau_b


def fit_lliboutry_law(vel, tau, initial_guess, velmin=0.01, velmax=220,
                      fix_CN=None, fix_q=None, fix_As=None, fix_m=None):
    """
    Fit a Lliboutry-type friction law.

    Only two configurations are implemented: q and m fixed (CN and As fitted),
    or all parameters free.

    Parameters
    ----------
    vel, tau : array
        Basal sliding velocity [m/yr] and basal shear stress [MPa].
    initial_guess : tuple
        Initial guess for (CN, q, As, m).
    velmin, velmax : float
        Velocity range of the returned fitted curve [m/yr].
    fix_CN, fix_q, fix_As, fix_m : float or None
        Fixed parameter value; fitted if None.

    Returns
    -------
    dict
        CN, q, As, m: fitted values; vel_fit, tau_fit: fitted curve; rmse: fit error.
    """
    guess_CN, guess_q, guess_As, guess_m = initial_guess

    if fix_CN is None and fix_q is not None and fix_As is None and fix_m is not None:
        popt, cov = curve_fit(lambda u, CN, As: cavitation_law(u, CN, fix_q, As, fix_m),
                              vel, tau, p0=[guess_CN, guess_As], maxfev=10000)
        q_fit, m_fit = fix_q, fix_m
        CN_fit, As_fit = popt

    else:
        popt, cov = curve_fit(cavitation_law, vel, tau, p0=initial_guess, maxfev=10000)
        CN_fit, q_fit, As_fit, m_fit = popt

    tau_pred = cavitation_law(vel, CN_fit, q_fit, As_fit, m_fit)

    vel_fit = np.linspace(velmin, velmax, 1000)
    tau_fit = cavitation_law(vel_fit, CN_fit, q_fit, As_fit, m_fit)

    return {"CN": CN_fit, "q": q_fit, "As": As_fit, "m": m_fit,
            "rmse": rmse(tau, tau_pred), "vel_fit": vel_fit, "tau_fit": tau_fit}


# ============================================================================
# TSAI-TYPE LAW
# ============================================================================

def tsai_law(u_bed, CN, As, m):
    """Tsai-type friction law: Weertman law capped at CN."""
    tau_b = np.minimum((u_bed / As)**(1/m), CN)
    return tau_b


def fit_tsai_law(vel, tau, initial_guess, velmin=0.01, velmax=220,
                 fix_CN=None, fix_As=None, fix_m=None):
    """
    Fit a Tsai-type friction law.

    Only two configurations are implemented: all parameters free,
    or CN fixed (As and m fitted).

    Parameters
    ----------
    vel, tau : array
        Basal sliding velocity [m/yr] and basal shear stress [MPa].
    initial_guess : tuple
        Initial guess for (CN, As, m).
    velmin, velmax : float
        Velocity range of the returned fitted curve [m/yr].
    fix_CN, fix_As, fix_m : float or None
        Fixed parameter value; fitted if None.

    Returns
    -------
    dict
        CN, As, m: fitted values; vel_fit, tau_fit: fitted curve; rmse: fit error.
    """
    CN0, As0, m0 = initial_guess

    if fix_CN is None and fix_As is None and fix_m is None:
        popt, cov = curve_fit(lambda u, CN, As, m: tsai_law(u, CN, As, m),
                              vel, tau, p0=[CN0, As0, m0], maxfev=10000)
        CN_fit, As_fit, m_fit = popt

    elif fix_CN is not None and fix_As is None and fix_m is None:
        popt, cov = curve_fit(lambda u, As, m: tsai_law(u, fix_CN, As, m),
                              vel, tau, p0=[As0, m0], maxfev=10000)
        CN_fit, As_fit, m_fit = fix_CN, popt[0], popt[1]

    else:
        raise ValueError("This fixed-parameter configuration is not implemented.")

    tau_pred = tsai_law(vel, CN_fit, As_fit, m_fit)

    vel_fit = np.linspace(velmin, velmax, 1000)
    tau_fit = tsai_law(vel_fit, CN_fit, As_fit, m_fit)

    return {"CN": CN_fit, "As": As_fit, "m": m_fit,
            "rmse": rmse(tau, tau_pred), "vel_fit": vel_fit, "tau_fit": tau_fit}


# ============================================================================
# NORMALISED FRICTION LAW
# ============================================================================

def calcul_normalised_friction_law(vel, tau, CN, As, m=3):
    """Normalise velocity by As * CN^m and stress by CN (raised to the power m)."""
    vel_norm = vel / (As*(CN**m))
    tau_norm = (tau / CN)**m
    return vel_norm, tau_norm


def scaled_friction_law(u_bed, q):
    """
    Normalised Lliboutry-type law.
    For q > 1, capped at 1 beyond its maximum (no rate weakening).
    """
    if q == 1:
        tau_b = (u_bed/(1+u_bed))
    else:
        alpha = ((q-1)**(q-1))/(q**q)
        tau_b = (u_bed/(1+alpha*u_bed**q))
        u_bed_max = (1 / (alpha * (q-1)))**(1/q)
        tau_b = np.where(u_bed > u_bed_max, 1, tau_b)
    return tau_b