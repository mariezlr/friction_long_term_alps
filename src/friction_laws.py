import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


script_dir = Path(__file__).resolve().parent


### ----- Compute best_fit friction law for each glacier -----

#### 1st case: Weertman-type law

def power_law(u_bed, As, m=3):
    tau_b = (u_bed/As)**(1/m)
    return tau_b

def fit_weertman_law(vel, tau, initial_guess, velmin = 2, velmax = 220, 
                     fix_m=None, fix_As=None):
    """
    Fits a Weertman-type friction law to velocity/shear-stress data.

    Parameters
    ----------
    vel, tau : array
        Observed sliding velocity (m/yr) & basal shear stress (MPa).
    initial_guess : tuple
        Initial guess for (m, As) or (As) if m fixed.
    friclaw : function
        Friction law function of form f(vel, m, As) or f(vel, As) for fixed m.
    fix_As, fix_m : float or None
        Fix a parameter if provided, otherwise it is fitted.

    Returns
    -------
    dict with:
        As, m               fitted values
        vel_fit, tau_fit    fitted curves ready for plotting
        rmse                goodness of fit
    """
    
    guess_As, guess_m = initial_guess

    # Fit cases
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

    else:  # both free
        popt, cov = curve_fit(power_law, vel, tau, p0=initial_guess, maxfev=10000)
        As_fit, m_fit = popt

    # predictions + RMSE
    tau_pred = power_law(vel, As_fit, m_fit)
    rmse = np.sqrt(mean_squared_error(tau, tau_pred))

    vel_fit = np.linspace(velmin, velmax, 100)
    tau_fit = power_law(vel_fit, As_fit, m_fit)

    return {"As": As_fit, "m": m_fit,
            "rmse": rmse, "vel_fit": vel_fit, "tau_fit": tau_fit}




#### 2nd case: Lliboutry-type law

def cavitation_law(u_bed, CN, q, As, m=3): # no rate weakening
    alpha = ((q-1)**(q-1))/(q**q)
    chi = u_bed /(As*(CN)**m)
    tau_b = (CN)*(chi/(1+alpha*chi**q))**(1/m)
    
    if q != 1:  # Avoid division by zero error
        try:
            # Find u_bed_max so that tau_b is maximal and define an asymptote for this value
            u_bed_max = (As * CN**m) * (1 / (alpha * (q-1)))**(1/q)
            tau_b = np.where(u_bed > u_bed_max, CN, tau_b)
        except ZeroDivisionError:
            pass

    return tau_b


def fit_lliboutry_law(vel, tau, initial_guess, velmin = 2, velmax = 220,
                  fix_CN=None, fix_q=None, fix_As=None, fix_m=None,):
    """
    Fits Lliboutry friction law for basal sliding, allowing free or fixed parameters.

    Parameters
    ----------
    vel, tau : array
        Observed sliding velocity (m/yr) & basal shear stress (MPa).
    initial_guess : tuple
        Initial guess for (CN, q, As, m).
    friclaw : function
        Lliboutry friction function f(vel, CN, q, As, m).
    fix_CN, fix_q, fix_As, fix_m : float or None
        Fix a parameter if provided, otherwise it is fitted.

    Returns
    -------
    dict with:
        CN, q, As, m        fitted values
        vel_fit, tau_fit    fitted curves ready for plotting
        rmse                goodness of fit
    """

    guess_CN, guess_q, guess_As, guess_m = initial_guess

    # Fit cases
    if fix_CN is None and fix_q is not None and fix_As is None and fix_m is not None:
        popt, cov = curve_fit(lambda u, CN, As: cavitation_law(u, CN, fix_q, As, fix_m),
                              vel, tau, p0=[guess_CN, guess_As], maxfev=10000)
        q_fit, m_fit = fix_q, fix_m
        CN_fit, As_fit = popt

    else:  # all free
        popt, cov = curve_fit(cavitation_law, vel, tau, p0=initial_guess, maxfev=10000)
        As_fit, m_fit = popt

    # Predictions and error
    tau_pred = cavitation_law(vel, CN_fit, q_fit, As_fit, m_fit)
    rmse = np.sqrt(mean_squared_error(tau, tau_pred))

    vel_fit = np.linspace(velmin, velmax, 100)
    tau_fit = cavitation_law(vel_fit, CN_fit, q_fit, As_fit, m_fit)


    return {"CN": CN_fit, "q": q_fit, "As": As_fit, "m": m_fit, 
            "rmse": rmse, "vel_fit": vel_fit, "tau_fit": tau_fit}



#### 3rd case: Tsai-type friction law

def tsai_law(u_bed, CN, As, m):
    tau_b = np.minimum((u_bed / As)**(1/m), CN)
    return tau_b


def fit_tsai_law(vel, tau, initial_guess, velmin = 2, velmax = 220,
                 fix_CN=None, fix_As=None, fix_m=None):
    """
    Fit the Tsai basal friction law tau_b(u) to observed (velocity, stress) data.

    Parameters
    ----------
    vel : array
        Basal sliding velocity (m/yr).
    tau : array
        Basal shear stress (MPa).
    initial_guess : tuple (CN0, As0, m0)
        Initial guess for fitting parameters.
    friclaw : function
        Tsai Friction law function f(vel, CN, As, m).
    fix_CN, fix_As, fix_m : float or None
        Fix a parameter if provided, otherwise it is fitted.

    Returns
    -------
    CN_fit, As_fit, m_fit : floats
        Best-fit friction law parameters.
    rmse : float
        Root Mean Square Error between data and model.

    dict with:
        CN, As, m           fitted values
        vel_fit, tau_fit    fitted curves ready for plotting
        rmse                goodness of fit
    """

    CN0, As0, m0 = initial_guess

    # Handle fixed parameters combinations
    if fix_CN is None and fix_As is None and fix_m is None:
        popt, cov = curve_fit(lambda u, CN, As, m: tsai_law(u, CN, As, m),
                                vel, tau, p0=[CN0, As0, m0], maxfev=10000)
        CN_fit, As_fit, m_fit = popt

    elif fix_CN is not None and fix_As is None and fix_m is None:
        popt, cov = curve_fit(lambda u, As, m: tsai_law(u, fix_CN, As, m),
                                vel, tau, p0=[As0, m0], maxfev=10000)
        CN_fit, As_fit, m_fit = fix_CN, popt[0], popt[1]

    else:
        raise ValueError("This fixed-parameter configuration is not implemented yet.")

    # Predictions and error
    tau_pred = tsai_law(vel, CN_fit, As_fit, m_fit)
    rmse = np.sqrt(mean_squared_error(tau, tau_pred))

    vel_fit = np.linspace(velmin, velmax, 100)
    tau_fit = tsai_law(vel_fit, CN_fit, As_fit, m_fit)

    return {"CN": CN_fit, "As": As_fit, "m": m_fit, 
            "rmse": rmse, "vel_fit": vel_fit, "tau_fit": tau_fit}






### ----- Normalised friction law -----

def calcul_normalised_friction_law(vel, tau, CN, As, m=3, velmin=2):
    vel_norm = vel / (As*(CN**m))
    tau_norm = (tau / CN)**m
    return vel_norm, tau_norm


## Horizontal asymptote to avoid rate-weakening
def scaled_friction_law(u_bed, q):
    if q==1:
        tau_b = (u_bed/(1+u_bed))
    else:
        alpha = ((q-1)**(q-1))/(q**q)
        tau_b = (u_bed/(1+alpha*u_bed**q))
        u_bed_max = (1 / (alpha * (q-1)))**(1/q)
        tau_b = np.where(u_bed > u_bed_max, 1, tau_b)
    return tau_b