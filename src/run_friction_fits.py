"""
Fit friction laws to the basal velocity and shear stress timeseries.

- run_all_fits: fit each stake of each glacier and save the fitted curves
  and parameters in processed_timeseries/mw{1/m}/friction_fits/.
- run_uncertainty_fits: fit each sensitivity experiment at Argentière Profile 4
  and save the results in processed_timeseries/uncertainty_fits/.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import GLACIERS, proc_data_dir
from friction_laws import fit_lliboutry_law, fit_tsai_law
import warnings
import logging
from scipy.optimize import OptimizeWarning

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Silence scipy fitting warnings
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

script_dir = Path(__file__).resolve().parent


def compile_vel_tau_timeseries(glacier_key, stake, m=3):
    """
    Read the basal velocity and shear stress timeseries of a stake.

    Returns
    -------
    date, vel, tau : pd.Series, or (None, None, None) if no data
    """
    if stake == "Wheel":  # comparison point only, not studied
        return None, None, None

    file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv"

    if not file.exists():
        print(f"[WARNING] missing file: {file}")
        return None, None, None

    # Exclude recent low-velocity periods, where data are unreliable
    if stake in ["A4"]:
        df = pd.read_csv(file)[lambda df: (df['date'] < 2010)]
    elif glacier_key == "GB":
        df = pd.read_csv(file)[lambda df: (df['date'] < 1990)]
    else:
        df = pd.read_csv(file)

    df = df[np.isfinite(df["date"])].copy()

    date, vel, tau = df["date"], df['obs_u_bed'], df['obs_tau_b']
    velmin = 0.01

    valid_indices = np.isfinite(vel) & np.isfinite(tau) & (vel > velmin)
    date, vel, tau = date[valid_indices], vel[valid_indices], tau[valid_indices]

    return date, vel, tau


def run_all_fits(m=3):
    """Fit the friction law at each stake and save curves and parameters."""
    out_dir = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits"
    out_dir.mkdir(exist_ok=True, parents=True)
    results = {}

    guess_m, guess_As, guess_q = 3, 20000, 1

    fit_rows = []

    for glacier_key, glacier_data in GLACIERS.items():
        results[glacier_key] = {}

        for stake in glacier_data['xy_coords'].keys():

            if stake == "Wheel":  # comparison point only, not studied
                continue

            date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)

            if tau is None or len(tau) == 0:
                print(f"No data available for {glacier_key} {stake}")
                continue
            else:
                guess_CN = np.max(tau.values)

            # Friction law used for each glacier/stake
            if glacier_key in ["All", "Arg", "Cor", "Gie", "GB", "MDG", "StSo"]:
                fit = fit_lliboutry_law(
                    vel, tau, [guess_CN, guess_q, guess_As, guess_m], fix_m=3, fix_q=1)
                fit_type = "Lliboutry"

            elif (glacier_key == "Geb") & (stake == "sup"):
                fit = fit_tsai_law(vel, tau, [guess_CN, guess_As, guess_m])
                fit_type = "Tsai"

            elif (glacier_key == "Geb") & (stake == "ss"):
                # No early data to constrain the plateau: CN is prescribed
                fit = fit_tsai_law(vel, tau, [guess_CN, guess_As, guess_m],
                                   fix_CN=0.05, velmax=np.max(vel))
                fit_type = "Tsai"

            else:
                print(f"No friction law defined for {glacier_key} {stake}")
                continue

            results[glacier_key][stake] = fit

            # Fitted friction law curve
            fits_df = pd.DataFrame({
                "vel_fit": fit.get("vel_fit"),
                "tau_fit": fit.get("tau_fit")})
            fits_df.to_csv(out_dir / f"{glacier_key}_{stake}_friclaw_ts.csv", index=False)

            fit_rows.append({
                "glacier": glacier_key,
                "stake": stake,
                "fit_type": fit_type,
                "CN": fit.get("CN", None),
                "q": fit.get("q", None),
                "As": fit.get("As", None),
                "m": fit.get("m", None)})

    # Best-fit parameters for all stakes
    params_df = pd.DataFrame(fit_rows)
    params_df.to_csv(out_dir / "friction_fit_params.csv", index=False)

    return results


def run_uncertainty_fits(runs_by_type, velmin=0.01):
    """
    Fit the friction law to each sensitivity experiment.

    Parameters
    ----------
    runs_by_type : dict
        Maps a perturbation type to a {run_name: DataFrame} dict.
    velmin : float
        Minimum basal velocity retained in the fit [m/yr].

    Returns
    -------
    params_df : pd.DataFrame
        Fitted parameters for each run.
    summary : pd.DataFrame
        Mean and standard deviation of As and CN per perturbation type.
    """
    out_dir = proc_data_dir / "uncertainty_fits"
    out_dir.mkdir(exist_ok=True, parents=True)

    guess_m, guess_As, guess_q = 3, 20000, 1
    rows = []

    for ptype, runs in runs_by_type.items():
        for name, df in runs.items():
            vel, tau = df["obs_u_bed"], df["obs_tau_b"]
            ok = np.isfinite(vel) & np.isfinite(tau) & (vel > velmin)
            vel, tau = vel[ok], tau[ok]
            if len(vel) == 0:
                logger.info(f"No valid data for {ptype} {name}")
                continue

            fit = fit_lliboutry_law(vel, tau, [np.max(tau.values), guess_q, guess_As, guess_m],
                                    fix_m=3, fix_q=1)

            pd.DataFrame({"vel_fit": fit.get("vel_fit"),
                          "tau_fit": fit.get("tau_fit")}
                         ).to_csv(out_dir / f"{ptype}_{name}_friclaw_ts.csv", index=False)

            rows.append({"perturbation": ptype, "run": name,
                         "CN": fit.get("CN"), "q": fit.get("q"),
                         "As": fit.get("As"), "m": fit.get("m")})

    params_df = pd.DataFrame(rows)
    params_df.to_csv(out_dir / "uncertainty_fit_params.csv", index=False)

    # Mean and standard deviation of the fitted parameters per perturbation type
    summary = params_df.groupby("perturbation")[["As", "CN"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "uncertainty_fit_summary.csv")
    print(summary)

    return params_df, summary


if __name__ == "__main__":
    run_all_fits(1)
    run_all_fits(3)
    run_all_fits(6)