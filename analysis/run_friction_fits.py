import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils import GLACIERS
from friction_laws import fit_lliboutry_law, fit_tsai_law, fit_weertman_law

script_dir = Path(__file__).resolve().parent

out_dir = script_dir / ".." / "data" / "processed_timeseries" / "friction_fits"
out_dir.mkdir(exist_ok=True, parents=True)

def compile_vel_tau_timeseries(glacier_key, stake):
    if stake == "Wheel": # ignoring wheel data
        return None, None

    file = GLACIERS[glacier_key]["all_data"][stake]
            
    if stake in ["C", "ech"]: # recent data are unreliable at StSo C and MDG ech
        df = pd.read_csv(file)[lambda df: (df['date'] < 2015)]
    else:
        df = pd.read_csv(file)    
    
    df = df[np.isfinite(df["date"])].copy()

    if df['date'].isna().any():
        print(f"Warning: {glacier_key} {stake} contient des NaN dans 'date'.")

    # interpolate elmer/ice values to have a complete timeseries
    df['tau_b_elmer_interp'] = df.set_index('date')["tau_b_elmer"].interpolate(method='index').tolist()
    df['u_def_elmer_interp'] = df.set_index('date')["u_def_elmer"].interpolate(method='index').tolist()
    df['u_bed_elmer_interp'] = df["velocity"] - df['u_def_elmer_interp']
    
    if glacier_key == "StSo":
        vel, tau = df["u_bed_elmer_interp"], df["tau_b_elmer_interp"]
    else:
        vel, tau = df['obs_u_bed'], df['obs_tau_b']
    
    if stake=="A4": # Corbassière A4 very recent data are uninterpretable
        velmin=4
    else:
        velmin=2

    valid_indices = np.isfinite(vel) & np.isfinite(tau) & (vel > velmin)
    vel, tau = vel[valid_indices], tau[valid_indices]

    return vel, tau



def run_all_fits():
    results = {}

    guess_m, guess_As, guess_q = 3, 20000, 1

    fit_rows = []

    for glacier_key, glacier_data in GLACIERS.items():
        results[glacier_key] = {}

        for stake, file in glacier_data["all_data"].items():

            if stake == "Wheel": # ignoring wheel data
                continue
            
            vel, tau = compile_vel_tau_timeseries(glacier_key, stake)

            if tau.empty:
                print(f"No data available for {stake} on {glacier_key}")
                guess_CN = None  # ou une valeur par défaut raisonnable
            else:
                guess_CN = np.max(tau.values)

            # controls which model by glacier/stake
            if glacier_key in ["All", "Arg", "Cor", "Gie", "GB", "MDG", "StSo"]:
                fit = fit_lliboutry_law(
                        vel, tau, [guess_CN, guess_q, guess_As, guess_m], 
                        fix_m=3, fix_q=1)
                fit_type="Lliboutry"
                
            elif (glacier_key == "Geb") & (stake == "sup"):
                fit = fit_tsai_law(vel, tau, [guess_CN, guess_As, guess_m])
                fit_type="Tsai"
            elif (glacier_key == "Geb") & (stake == "ss"): # no ancient data to constrain the plateau so we put an artificial threshold for the fit
                fit = fit_tsai_law(vel, tau, [guess_CN, guess_As, guess_m], fix_CN=0.05, velmax=np.max(vel))
                fit_type="Tsai"
            else:
                fit = fit_weertman_law(vel, tau, [guess_As, guess_m], fix_m=3)
                fit_type="Weertman"

            results[glacier_key][stake] = fit

            # saving friction law timeseries for each stake
            fits_df = pd.DataFrame({"vel_fit" : fit.get("vel_fit"),
                                    "tau_fit" : fit.get("tau_fit")})
            fits_df.to_csv(out_dir / f"{glacier_key}_{stake}_friclaw_ts.csv", index=False)

            fit_rows.append({
                "glacier": glacier_key,
                "stake": stake,
                "fit_type": fit_type,
                "CN": fit.get("CN", None),
                "q": fit.get("q", None),
                "As": fit.get("As", None),
                "m": fit.get("m", None)})
    
    # saving best fit friction law parameters for all stakes
    params_df = pd.DataFrame(fit_rows)
    params_df.to_csv(out_dir / "friction_fit_params.csv", index=False)
            
    return results

if __name__=="__main__":
    fits = run_all_fits()