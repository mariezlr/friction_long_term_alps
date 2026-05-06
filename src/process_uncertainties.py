import re
from process_timeseries import *
from utils import GLACIERS, geom_data_dir, proc_data_dir
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIG FIXE POUR ARGENTIÈRE PROFIL 4
# ============================================================================
script_dir = Path(__file__).resolve().parent

UNCERTAINTIES_DIR = Path(script_dir / '..' / 'data' / 'uncertainties')

# ============================================================================
# LISTER TOUS LES DOSSIERS DE SIMULATION
# ============================================================================

def list_uncertainty_runs():
    """
    Retourne une liste de tuples (run_name, input_dir, run_type)
    
    run_type parmi : 'As_A', 'bedrock', 'As_spatial'
    """
    runs = []

    for d in sorted(UNCERTAINTIES_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # As{val}_A{val} -> 49 combinaisons
        if re.match(r'^As\d+_A[\d.e+-]+$', name):
            runs.append((name, d, 'As_A'))

        # B1, B2, B3, B4 -> perturbations bedrock
        elif re.match(r'^B\d+$', name):
            runs.append((name, d, 'bedrock'))

        # as_0, as_1, as_2, as_3 -> As spatial
        elif re.match(r'^as_\d+$', name):
            runs.append((name, d, 'As_spatial'))

    print(f"  → {len(runs)} runs trouvés")
    for name, d, rtype in runs:
        print(f"     [{rtype:12s}] {name}")
    return runs


def process_all_uncertainty_runs():

    runs = list_uncertainty_runs()

    for run_name, input_dir, run_type in runs:
        print(f"\n--- {run_name} [{run_type}] ---")

        m = 3

        match = re.match(r'^As(\d+)_A([\d.e+-]+)$', run_name)
        if match:
            As = float(match.group(1))
            C = As ** (-1/m)

        else:
            # bedrock, as_spatial : garder m et C de référence
            C = GLACIERS["Arg"]['mval_Cval'][1][1]

        try:
            config = GLACIERS["Arg"]

            df_final = process_glacier_stake(
                "Arg",
                "4",
                config,
                m=m,
                C=C,
                Arg_simu=run_name
            )

            if df_final is None:
                continue

            outfile = UNCERTAINTIES_DIR / f"timeseries_{run_name}.csv"
            df_final.to_csv(outfile, index=False)

            print(f"Saved {outfile.name}")

        except Exception as e:
            print(f"Error : {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    process_all_uncertainty_runs()