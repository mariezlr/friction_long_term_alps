"""
Build basal shear stress and sliding velocity timeseries for the sensitivity
experiments at Argentière Glacier, Profile 4.

Each experiment is stored in data/uncertainties/{run_name}/ and processed with
the same workflow as the reference simulations (process_timeseries.py).
Outputs are saved as data/uncertainties/timeseries_{run_name}.csv.
"""
import re
import traceback
from pathlib import Path

from process_timeseries import process_glacier_stake
from utils import GLACIERS

script_dir = Path(__file__).resolve().parent

UNCERTAINTIES_DIR = Path(script_dir / '..' / 'data' / 'uncertainties')

GLACIER_NAME = "Arg"
STAKE_NAME = "4"
M_REF = 3


def list_uncertainty_runs():
    """
    Return a list of (run_name, input_dir, run_type) tuples,
    with run_type in {'As_A', 'bedrock', 'As_spatial'}.
    """
    runs = []

    for d in sorted(UNCERTAINTIES_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # As{val}_A{val}: uniform As and A perturbations
        if re.match(r'^As\d+_A[\d.e+-]+$', name):
            runs.append((name, d, 'As_A'))

        # B1-B4: bedrock DEM perturbations
        elif re.match(r'^B\d+$', name):
            runs.append((name, d, 'bedrock'))

        # as_0-as_3: spatially variable As fields
        elif re.match(r'^as_\d+$', name):
            runs.append((name, d, 'As_spatial'))

    print(f"{len(runs)} runs found")
    for name, d, rtype in runs:
        print(f"  [{rtype:10s}] {name}")
    return runs


def process_all_uncertainty_runs():
    """Process all sensitivity experiments."""
    runs = list_uncertainty_runs()
    config = GLACIERS[GLACIER_NAME]

    for run_name, input_dir, run_type in runs:
        print(f"\n{run_name} [{run_type}]")

        match = re.match(r'^As(\d+)_A([\d.e+-]+)$', run_name)
        if match:
            As = float(match.group(1))
            C = As ** (-1 / M_REF)
        else:
            # Reference C for bedrock and spatial As runs
            C = config['mval_Cval'][1][1]

        try:
            outfile = UNCERTAINTIES_DIR / f"timeseries_{run_name}.csv"

            df_final = process_glacier_stake(
                GLACIER_NAME,
                STAKE_NAME,
                config,
                m=M_REF,
                C=C,
                Arg_simu=run_name,
                spatial_as=(run_type == 'As_spatial'),
                output_file=outfile
            )

            if df_final is None:
                continue

        except Exception as e:
            print(f"[ERROR] {run_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    process_all_uncertainty_runs()