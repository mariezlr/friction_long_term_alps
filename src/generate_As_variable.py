import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# === Lire seulement x, y du bedrock ===
input_file = script_dir / "../data/structural/bedrocks/DEM_bedrock_ARG.dat"
df_original = pd.read_csv(input_file, sep=' ', header=None, names=['x', 'y', 'z'])

# === Paramètres ===
m = 3.0
Cw_base = 0.074
As_base = Cw_base ** (-m)

resolution = 20
sigma_pixels = 200 / resolution  # corrélation spatiale ~200m
variation_fraction = 0.30

# === Grille ===
x_vals = np.sort(df_original['x'].unique())
y_vals = np.sort(df_original['y'].unique())
nx, ny = len(x_vals), len(y_vals)

x_to_idx = {v: i for i, v in enumerate(x_vals)}
y_to_idx = {v: i for i, v in enumerate(y_vals)}

for i in range(4):
    noise_2d = np.random.normal(0, 1, (ny, nx))
    noise_filtered = gaussian_filter(noise_2d, sigma=sigma_pixels)
    noise_filtered = noise_filtered / noise_filtered.std() * variation_fraction
    noise_filtered = np.clip(noise_filtered, -0.5, 0.5)
    
    factor_grid = 1.0 + noise_filtered

    # Indexation vectorisée
    idx_x = df_original['x'].map(x_to_idx).values
    idx_y = df_original['y'].map(y_to_idx).values
    factors_flat = factor_grid[idx_y, idx_x]

    As_perturbed = As_base * factors_flat
    Cw_perturbed = As_perturbed ** (-1.0 / m)

    # Sortie : x y Cw, mêmes x,y que le bedrock, sans la colonne z
    df_out = df_original[['x', 'y']].copy()
    df_out['Cw'] = Cw_perturbed

    output_path = script_dir / f"../data/structural/bedrocks/Cw_field_ARG_{i}.dat"
    with open(output_path, 'w') as f:
        for _, row in df_out.iterrows():
            f.write(f"{row['x']:.3f} {row['y']:.3f} {row['Cw']:.3f}\n")

    print(f"Perturbation {i}: Cw min={Cw_perturbed.min():.4f}, "
          f"max={Cw_perturbed.max():.4f}, mean={Cw_perturbed.mean():.4f}")