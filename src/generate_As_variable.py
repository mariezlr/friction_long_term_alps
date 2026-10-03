import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from pathlib import Path
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent

# ----- Read x and y from bedrock -----
input_file = script_dir / "../data/structural/bedrocks/DEM_bedrock_ARG.dat"
df_original = pd.read_csv(input_file, sep=' ', header=None, names=['x', 'y', 'z'])

# ----- Parameters -----
m = 3.0
Cw_base = 0.038
As_base = Cw_base ** (-m)

resolution = 20
sigma_pixels = 200 / resolution  # spatial correlation ~200m
variation_fraction = 0.30

# ----- Grid -----
x_vals = np.sort(df_original['x'].unique())
y_vals = np.sort(df_original['y'].unique())
nx, ny = len(x_vals), len(y_vals)

x_to_idx = {v: i for i, v in enumerate(x_vals)}
y_to_idx = {v: i for i, v in enumerate(y_vals)}

for i in range(4):
    output_path = script_dir / f"../data/structural/bedrocks/Cw_field_ARG_{i}.dat"
    if not output_path.exists():
        np.random.seed(15 + i)
        noise_2d = np.random.normal(0, 1, (ny, nx))
        noise_filtered = gaussian_filter(noise_2d, sigma=sigma_pixels)
        noise_filtered = noise_filtered / noise_filtered.std() * variation_fraction
        noise_filtered = np.clip(noise_filtered, -0.5, 0.5)
        
        factor_grid = 1.0 + noise_filtered

        idx_x = df_original['x'].map(x_to_idx).values
        idx_y = df_original['y'].map(y_to_idx).values
        factors_flat = factor_grid[idx_y, idx_x]

        As_perturbed = As_base * factors_flat
        Cw_perturbed = As_perturbed ** (-1.0 / m)

        # Output : x y Cw, same x,y as bedrock, without z column
        df_out = df_original[['x', 'y']].copy()
        df_out['Cw'] = Cw_perturbed

        with open(output_path, 'w') as f:
            for _, row in df_out.iterrows():
                f.write(f"{row['x']:.3f} {row['y']:.3f} {row['Cw']:.3f}\n")

        print(f"Perturbation {i}: Cw min={Cw_perturbed.min():.4f}, "
            f"max={Cw_perturbed.max():.4f}, mean={Cw_perturbed.mean():.4f}")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(4):
    output_path = script_dir / f"../data/structural/bedrocks/Cw_field_ARG_{i}.dat"
    df_plot = pd.read_csv(output_path, sep=' ', header=None, names=['x', 'y', 'Cw'])
    pivot = df_plot.pivot(index='y', columns='x', values='Cw')
    im = axes[i].pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='RdBu_r')
    axes[i].set_title(f'Perturbation {i}')
    axes[i].set_xlabel('x (m)')
    axes[i].set_ylabel('y (m)')
    plt.colorbar(im, ax=axes[i], label='$C_w$')

plt.suptitle(f'Champs de $C_w$ (base={Cw_base:.3f}, ±{int(variation_fraction*100)}%, corr~200m)')
plt.tight_layout()
plt.show()