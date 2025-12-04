from utils import GLACIERS
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter
from shapely.geometry import Point, Polygon
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent


def nan_uniform_filter(data, size):
    """Applique un filtre uniforme en ignorant les NaN."""
    mask = np.isnan(data)
    data_filled = np.where(mask, 0, data)
    
    filtered = uniform_filter(data_filled, size=size, mode='nearest')
    weight = uniform_filter(mask.astype(float) == 0, size=size, mode='nearest')
    
    weight[weight == 0] = 1
    return filtered / weight


def nan_uniform_filter(data, size):
    """Filtre uniforme qui ignore les NaN."""
    mask = ~np.isnan(data)                       # True = valide
    data_filled = np.where(mask, data, 0)

    filtered = uniform_filter(data_filled, size=size, mode='nearest')
    weight = uniform_filter(mask.astype(float), size=size, mode='nearest')

    weight[weight == 0] = np.nan                 # évite division par 0
    return filtered / weight



def slope_analysis(x_profil, y_profil, years_DEM, name_glacier, ray, df_contour):

    # Build polygon
    polygon = Polygon(df_contour.iloc[:, :2].to_numpy())

    # Preprocessing of data inside the polygon
    df0 = pd.read_csv(script_dir / f'../../../data/{name_glacier}/Elmer_Init/Data/DEM_surface_{years_DEM[0]}_{name_glacier.upper()}.dat',
                      sep=r'\s+', names=['x','y','z'])
    inside = df0[['x','y']].apply(lambda r: polygon.contains(Point(r[0],r[1])), axis=1)

    slopes = []
    
    for year in years_DEM:
        df = pd.read_csv(script_dir / f'../../../data/{name_glacier}/Elmer_Init/Data/DEM_surface_{year}_{name_glacier.upper()}.dat',
                         sep=r'\s+', names=['x','y','z'])
        df = df[inside]

        # grid
        x_unique = np.unique(df.x)
        y_unique = np.unique(df.y)
        Z = df.pivot(index='y', columns='x', values='z').to_numpy()

        dx, dy = np.diff(x_unique).mean(), np.diff(y_unique).mean()
        dz_dx, dz_dy = np.gradient(Z, dx, dy)

        slope = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_deg = np.degrees(np.arctan(slope))

        # local average
        window = max(1,int(ray/dx))
        slope_avg = nan_uniform_filter(slope_deg, size=window)

        # extract profile
        x_closest = np.argmin(np.abs(x_unique-x_profil))
        y_closest = np.argmin(np.abs(y_unique-y_profil))

        slopes.append({
            "year":year,
            "slope":slope_deg[y_closest,x_closest],
            "slope_avg":slope_avg[y_closest,x_closest]
        })

    return pd.DataFrame(slopes)


def mean_slope_over_time(years, slopes, date_min, date_max):
    df = pd.DataFrame({'date':years,'slope':slopes}).set_index('date')
    df = df.reindex(range(df.index.min(), df.index.max()+1)).interpolate()

    date_min = date_min or df.index.min()
    date_max = date_max or df.index.max()

    return df.loc[date_min:date_max,'slope'].mean()


def run_slope_mean_for_all(date_min, date_max):
    """Boucle automatiquement sur tous les glaciers/stakes stockés dans GLACIERS,
       calcule les pentes puis exporte un CSV final propre.
    """
    results = []

    glaciers_items = list(GLACIERS.items())
    for glacier_key, glacier_data in tqdm(glaciers_items, desc="Glaciers", unit="glacier"):
        full_name = glacier_data.get("full_name", glacier_key)
        contour = pd.read_csv(glacier_data["contour_file"], sep=r"\s+", header=None)
        years = glacier_data["years_DEM"]
        
        for stake, xy in glacier_data["xy_coords"].items():
            print(f"Glacier {full_name} | Stake {stake}")

            # calculates slopes for a stake
            df_slopes = slope_analysis(
                x_profil=xy[0],
                y_profil=xy[1],
                years_DEM=years,
                name_glacier=glacier_key,
                ray=glacier_data["avg_dist"][stake],
                df_contour=contour
            )

            # mean slope
            mean_slope = mean_slope_over_time(
                years=years,
                slopes=pd.to_numeric(df_slopes["slope_avg"], errors="coerce"),
                date_min=date_min,
                date_max=date_max
            )

            results.append({
                "glacier": glacier_key,
                "stake": stake,
                "mean_slope_deg": mean_slope,
                "mean_slope_rad": np.radians(mean_slope)
            })

    # export final
    df_out = pd.DataFrame(results)
    df_out.to_csv(script_dir / "../data/processed/mean_slopes.csv", index=False)

    return df_out


run_slope_mean_for_all(date_min=1960, date_max=1980)
