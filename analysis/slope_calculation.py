from utils import GLACIERS
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter
from shapely.geometry import Point, Polygon

script_dir = Path(__file__).resolve().parent

### ----- Compute mean slope for each stake (adapt path_to_DEM_files) -----

def nan_uniform_filter(data, size):
    """
    Apply a uniform filter while ignoring NaN values.
    """
    mask = ~np.isnan(data)
    data_filled = np.where(mask, data, 0)

    filtered = uniform_filter(data_filled, size=size, mode='nearest')
    weight = uniform_filter(mask.astype(float), size=size, mode='nearest')

    weight[weight == 0] = np.nan                 # avoid division by 0
    return filtered / weight



def slope_analysis(x_profil, y_profil, years_DEM, name_glacier, ray, df_contour):
    """
    Compute slopes and locally averaged slopes along a given profile for a glacier across multiple DEM years.
    """

    polygon = Polygon(df_contour.iloc[:, :2].to_numpy())

    df0 = pd.read_csv(script_dir / f'../../data/{name_glacier}/Elmer_Init/Data/DEM_surface_{years_DEM[0]}_{name_glacier.upper()}.dat',
                      sep=r'\s+', names=['x','y','z'])
    inside = df0[['x','y']].apply(lambda r: polygon.contains(Point(r[0],r[1])), axis=1)

    slopes = []
    
    for year in years_DEM:
        df = pd.read_csv(script_dir / f'../../data/{name_glacier}/Elmer_Init/Data/DEM_surface_{year}_{name_glacier.upper()}.dat',
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
    """
    Compute the mean slope over a given period, interpolating missing years.
    """
    df = pd.DataFrame({'date':years,'slope':slopes}).set_index('date')
    df = df.reindex(range(df.index.min(), df.index.max()+1)).interpolate()

    date_min = date_min or df.index.min()
    date_max = date_max or df.index.max()

    return df.loc[date_min:date_max,'slope'].mean()


def run_slope_mean_for_all(date_min, date_max):
    """
    Loop over all glaciers and stakes, compute slopes, and save final CSV.
    """
    results = []

    glaciers_items = list(GLACIERS.items())
    for glacier_key, glacier_data in glaciers_items:
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

    df_out = pd.DataFrame(results)
#    df_out.to_csv(script_dir / "../data/mean_slopes.csv", index=False)

    return df_out


#run_slope_mean_for_all(date_min=1960, date_max=1980)
