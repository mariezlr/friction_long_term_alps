from utils import GLACIERS, geom_data_dir
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



def slope_analysis(x_profil, y_profil, years_DEM, glacier_name, stake_name, ray, df_outlines):
    """
    Compute slopes and locally averaged slopes along a given profile for a glacier across multiple DEM years.
    """

    polygon = Polygon(df_outlines.iloc[:, :2].to_numpy())

    df0 = pd.read_csv(geom_data_dir / f'surfaces/{glacier_name}/DEM_surface_{years_DEM[0]}_{glacier_name.upper()}.dat',
                      sep=r'\s+', names=['x','y','z'])
    inside = df0[['x','y']].apply(lambda r: polygon.contains(Point(r.iloc[0], r.iloc[1])), axis=1)

    slopes = []
    
    for year in years_DEM:
        df = pd.read_csv(geom_data_dir / f'surfaces/{glacier_name}/DEM_surface_{year}_{glacier_name.upper()}.dat',
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

    df = pd.DataFrame(slopes)
    df = df.rename(columns={"slope_avg": f"slope_avg_{window*dx:.0f}m"})  # renommage uniquement pour le CSV
    df.to_csv(geom_data_dir / f"slopes/{glacier_name}_{stake_name}_slopes.csv", index=False)
    return df.rename(columns={f"slope_avg_{window*dx:.0f}m": "slope_avg"})  # on remet le nom fixe pour le return



def mean_slope_over_time(years, slopes, date_min, date_max):
    """
    Compute the mean slope over a given period, interpolating missing years.
    """
    df = pd.DataFrame({'date':years,'slope':slopes}).set_index('date')
    df = df.reindex(range(df.index.min(), df.index.max()+1)).interpolate()

    date_min = date_min or df.index.min()
    date_max = date_max or df.index.max()

    return df.loc[date_min:date_max,'slope'].mean()


def run_slope_mean_for_all():
    """
    Loop over all glaciers and stakes, compute slopes, and save final CSV.
    """
    results = []

    glaciers_items = list(GLACIERS.items())
    for glacier_key, glacier_data in glaciers_items:
        full_name = glacier_data.get("full_name", glacier_key)
        outlines = pd.read_csv(glacier_data["outlines_file"], sep=r"\s+", header=None)
        years = glacier_data["years_DEM"]
        
        for stake, xy in glacier_data["xy_coords"].items():
            print(f"Glacier {full_name} | Stake {stake}")

            # calculates slopes for a stake
            df_slopes = slope_analysis(
                x_profil=xy[0],
                y_profil=xy[1],
                years_DEM=years,
                glacier_name=glacier_key,
                stake_name = stake,
                ray=glacier_data["avg_dist"][stake],
                df_outlines=outlines
            )

            mean_slope_full = mean_slope_over_time(
                years=years,
                slopes=pd.to_numeric(df_slopes["slope_avg"], errors="coerce"),
                date_min=None,
                date_max=None
            )

            mean_slope_6080 = mean_slope_over_time(
                years=years,
                slopes=pd.to_numeric(df_slopes["slope_avg"], errors="coerce"),
                date_min=1960,
                date_max=1980
            )

            results.append({
                "glacier": glacier_key,
                "stake": stake,
                "mean_slope_deg_full": mean_slope_full,
                "mean_slope_rad_full": np.radians(mean_slope_full),
                "mean_slope_deg_6080": mean_slope_6080,
                "mean_slope_rad_6080": np.radians(mean_slope_6080)
            })


    df_out = pd.DataFrame(results)
    df_out.to_csv(geom_data_dir / "slopes/mean_slopes.csv", index=False)

    return df_out


run_slope_mean_for_all()
