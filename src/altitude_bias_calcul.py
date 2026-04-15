import numpy as np
import pandas as pd
from pathlib import Path


All_dem_1932 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_1932_ALL.dat', delimiter=r'\s+', header=None)
All_dem_1956 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_1956_ALL.dat', delimiter=r'\s+', header=None)
All_dem_1967 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_1967_ALL.dat', delimiter=r'\s+', header=None)
All_dem_1982 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_1982_ALL.dat', delimiter=r'\s+', header=None)
All_dem_1991 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_1991_ALL.dat', delimiter=r'\s+', header=None)
All_dem_2004 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_2004_ALL.dat', delimiter=r'\s+', header=None)
All_dem_2008 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_2008_ALL.dat', delimiter=r'\s+', header=None)
All_dem_2012 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_2012_ALL.dat', delimiter=r'\s+', header=None)
All_dem_2017 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_2017_ALL.dat', delimiter=r'\s+', header=None)
All_dem_2020 = pd.read_csv('../data/All/Elmer_Init/Data/DEM_surface_2020_ALL.dat', delimiter=r'\s+', header=None)


All_105_alt_stakes = [np.nan, np.nan, np.nan, 
       All_alt_105.loc[All_alt_105['date']==1982, 'altitude'].values[0], 
       All_alt_105.loc[All_alt_105['date']==1991, 'altitude'].values[0],
       All_alt_105.loc[All_alt_105['date']==2004, 'altitude'].values[0], 
       All_alt_105.loc[All_alt_105['date']==2008, 'altitude'].values[0], 
       All_alt_105.loc[All_alt_105['date']==2012, 'altitude'].values[0], 
       All_alt_105.loc[All_alt_105['date']==2017, 'altitude'].values[0], 
       All_alt_105.loc[All_alt_105['date']==2020, 'altitude'].values[0]]


All_105_alt_dems = [interp_zdem(All_dem_1932, x_All_105, y_All_105), 
            interp_zdem(All_dem_1956, x_All_105, y_All_105), 
            interp_zdem(All_dem_1967, x_All_105, y_All_105), 
            interp_zdem(All_dem_1982, x_All_105, y_All_105), 
            interp_zdem(All_dem_1991, x_All_105, y_All_105), 
            interp_zdem(All_dem_2004, x_All_105, y_All_105), 
            interp_zdem(All_dem_2008, x_All_105, y_All_105), 
            interp_zdem(All_dem_2012, x_All_105, y_All_105), 
            interp_zdem(All_dem_2017, x_All_105, y_All_105), 
            interp_zdem(All_dem_2020, x_All_105, y_All_105)]


All_101_alt_stakes = [np.nan, np.nan, np.nan, 
       All_alt_101.loc[All_alt_101['date']==1982, 'altitude'].values[0], 
       All_alt_101.loc[All_alt_101['date']==1991, 'altitude'].values[0],
       All_alt_101.loc[All_alt_101['date']==2004, 'altitude'].values[0], 
       All_alt_101.loc[All_alt_101['date']==2008, 'altitude'].values[0], 
       All_alt_101.loc[All_alt_101['date']==2012, 'altitude'].values[0], 
       All_alt_101.loc[All_alt_101['date']==2017, 'altitude'].values[0], 
       All_alt_101.loc[All_alt_101['date']==2020, 'altitude'].values[0]]


All_101_alt_dems = [interp_zdem(All_dem_1932, x_All_101, y_All_101), 
            interp_zdem(All_dem_1956, x_All_101, y_All_101), 
            interp_zdem(All_dem_1967, x_All_101, y_All_101), 
            interp_zdem(All_dem_1982, x_All_101, y_All_101), 
            interp_zdem(All_dem_1991, x_All_101, y_All_101), 
            interp_zdem(All_dem_2004, x_All_101, y_All_101), 
            interp_zdem(All_dem_2008, x_All_101, y_All_101), 
            interp_zdem(All_dem_2012, x_All_101, y_All_101), 
            interp_zdem(All_dem_2017, x_All_101, y_All_101), 
            interp_zdem(All_dem_2020, x_All_101, y_All_101)]

for date, stake, dem in zip(All_years_DEM, All_101_alt_stakes, All_101_alt_dems):
    print(f"Année {date} :   Stake = {stake} m   |   DEM = {dem} m")

# Trouver le décalage moyen entre les épaiseurs de mesures gps et de dem
diff_alt_All_105 = np.nanmean([a-b for a,b in zip(All_105_alt_stakes, All_105_alt_dems)])
diff_alt_All_101 = np.nanmean([a-b for a,b in zip(All_101_alt_stakes, All_101_alt_dems)])
diff_alt_All_101