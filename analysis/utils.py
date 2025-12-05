import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / ".." / "data" 
geom_data_dir = script_dir / ".." / "data" / "structural"
proc_data_dir = script_dir / ".." / "data" / "processed_timeseries"
fig_dir = script_dir / ".." / "figures" 

def get_years_for_glacier(glacier_name):
    df_years = pd.read_csv(data_dir / 'years_DEM.csv', sep=";")
    row = df_years[df_years['glacier'] == glacier_name]
    years_raw = row['years'].values[0] 
    years_list = ast.literal_eval(years_raw) 
    return years_list

def get_xy_coords(glacier_name, stake_name):
    df_xy = pd.read_csv(data_dir / 'coord_stakes_xy.csv', sep=",")
    row = df_xy[(df_xy['glacier'] == glacier_name) & (df_xy['stake'] == stake_name)]
    return float(row['x'].values[0]), float(row['y'].values[0])

def get_flowline_idx(glacier_name, stake_name):
    df_idx = pd.read_csv(geom_data_dir / 'flowlines' / 'flowline_idx.csv', sep=",")
    row = df_idx[(df_idx['glacier'] == glacier_name) & (df_idx['stake'] == stake_name)]
    return row['idx'].values[0]

def get_slope(glacier_name, stake_name):
    df_slopes = pd.read_csv(data_dir / 'mean_slopes.csv', sep=",")
    row = df_slopes[(df_slopes['glacier'] == glacier_name) & (df_slopes['stake'] == stake_name)]
    return row['mean_slope_deg'].values

def get_friclaw_params(glacier_name, stake_name):
    df_slopes = pd.read_csv(proc_data_dir / 'friction_fits' / 'friction_fit_params.csv', sep=",")
    row = df_slopes[(df_slopes['glacier'] == glacier_name) & (df_slopes['stake'] == stake_name)]
    if row.empty:
        print(f"[WARNING] Aucune entrée trouvée pour glacier='{glacier_name}' stake='{stake_name}'")
        return None
    CN, q, As, m = float(row['CN'].values[0]), float(row['q'].values[0]), float(row['As'].values[0]), float(row['m'].values[0])
    return CN, q, As, m


GLACIERS = {
    'All': {
        'full_name': "Allalin",
        'contour_file': geom_data_dir / 'outlines' / 'All_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'All_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'All_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("All"),
        'xy_coords': {'101': get_xy_coords("All", "101")},
        'flowline_idx': {'101': get_flowline_idx("All", "101")},
        'all_data': {'101': proc_data_dir / "All_all_data_101.csv"},   
        'avg_dist': {'101': 390},
        'slope_60_80': {'101': get_slope("All", "101")},
        'colors': {'101':'#5F9EA0'},
        'markers' :{'101': '8'},
        'friclaw_ts': {'101' : proc_data_dir / 'friction_fits' / 'All_101_friclaw_ts.csv'},
        'friclaw_params' : {'101' : get_friclaw_params("All", "101")}
    },
    'Arg': {
        'full_name': "Argentière",
        'contour_file': geom_data_dir / 'outlines' / 'Arg_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'Arg_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'Arg_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Arg"),
        'xy_coords': {'Wheel': get_xy_coords("Arg", "wheel"),
                      '4': get_xy_coords("Arg", "4"),
                      '5': get_xy_coords("Arg", "5")},
        'flowline_idx': {'Wheel': get_flowline_idx("Arg", "wheel"),
                      '4': get_flowline_idx("Arg", "4"),
                      '5': get_flowline_idx("Arg", "5")},        
        'all_data': {'Wheel': proc_data_dir / "Arg_all_data_w.csv",
                     '4': proc_data_dir / "Arg_all_data_4.csv",
                     '5': proc_data_dir / "Arg_all_data_5.csv",},  
        'slope_60_80': {'Wheel': get_slope("Arg", "Wheel"),
                        '4': get_slope("Arg", "4"),
                        '5': get_slope("Arg", "5")},
        'avg_dist': {'Wheel': 1, '4': 250, '5': 690},
        'colors': {'Wheel':'#CC79A7','4':'#AA4466','5':'#661100'},
        'markers': {'Wheel':'^','4':'v','5':'.'},
        'friclaw_ts': {'4' : proc_data_dir / 'friction_fits' / 'Arg_4_friclaw_ts.csv',
                       '5' : proc_data_dir / 'friction_fits' / 'Arg_5_friclaw_ts.csv'},
        'friclaw_params' : {'4' : get_friclaw_params("Arg", "4"),
                            '5' : get_friclaw_params("Arg", "5")}
    },
    'Cor': {
        'full_name': "Corbassière",
        'contour_file': geom_data_dir / 'outlines' / 'Cor_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'Cor_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'Cor_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Cor"),
        'xy_coords': {'A4': get_xy_coords("Cor", "A4"),
                      'B4': get_xy_coords("Cor", "B4")},
        'flowline_idx': {'A4': get_flowline_idx("Cor", "A4"),
                      'B4': get_flowline_idx("Cor", "B4")},        
        'all_data': {'A4': proc_data_dir / "Cor_all_data_A4.csv",
                     'B4': proc_data_dir / "Cor_all_data_B4.csv"}, 
        'slope_60_80': {'A4': get_slope("Cor", "A4"),
                        'B4': get_slope("Cor", "B4")},  
        'avg_dist': {'A4': 390, 'B4': 360},
        'colors': {'A4':'#FF5733','B4':'#FFC300'},
        'markers': {'A4':'p','B4':'^'},
        'friclaw_ts': {'A4' : proc_data_dir / 'friction_fits' / 'Cor_A4_friclaw_ts.csv',
                       'B4' : proc_data_dir / 'friction_fits' / 'Cor_B4_friclaw_ts.csv'},
        'friclaw_params' : {'A4' : get_friclaw_params("Cor", "A4"),
                            'B4' : get_friclaw_params("Cor", "B4")}
    },
    'Geb': {
        'full_name': "Gébroulaz",
        'contour_file': geom_data_dir / 'outlines' / 'Geb_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'Geb_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'Geb_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Geb"),
        'xy_coords': {'sup': get_xy_coords("Geb", "sup"),
                      'ss': get_xy_coords("Geb", "ss")},
        'flowline_idx': {'sup': get_flowline_idx("Geb", "sup"),
                      'ss': get_flowline_idx("Geb", "ss")}, 
        'all_data': {'sup': proc_data_dir / "Geb_all_data_sup.csv",
                     'ss': proc_data_dir / "Geb_all_data_ss.csv"}, 
        'slope_60_80': {'sup': get_slope("Geb", "sup"),
                        'ss': get_slope("Geb", "ss")},        
        'avg_dist':{'sup': 150, 'ss': 50},
        'colors': {'sup':'#669966','ss':'#999999'},
        'markers': {'sup':'s','ss':'2'},
        'friclaw_ts': {'sup' : proc_data_dir / 'friction_fits' / 'Geb_sup_friclaw_ts.csv',
                       'ss' : proc_data_dir / 'friction_fits' / 'Geb_ss_friclaw_ts.csv'},
        'friclaw_params' : {'sup' : get_friclaw_params("Geb", "sup"),
                            'ss' : get_friclaw_params("Geb", "ss")}
    },
    'Gie': {
        'full_name': "Giétro",
        'contour_file': geom_data_dir / 'outlines' / 'Gie_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'Gie_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'Gie_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Gie"),
        'xy_coords': {'5': get_xy_coords("Gie", "5"),
                      '102': get_xy_coords("Gie", "102")},
        'flowline_idx': {'5': get_flowline_idx("Gie", "5"),
                         '102': get_flowline_idx("Gie", "102")},  
        'all_data': {'5': proc_data_dir / "Gie_all_data_5.csv",
                     '102': proc_data_dir / "Gie_all_data_102.csv"},    
        'slope_60_80': {'5': get_slope("Gie", "5"),
                        '102': get_slope("Gie", "102")}, 
        'avg_dist':{'5': 300, '102': 70},
        'colors': {'5':'#117733','102':'#009E73'},
        'markers': {'5':'<','102':'>'},
        'friclaw_ts': {'5' : proc_data_dir / 'friction_fits' / 'Gie_5_friclaw_ts.csv',
                       '102' : proc_data_dir / 'friction_fits' / 'Gie_102_friclaw_ts.csv'},
        'friclaw_params' : {'5' : get_friclaw_params("Gie", "5"),
                            '102' : get_friclaw_params("Gie", "102")}
    },
    'GB': {
        'full_name': "Glacier Blanc",
        'contour_file': geom_data_dir / 'outlines' / 'GB_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'GB_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'GB_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("GB"),
        'xy_coords': {'inf': get_xy_coords("GB", "inf"),
                      'sup': get_xy_coords("GB", "sup")},
        'flowline_idx': {'inf': get_flowline_idx("GB", "inf"),
                      'sup': get_flowline_idx("GB", "sup")},        
        'all_data': {'inf': proc_data_dir / "GB_all_data_inf.csv",
                     'sup': proc_data_dir / "GB_all_data_sup.csv"}, 
        'slope_60_80': {'inf': get_slope("GB", "inf"),
                        'sup': get_slope("GB", "sup")},   
        'avg_dist':{'inf': 260, 'sup': 230},
        'colors': {'inf':'#56B4E9','sup':'#0072B2'},
        'markers': {'inf':'H','sup':'h'},
        'friclaw_ts': {'inf' : proc_data_dir / 'friction_fits' / 'GB_inf_friclaw_ts.csv',
                       'sup' : proc_data_dir / 'friction_fits' / 'GB_sup_friclaw_ts.csv'},
        'friclaw_params' : {'inf' : get_friclaw_params("GB", "inf"),
                            'sup' : get_friclaw_params("GB", "sup")}
    },
    'MDG': {
        'full_name': "Mer de Glace",
        'contour_file': geom_data_dir / 'outlines' / 'MDG_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'MDG_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'MDG_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("MDG"),
        'xy_coords': {'ech': get_xy_coords("MDG", "ech"),
                      'trel': get_xy_coords("MDG", "trel"),
                      'tac': get_xy_coords("MDG", "tac")},
        'flowline_idx': {'ech': get_flowline_idx("MDG", "ech"),
                      'trel': get_flowline_idx("MDG", "trel"),
                      'tac': get_flowline_idx("MDG", "tac")},
        'all_data': {'ech': proc_data_dir / "MDG_all_data_ech.csv",
                     'trel': proc_data_dir / "MDG_all_data_trel.csv",
                     'tac': proc_data_dir / "MDG_all_data_tac.csv"}, 
        'slope_60_80': {'ech': get_slope("MDG", "ech"),
                        'trel': get_slope("MDG", "trel"),
                        'tac': get_slope("MDG", "tac")},  
        'avg_dist':{'ech': 550, 'trel': 310, 'tac': 200},
        'colors': {'ech':'#F0E442','trel':'#E69F00', 'tac':'#D55E00'},
        'markers': {'ech':'D','trel':'*', 'tac':'o'},
        'friclaw_ts': {'ech' : proc_data_dir / 'friction_fits' / 'MDG_ech_friclaw_ts.csv',
                       'trel' : proc_data_dir / 'friction_fits' / 'MDG_trel_friclaw_ts.csv',
                       'tac' : proc_data_dir / 'friction_fits' / 'MDG_tac_friclaw_ts.csv'},
        'friclaw_params' : {'ech' : get_friclaw_params("MDG", "ech"),
                            'trel' : get_friclaw_params("MDG", "trel"),
                            'tac' : get_friclaw_params("MDG", "tac")}
    },
    'StSo': {
        'full_name': "Saint-Sorlin",
        'contour_file': geom_data_dir / 'outlines' / 'StSo_Contour.dat',
        'flowline': geom_data_dir / 'flowlines' / 'StSo_flowline.csv',
        'longit_cs': geom_data_dir / 'flowlines' / 'StSo_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("StSo"),
        'xy_coords': {'B': get_xy_coords("StSo", "B"),
                      'C': get_xy_coords("StSo", "C")},
        'flowline_idx': {'B': get_flowline_idx("StSo", "B"),
                      'C': get_flowline_idx("StSo", "C")},
        'all_data': {'B': proc_data_dir / "StSo_all_data_B.csv",
                     'C': proc_data_dir / "StSo_all_data_C.csv"},
        'slope_60_80': {'B': get_slope("StSo", "B"),
                        'C': get_slope("StSo", "C")},   
        'avg_dist':{'B': 160, 'C': 80},
        'colors': {'B':'#882255','C':'#AA4499'},
        'markers': {'B':'*','C':'d'},
        'friclaw_ts': {'B' : proc_data_dir / 'friction_fits' / 'StSo_B_friclaw_ts.csv',
                       'C' : proc_data_dir / 'friction_fits' / 'StSo_C_friclaw_ts.csv'},
        'friclaw_params' : {'B' : get_friclaw_params("StSo", "B"),
                            'C' : get_friclaw_params("StSo", "C")}
    }
}
