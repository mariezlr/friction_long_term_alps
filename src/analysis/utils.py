import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / ".." / ".." / "data"

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
    df_idx = pd.read_csv(data_dir / 'flowline_idx.csv', sep=",")
    row = df_idx[(df_idx['glacier'] == glacier_name) & (df_idx['stake'] == stake_name)]
    return row['idx'].values[0]


GLACIERS = {
    'All': {
        'full_name': "Allalin",
        'contour_file': data_dir / 'All_Contour.dat',
        'flowline': data_dir / 'All_flowline.csv',
        'longit_cs': data_dir / 'All_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("All"),
        'xy_coords': {'101': get_xy_coords("All", "101")},
        'flowline_idx': {'101': get_flowline_idx("All", "101")},
        'all_data': {'101': data_dir / "processed/All_all_data_101.csv"},   
        'avg_dist': {'101': 390},
        'colors': {'101':'#5F9EA0'},
        'markers' :{'101': '8'}
    },
    'Arg': {
        'full_name': "Argentière",
        'contour_file': data_dir / 'Arg_Contour.dat',
        'flowline': data_dir / 'Arg_flowline.csv',
        'longit_cs': data_dir / 'Arg_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Arg"),
        'xy_coords': {'Wheel': get_xy_coords("Arg", "wheel"),
                      '4': get_xy_coords("Arg", "4"),
                      '5': get_xy_coords("Arg", "5")},
        'flowline_idx': {'Wheel': get_flowline_idx("Arg", "wheel"),
                      '4': get_flowline_idx("Arg", "4"),
                      '5': get_flowline_idx("Arg", "5")},        
        'all_data': {'Wheel': data_dir / "Arg_all_data_w.csv",
                     '4': data_dir / "processed/Arg_all_data_4.csv",
                     '5': data_dir / "processed/Arg_all_data_5.csv",},   
        'avg_dist': {'Wheel': 1, '4': 250, '5': 690},
        'colors': {'Wheel':'#CC79A7','4':'#AA4466','5':'#661100'},
        'markers': {'Wheel':'^','4':'v','5':'.'},
    },
    'Cor': {
        'full_name': "Corbassière",
        'contour_file': data_dir / 'Cor_Contour.dat',
        'flowline': data_dir / 'Cor_flowline.csv',
        'longit_cs': data_dir / 'Cor_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Cor"),
        'xy_coords': {'A4': get_xy_coords("Cor", "A4"),
                      'B4': get_xy_coords("Cor", "B4")},
        'flowline_idx': {'A4': get_flowline_idx("Cor", "A4"),
                      'B4': get_flowline_idx("Cor", "B4")},        
        'all_data': {'A4': data_dir / "processed/Cor_all_data_A4.csv",
                     'B4': data_dir / "processed/Cor_all_data_B4.csv"},   
        'avg_dist': {'A4': 390, 'B4': 360},
        'colors': {'A4':'#FF5733','B4':'#FFC300'},
        'markers': {'A4':'p','B4':'^'},
    },
    'Geb': {
        'full_name': "Gébroulaz",
        'contour_file': data_dir / 'Geb_Contour.dat',
        'flowline': data_dir / 'Geb_flowline.csv',
        'longit_cs': data_dir / 'Geb_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Geb"),
        'xy_coords': {'sup': get_xy_coords("Geb", "sup"),
                      'ss': get_xy_coords("Geb", "ss")},
        'flowline_idx': {'sup': get_flowline_idx("Geb", "sup"),
                      'ss': get_flowline_idx("Geb", "ss")}, 
        'all_data': {'sup': data_dir / "processed/Geb_all_data_sup.csv",
                     'ss': data_dir / "processed/Geb_all_data_ss.csv"},        
        'avg_dist':{'sup': 150, 'ss': 50},
        'colors': {'sup':'#669966','ss':'#999999'},
        'markers': {'sup':'s','ss':'2'},
    },
    'Gie': {
        'full_name': "Giétro",
        'contour_file': data_dir / 'Gie_Contour.dat',
        'flowline': data_dir / 'Gie_flowline.csv',
        'longit_cs': data_dir / 'Gie_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("Gie"),
        'xy_coords': {'5': get_xy_coords("Gie", "5"),
                      '102': get_xy_coords("Gie", "102")},
        'flowline_idx': {'5': get_flowline_idx("Gie", "5"),
                         '102': get_flowline_idx("Gie", "102")},  
        'all_data': {'5': data_dir / "processed/Gie_all_data_5.csv",
                     '102': data_dir / "processed/Gie_all_data_102.csv"},    
        'avg_dist':{'5': 300, '102': 70},
        'colors': {'5':'#117733','102':'#009E73'},
        'markers': {'5':'<','102':'>'},
    },
    'GB': {
        'full_name': "Glacier Blanc",
        'contour_file': data_dir / 'GB_Contour.dat',
        'flowline': data_dir / 'GB_flowline.csv',
        'longit_cs': data_dir / 'GB_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("GB"),
        'xy_coords': {'inf': get_xy_coords("GB", "inf"),
                      'sup': get_xy_coords("GB", "sup")},
        'flowline_idx': {'inf': get_flowline_idx("GB", "inf"),
                      'sup': get_flowline_idx("GB", "sup")},        
        'all_data': {'inf': data_dir / "processed/GB_all_data_inf.csv",
                     'sup': data_dir / "processed/GB_all_data_sup.csv"},   
        'avg_dist':{'inf': 260, 'sup': 230},
        'colors': {'inf':'#56B4E9','sup':'#0072B2'},
        'markers': {'inf':'H','sup':'h'},
    },
    'MDG': {
        'full_name': "Mer de Glace",
        'contour_file': data_dir / 'MDG_Contour.dat',
        'flowline': data_dir / 'MDG_flowline.csv',
        'longit_cs': data_dir / 'MDG_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("MDG"),
        'xy_coords': {'ech': get_xy_coords("MDG", "ech"),
                      'trel': get_xy_coords("MDG", "trel"),
                      'tac': get_xy_coords("MDG", "tac")},
        'flowline_idx': {'ech': get_flowline_idx("MDG", "ech"),
                      'trel': get_flowline_idx("MDG", "trel"),
                      'tac': get_flowline_idx("MDG", "tac")},
        'all_data': {'ech': data_dir / "processed/MDG_all_data_ech.csv",
                     'trel': data_dir / "processed/MDG_all_data_trel.csv",
                     'tac': data_dir / "processed/MDG_all_data_tac.csv"},   
        'avg_dist':{'ech': 550, 'trel': 310, 'tac': 200},
        'colors': {'ech':'#F0E442','trel':'#E69F00', 'tac':'#D55E00'},
        'markers': {'ech':'D','trel':'*', 'tac':'o'},
    },
    'StSo': {
        'full_name': "Saint-Sorlin",
        'contour_file': data_dir / 'StSo_Contour.dat',
        'flowline': data_dir / 'StSo_flowline.csv',
        'longit_cs': data_dir / 'StSo_long_cross_section.csv',
        'years_DEM': get_years_for_glacier("StSo"),
        'xy_coords': {'B': get_xy_coords("StSo", "B"),
                      'C': get_xy_coords("StSo", "C")},
        'flowline_idx': {'B': get_flowline_idx("StSo", "B"),
                      'C': get_flowline_idx("StSo", "C")},
        'all_data': {'B': data_dir / "processed/StSo_all_data_B.csv",
                     'C': data_dir / "processed/StSo_all_data_C.csv"},   
        'avg_dist':{'B': 160, 'C': 80},
        'colors': {'B':'#882255','C':'#AA4499'},
        'markers': {'B':'*','C':'d'},
    }
}


datasets = {
    "All_101":  (data_dir / "All_all_data_101.csv", "obs_u_bed", "obs_tau_b"),
    "Wheel":    (data_dir / "Arg_all_data_w.csv", "velocity", "basal_stress"),
    "Arg_4":    (data_dir / "Arg_all_data_4.csv", "obs_u_bed", "obs_tau_b"),
    "Arg_5":    (data_dir / "Arg_all_data_5.csv", "obs_u_bed", "obs_tau_b"),
    "Cor_A4":   (data_dir / "Cor_all_data_A4.csv", "obs_u_bed", "obs_tau_b"),
    "Cor_B4":   (data_dir / "Cor_all_data_B4.csv", "obs_u_bed", "obs_tau_b"),
    "Geb_sup":  (data_dir / "Geb_all_data_sup.csv", "obs_u_bed", "obs_tau_b"),
    "Geb_ss":   (data_dir / "Geb_all_data_ss.csv", "obs_u_bed", "obs_tau_b"),
    "Gie_102":  (data_dir / "Gie_all_data_102.csv", "obs_u_bed", "obs_tau_b"),
    "Gie_5":    (data_dir / "Gie_all_data_5.csv", "obs_u_bed", "obs_tau_b"),
    "GB_sup":   (data_dir / "GB_all_data_sup.csv", "obs_u_bed", "obs_tau_b"),
    "GB_inf":   (data_dir / "GB_all_data_inf.csv", "obs_u_bed", "obs_tau_b"),
    "MDG_tac":  (data_dir / "MDG_all_data_tac.csv", "obs_u_bed", "obs_tau_b"),
    "MDG_trel": (data_dir / "MDG_all_data_trel.csv", "obs_u_bed", "obs_tau_b",
                 lambda df: df["obs_u_bed"] > 0.8),
    "MDG_ech":  (data_dir / "MDG_all_data_ech.csv", "obs_u_bed", "obs_tau_b"),
    "StSo_A":   (data_dir / "StSo_all_data_A.csv", "obs_u_bed", "obs_tau_b"),
    "StSo_B":   (data_dir / "StSo_all_data_B.csv", "obs_u_bed", "obs_tau_b"),
    "StSo_C":   (data_dir / "StSo_all_data_C.csv", "obs_u_bed", "obs_tau_b"),
}