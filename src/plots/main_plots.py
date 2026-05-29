import sys
from pathlib import Path

# Ajouter src au path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, data_dir, geom_data_dir, get_friclaw_params
from friction_laws import *
from run_friction_fits import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from collections import OrderedDict
from matplotlib.ticker import LogFormatter

script_dir = Path(__file__).resolve().parent

plt.rcParams["lines.linewidth"] = 0.9
plt.rcParams.update({"font.size": 12,       
                     "axes.labelsize": 14,  
                     "axes.titlesize": 14,  
                     "legend.fontsize": 12, 
                     "xtick.labelsize": 12, 
                     "ytick.labelsize": 12 
                    })



def plot_surface_vel_timeseries():
    """
    Plot observed surface vel timeseries for all glaciers defined in config.GLACIERS.
    """
    left_panel = {'Cor': ['B4','A4'], 'Geb': ['ss','sup'], 'Gie': ['5'], 'GB': ['sup','inf'], 'StSo': ['B','C']}
    right_panel = {'All': ['101'], 'Arg': ['5','4'], 'Gie': ['102'], 'MDG': ['tac','trel','ech']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={'width_ratios':[1,1]})

    # Intern function to plot a panel
    def plot_panel(ax, panel_dict):

        for glacier, stakes in panel_dict.items():

            full_name = GLACIERS[glacier]['full_name']
            for stake in stakes:

                file = data_dir / "obs_raw" / f"{glacier}_vel_{stake}.csv"
                color = GLACIERS[glacier]['colors'][stake]
                marker = 'o'

                df = pd.read_csv(file)

                label = f"{full_name} {stake}"

                mask = ~df['velocity'].isna()
                ax.plot(df['date'][mask], df['velocity'][mask], marker=marker, color=color,
                        linestyle='-', label=label)

        ax.legend()
        ax.grid(True, linestyle='dotted')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Surface velocity ($m.yr^{-1}$)', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plot_panel(ax1, left_panel)
    plot_panel(ax2, right_panel)

    # Add labels (a) & (b)
    ax1.text(-0.05, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')
    ax2.text(-0.05, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')

    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    fig.savefig(fig_dir / "timeseries_surface_vel.pdf", bbox_inches='tight')
    plt.close(fig)
    print("timeseries_surface_vel saved")


def plot_thk_changes_timeseries():
    """
    Plot observed thickness change timeseries for all glaciers defined in config.GLACIERS.
    """
    left_panel = {'Geb': ['ss','sup'], 'GB': ['sup','inf'], 'MDG': ['tac','trel','ech'], 'StSo': ['B','C']}
    right_panel = {'All': ['101'], 'Arg': ['5','4'], 'Cor': ['B4','A4'], 'Gie': ['5', '102']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={'width_ratios':[1,1]})

    # Intern function to plot a panel
    def plot_panel(ax, panel_dict):

        for glacier, stakes in panel_dict.items():

            full_name = GLACIERS[glacier]['full_name']
            for stake in stakes:

                file = data_dir / "obs_raw" / f"{glacier}_alt_{stake}.csv"
                color = GLACIERS[glacier]['colors'][stake]
                marker = 'o'

                df = pd.read_csv(file)

                label = f"{full_name} {stake}"
                
                mask = ~df['altitude'].isna()
                ax.plot(df['date'][mask], df['altitude'][mask] - df['altitude'][mask].iloc[0], marker=marker, color=color,
                        linestyle='-', label=label)

        ax.legend()
        ax.grid(True, linestyle='dotted')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Thickness Variation (m)', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plot_panel(ax1, left_panel)
    plot_panel(ax2, right_panel)

    # Add labels (a) & (b)
    ax1.text(-0.05, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')
    ax2.text(-0.05, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')

    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    fig.savefig(fig_dir / "timeseries_thk_changes.pdf", bbox_inches='tight')
    plt.close(fig)
    print("timeseries_thk_changes saved")


def plot_glaciers_longit_cs():
    """
    Plot flowlines, outlines, and longitudinal profiles for all glaciers defined in config.GLACIERS.
    """
    order = ['All', 'Gie', 'Arg', 'GB', 'Cor', 'MDG', 'Geb', 'StSo']
    GLACIERS_sorted = OrderedDict((k, GLACIERS[k]) for k in order if k in GLACIERS)


    n_glaciers = len(GLACIERS_sorted)
    n_rows = (n_glaciers * 2 + 3) // 4  # 2 axes per glacier / 4 columns
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, 15))
    axes = axes.ravel()

    for i, (glacier_name, glacier_data) in enumerate(GLACIERS_sorted.items()):
        # Read files
        glacier_full_name = glacier_data['full_name']
        df_outlines = pd.read_csv(glacier_data['outlines_file'], sep="\s+", header=None)
        df_flowline = pd.read_csv(glacier_data['flowline'], sep=',', header=0)
        df_longit_cs = pd.read_csv(glacier_data['longit_cs'])
        years = glacier_data['years_DEM']
        points = glacier_data['xy_coords']
        flowline_idx = glacier_data['flowline_idx']
        colors = glacier_data['colors']
        avg_dist = glacier_data['avg_dist']

        # Axes
        ax_outlines = axes[2*i]
        ax_longit = axes[2*i + 1]

        # Points and flowline
        ax_outlines.plot(df_outlines.iloc[:,0], df_outlines.iloc[:,1], 'k-')
        if points:
            ax_outlines.scatter(*zip(*points.values()), c=list(colors.values()), s=80, edgecolors='black', zorder=3)
            for label, (x, y) in points.items():
                ax_outlines.annotate(label, (x, y), xytext=(10,10), textcoords="offset points",
                                    ha='right', fontsize=12, color=colors[label])
        ax_outlines.plot(df_flowline.iloc[:,0], df_flowline.iloc[:,1], color='r', label='Smooth flowline')
        ax_outlines.set_title(glacier_name)
        ax_outlines.set_aspect('equal')
        ax_outlines.add_artist(ScaleBar(1, location='lower right'))
        for spine in ax_outlines.spines.values():
            spine.set_visible(False)
        ax_outlines.set_xticks([])
        ax_outlines.set_yticks([])
        ax_outlines.set_xlabel('')
        ax_outlines.set_ylabel('')
        ax_outlines.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)   
        ax_outlines.annotate('N', xy=(0.9, 0.95), xytext=(0.9, 0.85), arrowprops=dict(facecolor='black', arrowstyle='-|>'), 
            ha='center', va='center', fontsize=16, xycoords='axes fraction')
        ax_outlines.set_title(glacier_full_name, fontsize=22)

        # Longitudinal cross-section
        ax_longit.plot(df_longit_cs['dist'], df_longit_cs['z_bed'], color='k', label='Bedrock')
        for year in years:
            ax_longit.plot(df_longit_cs['dist'], df_longit_cs[f'z_surf_{year}'], label=str(year))

        for label, (x, y) in points.items():
            idx = flowline_idx[label]
            x_dist = df_longit_cs['dist'].iloc[idx]
            y_alt = df_longit_cs[f'z_surf_{years[0]}'].iloc[idx]
            ax_longit.fill_betweenx(ax_longit.get_ylim(),
                                    x_dist - avg_dist[label],
                                    x_dist + avg_dist[label],
                                    color=colors[label], alpha=0.2)
            ax_longit.axvline(x=x_dist, color=colors[label], linestyle='--')
            ax_longit.annotate(label, xy=(x_dist, y_alt), xytext=(x_dist + 50, y_alt + 50),
                               arrowprops=dict(facecolor='k', arrowstyle='->'))
        ax_longit.set_ylabel('Altitude (m)', fontsize=18)
        ax_longit.legend()
        ax_longit.yaxis.label.set_size(18) 
        ax_longit.grid(True, linestyle='dotted')
        ax_longit.set_title(glacier_full_name, fontsize=22)

    plt.tight_layout()
    fig.savefig(fig_dir / "longitudinal_cuts.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("longitudinal_cuts saved")



def plot_friction_laws(m=3):

    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
               0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))
    
    for glacier_key, glacier_data in GLACIERS.items():
        
        for stake in glacier_data['xy_coords'].keys():

            if stake == "Wheel": # only for comparison, not a studied point
                continue

            color = glacier_data['colors'][stake]
            marker = glacier_data['markers'][stake]

            # RAW DATA
            try:
                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                    print(f"No data for {glacier_key}-{stake}")
                    continue

                if marker == '2':
                    ax1.scatter(vel, tau, color=color, marker=marker,
                                label=f"{glacier_data['full_name']} {stake}", zorder=10)
                else:
                    ax1.scatter(vel, tau, color=color, edgecolor='k', marker=marker,
                                label=f"{glacier_data['full_name']} {stake}", zorder=10)

            except Exception as e:
                print(f"Skip {glacier_key}-{stake} (raw): {e}")
                continue

            # FIT DATA
            if glacier_key not in ["StSo"]:
                fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"
                if not Path(fit_file).exists():
                    print(f"Missing fit file {glacier_key}-{stake}")
                    continue
                df_fit = pd.read_csv(fit_file)
                ax1.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, linewidth=2)

            # PARAMS
            CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake, m)


            # if glacier_key == "Geb":
            #     ax1.axhline(y=CN_value, color=color, linestyle='--')

            # NORMALIZED
            if glacier_key in ["Geb", "StSo"]:
                continue

            try:
                vel_norm, tau_norm = calcul_normalised_friction_law(vel, tau, CN_value, As_value, m_value)

                if len(vel_norm) == 0:
                    continue

                ax2.scatter(vel_norm, tau_norm, color=color, edgecolor='k', marker=marker,
                            label=f"{glacier_data['full_name']} {stake}")

            except Exception as e:
                print(f"Skip normalized {glacier_key}-{stake}: {e}")

    # Theoritical law
    V_values = np.arange(0.05,50,0.1)
    ax2.plot(V_values, [scaled_friction_law(u, 1) for u in V_values], color='k', label='cavitation law')
    ax2.plot(np.arange(0.05,1.5,0.1), np.arange(0.05,1.5,0.1), 'b--', label='Weertman-type law')
    
    # Labels
    ax1.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$')
    ax1.set_ylabel(r'Basal shear stress (MPa)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.8, 200)
    ax1.set_ylim(0.012, 0.17)
    ax1.set_xticks([x for x in x_ticks if ax1.get_xlim()[0] <= x <= ax1.get_xlim()[-1]])
    ax1.set_yticks([y for y in y_ticks if ax1.get_ylim()[0] <= y <= ax1.get_ylim()[-1]])

    ax2.set_xlabel(r'Scaled sliding velocity $\frac{u_b}{A_s(CN)^m}$')
    ax2.set_ylabel(r'Scaled shear stress $\left(\frac{\tau_b}{CN}\right)^m$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    for ax in [ax1, ax2]:
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax.grid(which='both', linestyle='dotted')
    

    # (a) and (b)
    ax1.text(-0.05, 1.08, '(a)', transform=ax1.transAxes,
         fontsize=18, fontweight='bold', va='top', ha='right')
    ax2.text(-0.05, 1.08, '(b)', transform=ax2.transAxes,
         fontsize=18, fontweight='bold', va='top', ha='right')

    # Légende commune
    plt.tight_layout()
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=16)
    ax1.legend().remove()
    fig.subplots_adjust(bottom=0.28)
    
    fig.savefig(fig_dir / f"friction_laws_main_m{m}.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("friction_laws_main saved")




def plot_CN_vs_slope(mw=3):
    
    fig, ax = plt.subplots(figsize=(6,5))

    slopes_deg, CN_values = [], []

    for glacier_key, glacier_data in GLACIERS.items():
        
        if glacier_key in ["Geb", "StSo"]:   # ignoring soft-bedd and bad constrained glaciers
            continue

        for stake in glacier_data['xy_coords'].keys():

            df_slopes = pd.read_csv(geom_data_dir / 'slopes/mean_slopes.csv', sep=",")
            row = df_slopes[(df_slopes['glacier'] == glacier_key) & (df_slopes['stake'] == stake)]
            slope_deg = row['mean_slope_deg_6080'].values

            if stake == "Wheel":
                CN = 0.217
            
            else:
                params = get_friclaw_params(glacier_key, stake, mw=mw)
                if params is None:
                    print(f"  [SKIP] {glacier_key} {stake} : pas de params")
                    continue
                CN = params[0]
            
            slopes_deg.append(slope_deg[0])
            CN_values.append(CN)

            color = GLACIERS[glacier_key]['colors'][stake]
            label = f"{glacier_key} {stake}"

            ax.scatter(slope_deg, CN, c=color, zorder=2)

            sigma_CN = 0.002112704740016016
            ax.errorbar(slope_deg, CN, yerr=sigma_CN, 
                        fmt='none', ecolor='gray', capsize=3, zorder=1)
            
            # adding names of the stakes
            if stake == "A4":
                ax.text(0.92*slope_deg, 0.94*CN, label, fontsize=9, ha='left')
            elif stake == "4":
                ax.text(0.78*slope_deg, CN, label, fontsize=9, ha='left')
            elif stake == "trel":
                ax.text(1.07*slope_deg, CN, label, fontsize=9, ha='left')
            else:
                ax.text(0.92*slope_deg, 1.03*CN, label, fontsize=9, ha='left')
    
    tan_slopes = np.tan(np.radians(slopes_deg))

    def model_fixed_p(x, C):
        return C * x**0.47

    # Fit uniquement pour C
    popt, pcov = curve_fit(model_fixed_p, tan_slopes, CN_values, p0=[0.3])
    C_fit = popt[0]
    C_err = np.sqrt(np.diag(pcov))[0]

    alpha_values = np.arange(2, np.max(slopes_deg), 0.001)
    x_fit = np.tan(np.radians(alpha_values))
    ax.plot(alpha_values, model_fixed_p(x_fit, C_fit), 
            linestyle="-", color='red', linewidth=1.5,
            label=fr"$CN = {C_fit:.2f} \tan(\alpha)^{{0.47}}$")            
    
    ax.grid(True, linestyle='dotted')
    ax.set_xlabel('Mean slope (°)')
    ax.set_ylabel('CN (MPa)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    class LogFormatterDecimal(LogFormatter): # to plot decimal values
        def __call__(self, x, pos=None):
            return f"{x:.2f}"
        
    class LogFormatterInteger(LogFormatter):
        def __call__(self, x, pos=None):
            return f"{x:.0f}"

    ax = plt.gca()
    formatter_dec = LogFormatterDecimal(base=10, labelOnlyBase=False)
    formatter_int = LogFormatterInteger(base=10, labelOnlyBase=False)

    ax.xaxis.set_major_formatter(formatter_int)
    ax.xaxis.set_minor_formatter(formatter_int)
    ax.yaxis.set_major_formatter(formatter_dec)
    ax.yaxis.set_minor_formatter(formatter_dec)

    ax.grid(True, which='both', linestyle='dotted', color='gray', alpha=0.6)

    plt.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / fr"CN_vs_slope_mw{mw}.pdf", bbox_inches='tight')
    plt.close(fig)
    print("CN_vs_slope saved")



def plot_uncertainties():

    uncertainty_dir = script_dir / ".." / ".." / "data" / "uncertainties"

    runs = {}
    for f in uncertainty_dir.glob("timeseries_*.csv"):
        name = f.stem.replace("timeseries_", "")
        df = pd.read_csv(f)

        if "obs_u_bed" not in df.columns:
            continue

        df = df.sort_values("date")
        runs[name] = df


    runs_A  = {k:v for k,v in runs.items() if "As18000" in k and "_A" in k}
    runs_As  = {k:v for k,v in runs.items() if "_A2.4" in k and "As" in k}
    runs_B  = {k:v for k,v in runs.items() if k.startswith("B")}

    # référence
    ref_name = "As18000_A2.4"   # à adapter
    df_ref = runs[ref_name]

    def compute_std(runs_subset, df_ref):
        vel_std = []
        tau_std = []

        for i in range(len(df_ref)):
            v_all = []
            t_all = []

            for name, df in runs_subset.items():
                v = df["obs_u_bed"].values
                t = df["obs_tau_b"].values
                v_all.append(v[i])
                t_all.append(t[i])

            vel_std.append(np.std(v_all))
            tau_std.append(np.std(t_all))

        return np.array(vel_std), np.array(tau_std)


    vel_ref = df_ref["obs_u_bed"].values
    tau_ref = df_ref["obs_tau_b"].values

    vel_std_A,  tau_std_A  = compute_std(runs_A, df_ref)
    vel_std_As,  tau_std_As= compute_std(runs_As, df_ref)
    vel_std_B,  tau_std_B  = compute_std(runs_B, df_ref)


    fig, ax = plt.subplots(figsize=(10,7))

    ax.errorbar(vel_ref, tau_ref,
                xerr=vel_std_A, yerr=tau_std_A,
                fmt='o', ecolor='blue', color='black',
                label='A', capsize=4)

    ax.errorbar(vel_ref, tau_ref,
                xerr=vel_std_As, yerr=tau_std_As,
                fmt='o', ecolor='green', color='black',
                label='As', capsize=4)

    ax.errorbar(vel_ref, tau_ref,
                xerr=vel_std_B, yerr=tau_std_B,
                fmt='o', ecolor='red', color='black',
                label='Bedrock', capsize=4)

    ax.grid(True, linestyle='--')
    ax.tick_params(labelsize=22, width=0.9)
    ax.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$', fontsize=24)
    ax.set_ylabel('Basal shear stress (MPa)', fontsize=24)
    ax.set_xlim(25, 110)
    # ax.set_ylim(0.096, 0.119)
    ax.legend(loc='best', fontsize=22)

    plt.tight_layout()
    fig.savefig(fig_dir / "uncertainties.pdf", bbox_inches='tight')
    plt.close()
    print("uncertainties saved")




def plot_spatial_friction_law(start_year=2000, nb_years=10, m=3):

    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
               0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]
    
    fig, ax = plt.subplots(figsize=(6, 5))

    vel_list, tau_list = [], []
    
    label_added = False

    for glacier_key, glacier_data in GLACIERS.items():
        
        for stake in glacier_data['xy_coords'].keys():

            if (stake == "Wheel") or (glacier_key == "Geb"): # only for comparison, not a studied point
                continue

            color = glacier_data['colors'][stake]

            # RAW DATA
            try:
                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                
                if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                    print(f"No data for {glacier_key}-{stake}")
                    continue

                # Filtrer période
                df = pd.read_csv(proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv")

                mask = (df["date"] >= start_year) & (df["date"] < start_year + nb_years)
                df_masked = df[mask]
                vel_period, tau_period = df_masked["obs_u_bed"], df_masked["obs_tau_b"]
                    
                print(f"  {glacier_key} {stake} --> pts in [{start_year}, {start_year+nb_years}) : {mask.sum()}")

                if mask.sum() == 0:
                    print("  → SKIP: no data in period")
                    continue

                # Moyennes
                mean_vel = vel_period.mean(skipna=True)
                mean_tau = tau_period.mean(skipna=True)

                ax.scatter(mean_vel, mean_tau, color=color, edgecolor='k', marker='o',
                           label = f"{start_year} - {start_year+nb_years}" if not label_added else None, zorder=10)
                
                label_added = True
                
                label = f"{glacier_key} {stake}"

                if stake in ["101", "tac", "ech"]:
                    ax.text(0.84*mean_vel, 0.94*mean_tau, label, fontsize=9, ha='left')
                elif stake in ["4", "5", "B4"] and glacier_key!="Gie":
                    ax.text(1.1*mean_vel, 0.99*mean_tau, label, fontsize=9, ha='left')
                elif stake in ["102"]:
                    ax.text(0.55*mean_vel, 0.99*mean_tau, label, fontsize=9, ha='left')
                else:
                    ax.text(0.84*mean_vel, 1.03*mean_tau, label, fontsize=9, ha='left')

                vel_list.append(mean_vel), tau_list.append(mean_tau)
                    
            except Exception as e:
                print(f"Skip {glacier_key}-{stake} (raw): {e}")
                continue

            # FIT DATA
            if glacier_key != "StSo":
                try:
                    fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"

                    if not Path(fit_file).exists():
                        print(f"Missing fit file {glacier_key}-{stake}")
                        continue

                    df_fit = pd.read_csv(fit_file)
                    ax.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, alpha=0.4, linewidth=1)

                except Exception as e:
                    print(f"Skip fit {glacier_key}-{stake}: {e}")


    vel_arr = np.asarray(vel_list)
    tau_arr = np.asarray(tau_list)

    mask = np.isfinite(vel_arr) & np.isfinite(tau_arr)

    vel_arr = vel_arr[mask]
    tau_arr = tau_arr[mask]

    if len(vel_arr) < 3:
        print("→ SKIP global fit: not enough valid points")
        return
    
    res = fit_weertman_law(vel=vel_arr, tau=tau_arr, initial_guess=[20000, 3])
    m_fit = res["m"]
    As_fit = res["As"]
    vel_fit = res["vel_fit"]
    tau_fit = res["tau_fit"]

    ax.plot(vel_fit, tau_fit, color='k', linewidth = 1, label = f'm = {m_fit:.0f}')


    # Labels
    ax.legend()
    ax.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$')
    ax.set_ylabel(r'Basal shear stress (MPa)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(2, 200)
    ax.set_ylim(0.04, 0.17)
    ax.set_xticks([x for x in x_ticks if ax.get_xlim()[0] <= x <= ax.get_xlim()[-1]])
    ax.set_yticks([y for y in y_ticks if ax.get_ylim()[0] <= y <= ax.get_ylim()[-1]])

    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.grid(which='both', linestyle='dotted')
    
    fig.savefig(fig_dir / f"spatial_friction_laws_main_m{m}.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("spatial_friction_laws_main saved")



if __name__ == "__main__":
    plot_surface_vel_timeseries()
    plot_thk_changes_timeseries()
    plot_glaciers_longit_cs()
    plot_friction_laws(1)
    plot_friction_laws(3)
    plot_friction_laws(6)
    plot_CN_vs_slope(1)
    plot_CN_vs_slope(3)
    plot_CN_vs_slope(6)    
    plot_uncertainties()
    plot_spatial_friction_law()

    for glacier_key, glacier_data in GLACIERS.items():
        for stake in glacier_data['xy_coords'].keys():
            result = get_friclaw_params(glacier_key, stake, mw=3)
            if result is None:
                print(f"[SKIP] {glacier_key}-{stake}: no params")
                continue
            CN_value, q_value, As_value, m_value = result
            print(f"{glacier_key} {stake}  CN = {CN_value:.2f}  As = {round(As_value, -2):.0f}")

            df = pd.read_csv(proc_data_dir / f"mw{1/3:.3f}" / f"{glacier_key}_all_data_{stake}.csv")

            # valid = df.dropna(subset=["slope"])
            # years_full = np.arange(valid["date"].iloc[0], valid["date"].iloc[-1] + 1)
            # slopes_full = np.degrees(np.arctan(np.interp(years_full, valid["date"], valid["slope"])))
            # mean_slope = slopes_full[(years_full >= 1960) & (years_full <= 1980)].mean()

            # valid = df.dropna(subset=["altitude"])
            # years_full = np.arange(valid["date"].iloc[0], valid["date"].iloc[-1] + 1)
            # alt_full = np.interp(years_full, valid["date"], valid["altitude"])
            # print(f"slope : {np.round(np.degrees(np.arctan(df['slope'])).mean(), 2)}, altitude : {np.round(alt_full.mean(), 0)}")
