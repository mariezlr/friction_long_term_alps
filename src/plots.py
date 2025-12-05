from utils import GLACIERS, fig_dir, get_friclaw_params
from friction_laws import *
from run_friction_fits import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
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
    left_panel = {'All': ['101'], 'Arg': ['5','4'], 'Gie': ['102'], 'MDG': ['tac','trel','ech']}
    right_panel = {'Cor': ['B4','A4'], 'Geb': ['ss','sup'], 'Gie': ['5'], 'GB': ['sup','inf'], 'StSo': ['B','C']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={'width_ratios':[1,1]})

    # Intern function to plot a panel
    def plot_panel(ax, panel_dict):

        for glacier, stakes in panel_dict.items():

            full_name = GLACIERS[glacier]['full_name']
            for stake in stakes:

                file = GLACIERS[glacier]['all_data'][stake]
                color = GLACIERS[glacier]['colors'][stake]
                marker = 'o'

                df = pd.read_csv(file)

                label = f"{full_name} {stake}"
                ax.plot(df['date'], df['velocity'], marker=marker, color=color,
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
    fig.savefig(fig_dir / "timeseries_surface_vel.pdf")
    plt.close(fig)


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

                file = GLACIERS[glacier]['all_data'][stake]
                color = GLACIERS[glacier]['colors'][stake]
                marker = 'o'

                df = pd.read_csv(file)

                label = f"{full_name} {stake}"
                ax.plot(df['date'], df['altitude'] - df['altitude'].dropna().iloc[0], marker=marker, color=color,
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
    fig.savefig(fig_dir / "timeseries_thk_changes.pdf")
    plt.close(fig)


def plot_glaciers_longit_cs():
    """
    Plot flowlines, outlines, and longitudinal profiles for all glaciers defined in config.GLACIERS.
    """
    global GLACIERS 
    order = ['All', 'Gie', 'Arg', 'GB', 'Cor', 'MDG', 'Geb', 'StSo']
    GLACIERS = OrderedDict((k, GLACIERS[k]) for k in order if k in GLACIERS)


    n_glaciers = len(GLACIERS)
    n_rows = (n_glaciers * 2 + 3) // 4  # 2 axes per glacier / 4 columns
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, 15))
    axes = axes.ravel()

    for i, (glacier_name, glacier_data) in enumerate(GLACIERS.items()):
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
    fig.savefig(fig_dir / "longitudinal_cuts.pdf", dpi=200)
    plt.close(fig)



def plot_friction_laws():
    """
    Plots friction observed friction laws with best fits for all glaciers, and normalised frcition laws.

    """

    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))
    
    for glacier_key, glacier_data in GLACIERS.items():
        
        for i, stake in enumerate(glacier_data['xy_coords'].keys()):

            if stake == "Wheel": # only for comparison, not a studied point
                continue

            color = glacier_data['colors'][stake]
            marker = glacier_data['markers'][stake]

            # raw data          
            vel, tau = compile_vel_tau_timeseries(glacier_key, stake)
            ax1.scatter(vel, tau, color=color, edgecolor='k', marker=marker, label=f"{glacier_data['full_name']} {stake}", zorder=10)

            if glacier_key == "StSo": # too few data points to have a reliable fit
                continue

            # fit data
            fit_file = glacier_data['friclaw_ts'][stake]
            df_fit = pd.read_csv(fit_file)
            
            vel_fit = df_fit['vel_fit'].values
            tau_fit = df_fit['tau_fit'].values
            
            ax1.plot(vel_fit, tau_fit, color=color, linewidth = 2)

            # CN value
            CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake)
            if stake!="ss":
                ax1.axhline(y=CN_value, color=color, linestyle='--')
                
            # Normalized version (ax2)
            if glacier_key == "Geb": # keeping only hard bed glaciers
                continue
                 
            vel_norm, tau_norm = calcul_normalised_friction_law(vel, tau, CN_value, As_value, m_value)
            
            ax2.scatter(vel_norm, tau_norm, color=color, edgecolor='k', marker=marker, label=f"{glacier_data['full_name']} {stake}")
    
    # Theoritical law
    V_values = np.arange(0.05,50,0.1)
    ax2.plot(V_values, [scaled_friction_law(u, 1) for u in V_values], color='k', label='cavitation law')
    ax2.plot(np.arange(0.05,1.5,0.1), np.arange(0.05,1.5,0.1), 'b--', label='Weertman-type law')
    
    # Labels
    ax1.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$')
    ax1.set_ylabel(r'Basal shear stress (MPa)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
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
    
    fig.savefig(fig_dir / "friction_laws_main.pdf", dpi=200)
    plt.close(fig)



def plot_taub_vs_slope():
    fig, ax = plt.subplots(figsize=(6,5))

    slopes_deg, CN_values = [], []

    for glacier_key, glacier_data in GLACIERS.items():
        
        if glacier_key in ["Geb", "StSo"]:   # ignoring soft-bedd and bad constrained glaciers
            continue

        slope_data = glacier_data['slope_60_80']

        for stake, slope_deg in slope_data.items():

            if stake == "Wheel":
                CN = 0.217
            else:
                CN = get_friclaw_params(glacier_key, stake)[0]

            slopes_deg.append(slope_deg[0])
            CN_values.append(CN)

            color = GLACIERS[glacier_key]['colors'][stake]
            marker = 'o'
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
    fig.savefig(fig_dir / "CN_vs_slope.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_surface_vel_timeseries()
    plot_thk_changes_timeseries()
    plot_glaciers_longit_cs()
    plot_friction_laws()
    plot_taub_vs_slope()
