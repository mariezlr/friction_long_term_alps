from utils import GLACIERS, fig_dir, get_friclaw_params
from friction_laws import *
from run_friction_fits import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
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
    left_panel = {'Cor': ['B4','A4'], 'Geb': ['ss','sup'], 'Gie': ['5'], 'GB': ['sup','inf'], 'StSo': ['B','C']}
    right_panel = {'All': ['101'], 'Arg': ['5','4'], 'Gie': ['102'], 'MDG': ['tac','trel','ech']}

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


plot_specs = [
    ("All", "101", 6, 1),
    ("Arg", "4", 0, 0),
    ("Arg", "5", 0, 1),
    ("Cor", "B4", 1, 0),
    ("Cor", "A4", 1, 1),
    ("Geb", "sup", 2, 0),
    ("Geb", "ss", 2, 1),
    ("Gie", "5", 3, 0),
    ("Gie", "102", 3, 1),
    ("GB", "inf", 4, 0),
    ("GB", "sup", 4, 1),
    ("MDG", "tac", 5, 0),
    ("MDG", "trel", 5, 1),
    ("MDG", "ech", 6, 0),
    ("StSo", "B", 7, 0),
    ("StSo", "C", 7, 1),
]

def plot_reglin_taub_thk(m=3):

    def plot_panel(ax, df, color, title, transform="thk4", extra_text=None):

        if df is None or len(df) == 0:
            return

        x = df["thickness"]

        if transform == "thk_slope":
            x = x * (df["slope"])

        y = df["tau_b_elmer"]

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) == 0:
            return

        ax.scatter(x, y, color=color, marker='o')

        if len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.poly1d(p)(xx), '--', linewidth=1.2, color=color)

        ax.set_title(title, fontsize=20)

        if extra_text:
            ax.text(
                0.5, -0.2, extra_text,
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=14, alpha=0.8
            )


    fig, axes = plt.subplots(8, 2, figsize=(12, 26))

    for glacier, stake, r, c in plot_specs:

        ax = axes[r, c]

        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"

        if not file.exists():
            print(f"[WARNING] missing file: {glacier}-{stake}")
            continue

        df = pd.read_csv(file)

        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"
        if glacier =="GB":
            trans, extra = "thk_slope", "Thickness * Slope"
        else:
            trans, extra = "thk", None
        plot_panel(ax, df, color, title, trans, extra)

    fig.supxlabel(r'Thickness (m)', fontsize=24)
    fig.supylabel(r'Basal shear stress (MPa)', fontsize=24)

    for ax in axes.ravel():
        ax.tick_params(axis='both', labelsize=18, width=0.9)
        ax.grid(True, linestyle='dotted')

    plt.tight_layout()
    fig.savefig(fig_dir / f"reglin_taub_thk_m{m}.pdf", dpi=200)
    print("reglin_taub_thk saved")


def plot_reglin_udef_thk4(m=3):

    def plot_panel(ax, df, color, title, transform="thk4", extra_text=None):

        if df is None or len(df) == 0:
            return

        x = df["thickness"]**4

        if transform == "thk4_slope":
            x = x * (df["slope"]**3)

        y = df["u_def_elmer"]

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) == 0:
            return

        ax.scatter(x, y, color=color, marker='o')

        if len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.poly1d(p)(xx), '--', linewidth=1.2, color=color)

        ax.set_title(title, fontsize=20)

        if extra_text:
            ax.text(
                0.5, -0.2, extra_text,
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=14, alpha=0.8
            )


    fig, axes = plt.subplots(8, 2, figsize=(12, 26))

    for glacier, stake, r, c in plot_specs:

        ax = axes[r, c]

        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"

        if not file.exists():
            print(f"[WARNING] missing file: {glacier}-{stake}")
            continue

        df = pd.read_csv(file)

        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"
        if glacier =="GB":
            trans, extra = "thk4_slope", "Thickness$^4$ * Slope$^3$"
        else:
            trans, extra = "thk4", None
        plot_panel(ax, df, color, title, trans, extra)

    fig.supxlabel(r'Thickness$^4$', fontsize=24)
    fig.supylabel(r'Deformation velocity $(m \cdot yr^{-1})$', fontsize=24)

    for ax in axes.ravel():
        ax.tick_params(axis='both', labelsize=18, width=0.9)
        ax.grid(True, linestyle='dotted')

    plt.tight_layout()    
    fig.savefig(fig_dir / f"reglin_udef_thk4_m{m}.pdf", dpi=200)
    print("reglin_udef_thk4 saved")



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
                vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                
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
            if glacier_key != "StSo":
                try:
                    fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"

                    if not Path(fit_file).exists():
                        print(f"Missing fit file {glacier_key}-{stake}")
                        continue

                    df_fit = pd.read_csv(fit_file)

                    ax1.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, linewidth=2)

                except Exception as e:
                    print(f"Skip fit {glacier_key}-{stake}: {e}")

            # PARAMS
            try:
                CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake, m)
            except Exception as e:
                print(f"Skip params {glacier_key}-{stake}: {e}")
                continue

            if stake != "ss":
                ax1.axhline(y=CN_value, color=color, linestyle='--')

            # NORMALIZED
            if (glacier_key == "Geb") or (glacier_key == "StSo"):
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
    ax1.set_ylim(0.01, 0.17)
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
    
    fig.savefig(fig_dir / f"friction_laws_main_m{m}.pdf", dpi=200)
    plt.close(fig)
    print("friction_laws_main saved")




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



def plot_methods_synthesis():

    # Palette
    C_INPUT   = "#c0392b"   # rouge — données d'entrée
    C_MODEL   = "#1a6b9a"   # bleu  — étapes de modélisation
    C_INTERP  = "#e67e22"   # orange — interpolation temporelle
    C_OBS     = "#7a3c9c"   # violet — séries continues observées/dérivées
    C_OUTPUT  = "#0f6b52"   # vert  — résultat final
    C_ARROW   = "#555555"

    def box(ax, x, y, w, h, col, title, sub1="", sub2="",
            style="normal", fontsize_title=12):
        """Dessine une boîte avec titre + sous-titres."""
        alpha_fill = 0.12 if style == "normal" else 0.22
        lw = 1.5 if style == "normal" else 2.0
        ls = "-" if style != "dashed" else "--"

        rect = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.01",
                            transform=ax.transAxes,
                            facecolor=col + f"{int(alpha_fill*255):02x}",
                            edgecolor=col, linewidth=lw,
                            linestyle=ls, clip_on=False)
        ax.add_patch(rect)

        # Titre
        n_subs = sum(1 for s in [sub1, sub2] if s)
        if n_subs == 0:
            ty = y + h * 0.50
        elif n_subs == 1:
            ty = y + h * 0.65
        else:
            ty = y + h * 0.75

        ax.text(x + w/2, ty, title,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=fontsize_title, fontweight="bold", color=col,
                clip_on=False)
        if sub1:
            sy1 = y + h * (0.42 if n_subs == 2 else 0.32)
            ax.text(x + w/2, sy1, sub1,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=col, alpha=0.92, clip_on=False)
        if sub2:
            ax.text(x + w/2, y + h * 0.15, sub2,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=col, alpha=0.85,
                    style="italic", clip_on=False)


    def arrow(ax, x0, y0, x1, y1, col=C_ARROW, lw=1.3, style="->",
            label="", label_side="right"):
        ax.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle=style, color=col,
                                    lw=lw, clip_on=False))
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            dx = 0.03 if label_side == "right" else -0.03
            ax.text(mx + dx, my, label,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color=col, alpha=0.85, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                            ec="none", alpha=0.7))


    def label_badge(ax, x, y, text, col):
        """Petit badge de colonne (discret / continu)."""
        ax.text(x, y, text,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))


    # Figure
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Zones de fond : DISCRET vs CONTINU ───────────────────────────────────────
    ax.add_patch(Rectangle((0.02, 0.52), 0.46, 0.48,
                            transform=ax.transAxes,
                            facecolor="#f8f0f0", edgecolor=C_INPUT,
                            linewidth=1.2, linestyle=":", clip_on=False, zorder=0))
    ax.add_patch(Rectangle((0.52, 0.34), 0.46, 0.34,
                            transform=ax.transAxes,
                            facecolor="#f7f3f0", edgecolor="#7a5a40",
                            linewidth=1.2, linestyle=":", clip_on=False, zorder=0))

    ### COLONNE GAUCHE — DISCRET (dates avec DEM)
    # [1] DEMs éparses
    box(ax, 0.04, 0.90, 0.42, 0.08, C_INPUT,
        r"Surface & bed DEMs",
        r"$z_s(x,y,t_{\mathrm{DEM}})$,  $z_b(x,y)$")

    # [2] Force balance model
    box(ax, 0.04, 0.72, 0.42, 0.12, C_MODEL,
        "Force balance with Elmer/Ice",
        r"Stokes solved at each $t_{\mathrm{DEM}}$")

    # [3a] tau_b discret
    box(ax, 0.04, 0.54, 0.15, 0.12, C_MODEL,
        r"$\mathbf{\tau_b(t_{\mathrm{DEM}})}$",
        "Basal shear stress",
        "Force balance residual")

    # [3b] u_def discret
    box(ax, 0.31, 0.54, 0.15, 0.12, C_MODEL,
        r"$\mathbf{u_\mathrm{def}(t_{\mathrm{DEM}})}$",
        "Deformational velocity")

    # [4a] SIA empirical relation
    box(ax, 0.04, 0.36, 0.15, 0.12, C_INTERP,
        r"$\mathbf{\tau_b (t) \approx f(H (t))}$",
        r"via empirical relation",
        r"$\tau_b \propto H$ (SIA)",
        style="normal")

    # [4b] SIA empirical relation
    box(ax, 0.31, 0.36, 0.15, 0.12, C_INTERP,
        r"$\mathbf{u_\mathrm{def} (t) \approx f(H (t))}$",
        r"via empirical relation",
        r"$u_\mathrm{def} \propto H^{n+1}$ (SIA)",
        style="normal")

    ### COLONNE DROITE — CONTINU (toutes les dates d'obs)
    # [5] Observations
    box(ax, 0.76, 0.36, 0.20, 0.30, C_OBS,
        "Field\nobservations")

    # [6a] H(t) continu
    box(ax, 0.54, 0.54, 0.15, 0.12, C_OBS,
        r"$\mathbf{H(t)}$",
        "Continuous timeseries\nof ice thickness")

    # [6b] u_surf continu
    box(ax, 0.54, 0.36, 0.15, 0.12, C_OBS,
        r"$\mathbf{u_\mathrm{surf}(t)}$",
        "Continuous timeseries\nof surface velocity ")


    ### RÉSULTAT FINAL — largeur totale
    # [7] u_bed continu
    box(ax, 0.31, 0.18, 0.15, 0.12, C_INTERP,
        r"$\mathbf{u_\mathrm{bed}(t)}$", 
        r"$=$", 
        r"$u_\mathrm{surf}(t) - u_\mathrm{def}(t)$",
        style="normal")

    # [8] Final friction law
    box(ax, 0.10, 0.03, 0.80, 0.12, C_OUTPUT,
        r"Friction law :  $\mathbf{\tau_b = f(u_b)}$",
        r"Fit  $\tau_b(t)$ vs $u_\mathrm{bed}(t)$  with Weertman- and Lliboutry-type laws",
        style="normal", fontsize_title=14)

    # Flèches

    arrow(ax, 0.25, 0.90, 0.25, 0.84, col=C_INPUT)  # DEM → force balance

    arrow(ax, 0.15, 0.72, 0.15, 0.66, col=C_MODEL)  # Force balance → u_def
    arrow(ax, 0.35, 0.72, 0.35, 0.66, col=C_MODEL)  # Force balance → tau_b

    arrow(ax, 0.76, 0.60, 0.69, 0.60, col=C_OBS)    # Field obs → H
    arrow(ax, 0.76, 0.42, 0.69, 0.42, col=C_OBS)    # Field obs → u_surf

    arrow(ax, 0.35, 0.54, 0.35, 0.48, col=C_INTERP) # u_def → u_def interp
    arrow(ax, 0.54, 0.54, 0.36, 0.48, col=C_OBS)    # H → u_def interp

    arrow(ax, 0.15, 0.54, 0.15, 0.48, col=C_INTERP) # tau_b → tau_b interp
    arrow(ax, 0.54, 0.54, 0.16, 0.48, col=C_OBS)    # H → tau_b interp

    arrow(ax, 0.35, 0.36, 0.35, 0.30, col=C_INTERP) # u_def interp → u_bed interp
    arrow(ax, 0.54, 0.36, 0.36, 0.30, col=C_OBS)    # u_surf obs → u_bed interp

    arrow(ax, 0.15, 0.36, 0.36, 0.14, col=C_OUTPUT) # tau_b continu → friction law
    arrow(ax, 0.35, 0.18, 0.38, 0.14, col=C_OUTPUT) # u_bed continu → friction law



    # Annotations
    ax.annotate("",
                xy=(0.20, 0.58), xytext=(0.30, 0.58),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color=C_MODEL,
                                lw=1.4, clip_on=False))

    ax.text(0.25, 0.60, r"$u_b = A_s \tau_b^m$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color=C_MODEL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                    ec=C_MODEL, alpha=0.85))

    ax.text(0.25, 0.42, "Temporal\ninterpolation",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color=C_INTERP, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                    ec=C_INTERP, alpha=0.85))

    # Badge "few dates"
    label_badge(ax, 0.25, 0.992, "few dates  $t_\\mathrm{DEM}$\nfull spatial coverage", C_INPUT)
    # Badge "continuous"
    label_badge(ax, 0.86, 0.67, "many dates t\nsparse spatial sampling", "#7a5a40")

    plt.tight_layout(pad=0.3)
    fig.savefig(fig_dir / "methods_synthesis.pdf", bbox_inches="tight", dpi=200)
    print("methods_synthesis saved")


if __name__ == "__main__":
    # plot_surface_vel_timeseries()
    # plot_thk_changes_timeseries()
    # plot_glaciers_longit_cs()
    plot_reglin_taub_thk(3)
    plot_reglin_udef_thk4(3)
    plot_friction_laws(1)
    plot_friction_laws(3)
    plot_friction_laws(6)
    # plot_taub_vs_slope()
    # plot_methods_synthesis()
