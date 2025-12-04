from utils import GLACIERS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from collections import OrderedDict

script_dir = Path(__file__).resolve().parent

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
    fig.savefig(script_dir / "../../figures/timeseries_surface_vel.pdf")
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
    fig.savefig(script_dir / "../../figures/timeseries_thk_changes.pdf")
    plt.close(fig)


def plot_glaciers_longit_cs():
    """
    Plot flowlines, contours, and longitudinal profiles for all glaciers defined in config.GLACIERS.
    """
    global GLACIERS 
    order = ['All', 'Gie', 'Arg', 'GB', 'Cor', 'MDG', 'Geb', 'StSo']
    GLACIERS = OrderedDict((k, GLACIERS[k]) for k in order if k in GLACIERS)


    n_glaciers = len(GLACIERS)
    n_rows = (n_glaciers * 2 + 3) // 4  # deux axes par glacier, 4 colonnes
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, 15))
    axes = axes.ravel()

    for i, (glacier_name, glacier_data) in enumerate(GLACIERS.items()):
        # Read files
        glacier_full_name = glacier_data['full_name']
        df_contour = pd.read_csv(glacier_data['contour_file'], sep="\s+", header=None)
        df_flowline = pd.read_csv(glacier_data['flowline'], sep=',', header=0)
        df_longit_cs = pd.read_csv(glacier_data['longit_cs'])
        years = glacier_data['years_DEM']
        points = glacier_data['xy_coords']
        flowline_idx = glacier_data['flowline_idx']
        colors = glacier_data['colors']
        avg_dist = glacier_data['avg_dist']

        # Axes
        ax_contour = axes[2*i]
        ax_longit = axes[2*i + 1]

        # Points and flowline
        ax_contour.plot(df_contour.iloc[:,0], df_contour.iloc[:,1], 'k-')
        if points:
            ax_contour.scatter(*zip(*points.values()), c=list(colors.values()), s=80, edgecolors='black', zorder=3)
            for label, (x, y) in points.items():
                ax_contour.annotate(label, (x, y), xytext=(10,10), textcoords="offset points",
                                    ha='right', fontsize=12, color=colors[label])
        ax_contour.plot(df_flowline.iloc[:,0], df_flowline.iloc[:,1], color='r', label='Smooth flowline')
        ax_contour.set_title(glacier_name)
        ax_contour.set_aspect('equal')
        ax_contour.add_artist(ScaleBar(1, location='lower right'))
        for spine in ax_contour.spines.values():
            spine.set_visible(False)
        ax_contour.set_xticks([])
        ax_contour.set_yticks([])
        ax_contour.set_xlabel('')
        ax_contour.set_ylabel('')
        ax_contour.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)   
        ax_contour.annotate('N', xy=(0.9, 0.95), xytext=(0.9, 0.85), arrowprops=dict(facecolor='black', arrowstyle='-|>'), 
            ha='center', va='center', fontsize=16, xycoords='axes fraction')
        ax_contour.set_title(glacier_full_name, fontsize=22)

        # Longitudinal cross-section
        ax_longit.plot(df_longit_cs['dist'], df_longit_cs['z_bed'], color='k', label='Bedrock')
        for year in years:
            ax_longit.plot(df_longit_cs['dist'], df_longit_cs[f'z_surf_{year}'], label=str(year))

        for label, (x, y) in points.items():
            idx = glacier_data['flowline_idx'][label]
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
    fig.savefig(script_dir / "../../figures/longitudinal_cuts.pdf")
    plt.close(fig)
    

def plot_friction_laws():
    return True

def plot_taub_vs_slope():
    fig, ax = plt.subplots(figsize=(6,5))

filtered_df = mean_slopes.dropna(subset=['CN', 'slope(rad)']).reset_index(drop=True)
print(filtered_df)
filtered_df.to_csv("../data/mean_slopes.csv", index=False)

# Tracé des points
ax.scatter(filtered_df['slope(deg)'], filtered_df['CN'], c=filtered_df['color'], zorder=2)

# Ajout des barres d'erreur verticales (sur CN)
sigma_CN = 0.002112704740016016 # Voir notebook Uncertainty
ax.errorbar(filtered_df['slope(deg)'], filtered_df['CN'], 
             yerr=sigma_CN, 
             fmt='none', ecolor='gray', capsize=3, zorder=1)

# Ajouter des étiquettes provenant de la colonne 'name' filtrée
texts = []
for i, name in enumerate(filtered_df['name']):
        # Cas particuliers
    if name == "MDG Trélaporte":
        x = np.log10(0.55 * filtered_df['slope(deg)'].iloc[i])
        y = np.log10(1.03 * filtered_df['CN'].iloc[i])
    elif name == "MDG Tacul":
        x = np.log10(0.8 * filtered_df['slope(deg)'].iloc[i])
        y = np.log10(0.94 * filtered_df['CN'].iloc[i])
    elif name == "Corbassière A4":
        x = np.log10(0.95 * filtered_df['slope(deg)'].iloc[i])
        y = np.log10(0.94 * filtered_df['CN'].iloc[i])
    elif name == "Glacier Blanc sup" or name == "Glacier Blanc inf":
        x = np.log10(0.7 * filtered_df['slope(deg)'].iloc[i])
        y = np.log10(1.03 * filtered_df['CN'].iloc[i])
    else:
        x = np.log10(0.95 * filtered_df['slope(deg)'].iloc[i])
        y = np.log10(1.03 * filtered_df['CN'].iloc[i])
    
    texts.append(ax.text(10**x, 10**y, name, fontsize=9, ha='left'))


alpha_values = np.arange(2, np.max(mean_slopes['slope(deg)']), 0.001)

# Convertir les pentes en tangente et extraire uniquement les valeurs valides
mask = np.isfinite(mean_slopes['CN']) & np.isfinite(mean_slopes['slope(deg)'])
x = np.tan(np.deg2rad(mean_slopes.loc[mask, 'slope(deg)']))
y = mean_slopes.loc[mask, 'CN']

# Fonction modèle : CN = C * tan(alpha)^p
def model(x, C, p):
    return C * x**p

# Fit des deux paramètres (C et p)
popt, pcov = curve_fit(model, x, y, p0=[0.3, 0.47])
C_fit, p_fit = popt
perr = np.sqrt(np.diag(pcov))  # incertitudes 1σ
C_err, p_err = perr

print(f"Best fit: C = {C_fit:.3f} ± {C_err:.3f}, p = {p_fit:.3f} ± {p_err:.3f}")

# Tracé du fit
x_fit = np.tan(np.deg2rad(alpha_values))


# Fonction modèle avec p fixé
def model_fixed_p(x, C):
    p = 0.47  # exposant fixe
    return C * x**p

# Fit uniquement pour C
popt, pcov = curve_fit(model_fixed_p, x, y, p0=[0.3])
C_fit = popt[0]
C_err = np.sqrt(np.diag(pcov))[0]

print(f"Best fit avec p=0.47 : C = {C_fit:.3f} ± {C_err:.3f}")

# Tracé du modèle imposé p=0.47 pour comparaison
x_fit = np.tan(np.deg2rad(alpha_values))
ax.plot(alpha_values, model_fixed_p(x_fit, C_fit), 
        linestyle="-", color='red', linewidth=1.5,
        label=fr"$CN = {C_fit:.2f} \tan(\alpha)^{{0.47}}$")


ax.grid(True, linestyle='dotted')
ax.set_xlabel('Mean slope (°)')
ax.set_ylabel('CN (MPa)')

ax.set_xscale('log')
ax.set_yscale('log')

# Formatter log qui affiche les valeurs décimales "normales"
class LogFormatterDecimal(LogFormatter):
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
plt.show()

fig.savefig("../output/figures/CN_slopes.pdf")



if __name__ == "__main__":
    plot_surface_vel_timeseries()
    plot_thk_changes_timeseries()
    plot_glaciers_longit_cs()
    plot_friction_laws()
    plot_taub_vs_slope()
