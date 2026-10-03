"""
Figures of the manuscript.
"""
import sys
from pathlib import Path

# Make src/ importable
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, data_dir, geom_data_dir, proc_data_dir, get_friclaw_params
from friction_laws import calcul_normalised_friction_law, scaled_friction_law, fit_weertman_law
from run_friction_fits import compile_vel_tau_timeseries, run_uncertainty_fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from collections import OrderedDict
from matplotlib.ticker import LogFormatter, FixedLocator, NullLocator, FormatStrFormatter
from scipy.optimize import curve_fit

script_dir = Path(__file__).resolve().parent

plt.rcParams["lines.linewidth"] = 0.9
plt.rcParams.update({"font.size": 12,
                     "axes.labelsize": 14,
                     "axes.titlesize": 14,
                     "legend.fontsize": 12,
                     "xtick.labelsize": 12,
                     "ytick.labelsize": 12
                    })

VEL_LABEL = r'Basal sliding velocity (m yr$^{-1}$)'
TAU_LABEL = 'Basal shear stress (MPa)'


# ============================================================================
# OBSERVED TIMESERIES
# ============================================================================

def plot_surface_vel_timeseries():
    """Observed surface velocity timeseries at all stakes."""
    left_panel = {'Cor': ['B4', 'A4'], 'Geb': ['ss', 'sup'], 'Gie': ['5'], 'GB': ['sup', 'inf'], 'StSo': ['B', 'C']}
    right_panel = {'All': ['101'], 'Arg': ['5', '4'], 'Gie': ['102'], 'MDG': ['tac', 'trel', 'ech']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [1, 1]})

    def plot_panel(ax, panel_dict):
        for glacier, stakes in panel_dict.items():
            full_name = GLACIERS[glacier]['full_name']
            for stake in stakes:
                df = pd.read_csv(data_dir / "obs_raw" / f"{glacier}_vel_{stake}.csv")
                color = GLACIERS[glacier]['colors'][stake]

                mask = ~df['velocity'].isna()
                ax.plot(df['date'][mask], df['velocity'][mask], marker='o', color=color,
                        linestyle='-', label=f"{full_name} {stake}")

        ax.legend()
        ax.grid(True, linestyle='dotted')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel(r'Surface velocity (m yr$^{-1}$)', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plot_panel(ax1, left_panel)
    plot_panel(ax2, right_panel)

    for ax, letter in zip((ax1, ax2), ('(a)', '(b)')):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    fig.savefig(fig_dir / "Fig_4_timeseries_surface_vel.pdf", bbox_inches='tight')
    plt.close(fig)
    print("timeseries_surface_vel saved")


def plot_thk_changes_timeseries():
    """Observed thickness change timeseries at all stakes, relative to the first measurement."""
    left_panel = {'Geb': ['ss', 'sup'], 'GB': ['sup', 'inf'], 'MDG': ['tac', 'trel', 'ech'], 'StSo': ['B', 'C']}
    right_panel = {'All': ['101'], 'Arg': ['5', '4'], 'Cor': ['B4', 'A4'], 'Gie': ['5', '102']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [1, 1]})

    def plot_panel(ax, panel_dict):
        for glacier, stakes in panel_dict.items():
            full_name = GLACIERS[glacier]['full_name']
            for stake in stakes:
                df = pd.read_csv(data_dir / "obs_raw" / f"{glacier}_alt_{stake}.csv")
                color = GLACIERS[glacier]['colors'][stake]

                mask = ~df['altitude'].isna()
                ax.plot(df['date'][mask], df['altitude'][mask] - df['altitude'][mask].iloc[0],
                        marker='o', color=color, linestyle='-', label=f"{full_name} {stake}")

        ax.legend()
        ax.grid(True, linestyle='dotted')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Thickness change (m)', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plot_panel(ax1, left_panel)
    plot_panel(ax2, right_panel)

    for ax, letter in zip((ax1, ax2), ('(a)', '(b)')):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    fig.savefig(fig_dir / "Fig_3_timeseries_thk_changes.pdf", bbox_inches='tight')
    plt.close(fig)
    print("timeseries_thk_changes saved")


# ============================================================================
# GLACIER GEOMETRY
# ============================================================================
# Display names of stakes, when different from their identifier
STAKE_DISPLAY = {
    ("MDG", "tac"): "Tacul",
    ("MDG", "trel"): "Trélaporte",
    ("MDG", "ech"): "Echelets",
    ("Geb", "ss"): "SS'",
    ("Geb", "sup"): "Sup",
    ("GB", "inf"): "Inf",
    ("GB", "sup"): "Sup",
}

def stake_name(glacier, stake):
    return STAKE_DISPLAY.get((glacier, stake), stake)

def plot_glaciers_longit_cs():
    """Outlines, flowlines and longitudinal profiles of all glaciers."""
    order = ['All', 'Gie', 'Arg', 'GB', 'Cor', 'MDG', 'Geb', 'StSo']
    GLACIERS_sorted = OrderedDict((k, GLACIERS[k]) for k in order if k in GLACIERS)

    n_glaciers = len(GLACIERS_sorted)
    n_rows = (n_glaciers * 2 + 3) // 4  # two panels per glacier, four columns
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, 15))
    axes = axes.ravel()

    for i, (glacier_name, glacier_data) in enumerate(GLACIERS_sorted.items()):
        glacier_full_name = glacier_data['full_name']
        df_outlines = pd.read_csv(glacier_data['outlines_file'], sep=r"\s+", header=None)
        df_flowline = pd.read_csv(glacier_data['flowline'], sep=',', header=0)
        df_longit_cs = pd.read_csv(glacier_data['longit_cs'])
        years = glacier_data['years_DEM']
        points = glacier_data['xy_coords']
        flowline_idx = glacier_data['flowline_idx']
        colors = glacier_data['colors']
        avg_dist = glacier_data['avg_dist']

        ax_outlines = axes[2*i]
        ax_longit = axes[2*i + 1]

        # Map view: outline, stakes and flowline
        ax_outlines.plot(df_outlines.iloc[:, 0], df_outlines.iloc[:, 1], 'k-')
        if points:
            ax_outlines.scatter(*zip(*points.values()), c=list(colors.values()), s=80, edgecolors='black', zorder=3)
            for label, (x, y) in points.items():
                ax_outlines.annotate(stake_name(glacier_name, label), (x, y), xytext=(10, 10),
                                     textcoords="offset points", ha='right', fontsize=12,
                                     color=colors[label])
        ax_outlines.plot(df_flowline.iloc[:, 0], df_flowline.iloc[:, 1], color='r', label='Smooth flowline')
        ax_outlines.set_aspect('equal')
        ax_outlines.add_artist(ScaleBar(1, location='lower right'))
        for spine in ax_outlines.spines.values():
            spine.set_visible(False)
        ax_outlines.set_xticks([])
        ax_outlines.set_yticks([])
        ax_outlines.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_outlines.annotate('N', xy=(0.9, 0.95), xytext=(0.9, 0.85),
                             arrowprops=dict(facecolor='black', arrowstyle='-|>'),
                             ha='center', va='center', fontsize=16, xycoords='axes fraction')
        ax_outlines.set_title(glacier_full_name, fontsize=22)

        # Longitudinal profile, with the averaging zone around each stake
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
            ax_longit.annotate(stake_name(glacier_name, label), xy=(x_dist, y_alt),
                               xytext=(15, 15), textcoords="offset points",
                               arrowprops=dict(facecolor='k', arrowstyle='->'))

        ax_longit.set_ylabel('Altitude (m)', fontsize=18)
        ax_longit.legend(loc='lower left' if glacier_name in ['MDG', 'Arg'] else 'best')
        ax_longit.grid(True, linestyle='dotted')
        ax_longit.set_title(glacier_full_name, fontsize=22)

    plt.tight_layout()
    fig.savefig(fig_dir / "Fig_2_longitudinal_cuts.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("longitudinal_cuts saved")


# ============================================================================
# FRICTION LAWS
# ============================================================================

def plot_friction_laws(m=3):
    """(a) Basal shear stress versus sliding velocity with fitted laws; (b) normalised law."""
    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
               0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    for glacier_key, glacier_data in GLACIERS.items():
        for stake in glacier_data['xy_coords'].keys():

            if stake == "Wheel":  # comparison point only, not studied
                continue

            color = glacier_data['colors'][stake]
            marker = glacier_data['markers'][stake]
            label = f"{glacier_data['full_name']} {stake}"

            # Data
            try:
                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                    print(f"No data for {glacier_key} {stake}")
                    continue

                if marker == '2':  # unfilled marker, no edge colour
                    ax1.scatter(vel, tau, color=color, marker=marker, label=label, zorder=10)
                else:
                    ax1.scatter(vel, tau, color=color, edgecolor='k', marker=marker, label=label, zorder=10)

            except Exception as e:
                print(f"Skip {glacier_key} {stake} (data): {e}")
                continue

            # Fitted law (none for Saint-Sorlin)
            if glacier_key not in ["StSo"]:
                fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"
                if not fit_file.exists():
                    print(f"Missing fit file {glacier_key} {stake}")
                    continue
                df_fit = pd.read_csv(fit_file)
                ax1.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, linewidth=2)

            CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake, m)

            # Normalised law, hard-bedded sites only
            if glacier_key in ["Geb", "StSo"]:
                continue

            try:
                vel_norm, tau_norm = calcul_normalised_friction_law(vel, tau, CN_value, As_value, m_value)
                if len(vel_norm) == 0:
                    continue
                ax2.scatter(vel_norm, tau_norm, color=color, edgecolor='k', marker=marker, label=label)

            except Exception as e:
                print(f"Skip {glacier_key} {stake} (normalised): {e}")

    # Theoretical laws
    V_values = np.arange(0.05, 50, 0.1)
    ax2.plot(V_values, [scaled_friction_law(u, 1) for u in V_values], color='k', label='Lliboutry-type law')
    ax2.plot(np.arange(0.05, 1.5, 0.1), np.arange(0.05, 1.5, 0.1), 'b--', label='Weertman-type law')

    ax1.set_xlabel(VEL_LABEL)
    ax1.set_ylabel(TAU_LABEL)
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

    for ax, letter in zip((ax1, ax2), ('(a)', '(b)')):
        ax.text(-0.05, 1.08, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')

    # Common legend below both panels
    plt.tight_layout()
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=16)
    fig.subplots_adjust(bottom=0.28)

    fig.savefig(fig_dir / f"Fig_8_friction_laws_main_m{m}.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("friction_laws_main saved")


# ============================================================================
# SENSITIVITY EXPERIMENTS (ARGENTIÈRE PROFILE 4)
# ============================================================================

def plot_uncertainties():
    """Friction laws obtained for each sensitivity experiment, compared to the reference."""
    uncertainty_dir = script_dir / ".." / ".." / "data" / "uncertainties"

    runs = {}
    for f in sorted(uncertainty_dir.glob("timeseries_*.csv")):
        name = f.stem.replace("timeseries_", "")
        df = pd.read_csv(f)
        if "obs_u_bed" not in df.columns:
            continue
        runs[name] = df.sort_values("date")

    # Reference simulation
    df_ref = runs["As18000_A2.4"]
    vel_ref = df_ref["obs_u_bed"].values
    tau_ref = df_ref["obs_tau_b"].values
    ref_fit = pd.read_csv(proc_data_dir / f"mw{1/3:.3f}" / "friction_fits" / "Arg_4_friclaw_ts.csv")

    # Group runs by perturbation type, sorted by parameter value
    runs_A = {k: v for k, v in runs.items() if "As18000" in k and "_A" in k}
    runs_As = {k: v for k, v in runs.items() if "_A2.4" in k and "As" in k}
    runs_B = {k: v for k, v in runs.items() if k.startswith("B")}
    runs_As_fields = {k: v for k, v in runs.items() if k.startswith("as_")}
    runs_A = dict(sorted(runs_A.items(), key=lambda kv: float(kv[0].split('_A')[1])))
    runs_As = dict(sorted(runs_As.items(), key=lambda kv: float(kv[0].split('_')[0][2:])))

    runs_by_type = {"A": runs_A, "As": runs_As, "B": runs_B, "As_fields": runs_As_fields}
    titles = {"A": "A variations", "As": "As variations",
              "B": "Perturbation of B fields", "As_fields": "Perturbation of As fields"}

    # Fits are computed once and then read from file
    fit_dir = proc_data_dir / "uncertainty_fits"
    summary_file = fit_dir / "uncertainty_fit_summary.csv"
    run_uncertainty_fits(runs_by_type)
    summary = pd.read_csv(summary_file, header=[0, 1], index_col=0)

    cmaps = ["Blues", "Reds", "Greens", "Purples"]
    AS_UNIT = r"\mathrm{m\,yr^{-1}\,MPa^{-3}}"
    CN_UNIT = r"\mathrm{MPa}"

    label_funcs = [
        lambda name: rf"$A = {name.split('_A')[1]}$",               # As18000_A2.2 -> A = 2.2
        lambda name: rf"$A_s = {name.split('_')[0][2:]}$",          # As7200_A2.4  -> As = 7200
        lambda name: f"B field n°{name[1:]}",                       # B1           -> B field n°1
        lambda name: f"As field n°{1 + int(name.split('_')[1])}",   # as_0         -> As field n°1
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.1, "hspace": 0.35})

    for i, (ax, (ptype, runs_subset), cmap_name, make_label) in enumerate(zip(
            axes.flat, runs_by_type.items(), cmaps, label_funcs)):

        # One colour map per panel, shade increasing with the parameter value
        shades = plt.get_cmap(cmap_name)(np.linspace(0.45, 0.95, len(runs_subset)))

        for (name, df), c in zip(runs_subset.items(), shades):
            ax.scatter(df["obs_u_bed"].values, df["obs_tau_b"].values,
                       color=c, s=50, label=make_label(name))

            fit_file = fit_dir / f"{ptype}_{name}_friclaw_ts.csv"
            if fit_file.exists():
                df_fit = pd.read_csv(fit_file)
                ax.plot(df_fit["vel_fit"], df_fit["tau_fit"], color=c, linewidth=1.5)

        ax.scatter(vel_ref, tau_ref, color="black", s=50, label="Reference")
        ax.plot(ref_fit["vel_fit"], ref_fit["tau_fit"], color="black", linewidth=2)

        # Title, with mean and standard deviation of the fitted parameters as subtitle
        s = summary.loc[ptype]
        ax.set_title(titles[ptype], fontsize=22, fontweight="bold", pad=45)
        ax.text(0.5, 1.02,
                rf"$A_{{s_R}} = {s[('As', 'mean')]:.0f} \pm {s[('As', 'std')]:.0f}\ {AS_UNIT}$"
                rf" and $CN = {s[('CN', 'mean')]:.3f} \pm {s[('CN', 'std')]:.3f}\ {CN_UNIT}$",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=18)
        ax.text(0.02, 0.98, f"({'abcd'[i]})", transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left")

        ax.grid(True, linestyle="--")
        ax.tick_params(labelsize=22, width=0.9)
        ax.set_xlim(21, 114)
        ax.set_ylim(0.088, 0.125)
        ax.legend(loc="lower right", fontsize=18, ncols=2)

    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel(VEL_LABEL, fontsize=24)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel(TAU_LABEL, fontsize=24)

    fig.savefig(fig_dir / "Fig_10_uncertainties.pdf", bbox_inches='tight')
    plt.close(fig)
    print("uncertainties saved")


def compute_CN_uncertainty():
    """
    Standard deviation of CN for each perturbation type, and their
    combination in quadrature (assuming independent effects).
    """
    params_df = pd.read_csv(proc_data_dir / "uncertainty_fits" / "uncertainty_fit_params.csv")

    std_CN = params_df.groupby("perturbation")["CN"].std()
    sigma_tot = np.sqrt((std_CN**2).sum())

    for ptype, s in std_CN.items():
        print(f"{ptype:10s}: std = {s:.2e} MPa, 95% CI = ±{1.96*s:.2e} MPa")
    print(f"{'Total':10s}: std = {sigma_tot:.2e} MPa, 95% CI = ±{1.96*sigma_tot:.2e} MPa")

    return std_CN, sigma_tot


# ============================================================================
# SPATIAL VARIABILITY
# ============================================================================

def plot_CN_vs_slope(mw=3):
    """CN as a function of mean surface slope, with a fit of the form CN = C tan(alpha)^0.47."""
    fig, ax = plt.subplots(figsize=(6, 5))

    slopes_deg, CN_values = [], []

    # Uncertainty estimated at Argentière Profile 4, applied to all sites (±1 std)
    _, sigma_tot = compute_CN_uncertainty()

    df_slopes = pd.read_csv(geom_data_dir / 'slopes/mean_slopes.csv', sep=",")

    for glacier_key, glacier_data in GLACIERS.items():

        if glacier_key in ["Geb", "StSo"]:  # soft-bedded or poorly constrained
            continue

        for stake in glacier_data['xy_coords'].keys():

            row = df_slopes[(df_slopes['glacier'] == glacier_key) & (df_slopes['stake'] == stake)]
            slope_deg = row['mean_slope_deg_6080'].values

            if stake == "Wheel":
                CN = 0.217  # prescribed value, not fitted in this study
            else:
                params = get_friclaw_params(glacier_key, stake, mw=mw)
                if params is None:
                    print(f"[SKIP] {glacier_key} {stake}: no parameters")
                    continue
                CN = params[0]

            slopes_deg.append(slope_deg[0])
            CN_values.append(CN)

            color = GLACIERS[glacier_key]['colors'][stake]
            label = f"{glacier_key} {stake}"

            ax.scatter(slope_deg, CN, c=color, zorder=2)
            ax.errorbar(slope_deg, CN, yerr=sigma_tot,
                        fmt='none', ecolor='gray', capsize=3, zorder=1)

            # Stake labels, offset by hand to avoid overlaps
            if stake == "A4":
                ax.text(0.92*slope_deg, 0.94*CN, label, fontsize=9, ha='left')
            elif stake == "4":
                ax.text(0.78*slope_deg, CN, label, fontsize=9, ha='left')
            elif stake == "trel":
                ax.text(1.07*slope_deg, CN, label, fontsize=9, ha='left')
            else:
                ax.text(0.92*slope_deg, 1.03*CN, label, fontsize=9, ha='left')

    # Exponent prescribed by the Röthlisberger channel model, prefactor fitted
    tan_slopes = np.tan(np.radians(slopes_deg))

    def model_fixed_p(x, C):
        return C * x**0.47

    popt, pcov = curve_fit(model_fixed_p, tan_slopes, CN_values, p0=[0.3])
    C_fit = popt[0]

    alpha_values = np.arange(2, np.max(slopes_deg), 0.001)
    x_fit = np.tan(np.radians(alpha_values))
    ax.plot(alpha_values, model_fixed_p(x_fit, C_fit),
            linestyle="-", color='red', linewidth=1.5,
            label=fr"$CN = {C_fit:.2f} \tan(\alpha)^{{0.47}}$")

    ax.set_xlabel('Mean slope (°)')
    ax.set_ylabel('CN (MPa)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Integer labels on the log x-axis
    class LogFormatterInteger(LogFormatter):
        def __call__(self, x, pos=None):
            return f"{x:.0f}"

    formatter_int = LogFormatterInteger(base=10, labelOnlyBase=False)
    ax.xaxis.set_major_formatter(formatter_int)
    ax.xaxis.set_minor_formatter(formatter_int)

    yticks = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(0.055, 0.21)

    ax.grid(True, which='both', linestyle='dotted', color='gray', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / f"Fig_11_CN_vs_slope_mw{mw}.pdf", bbox_inches='tight')
    plt.close(fig)
    print("CN_vs_slope saved")


def plot_spatial_friction_law(start_year=2000, nb_years=10, m=3):
    """
    Basal shear stress versus sliding velocity averaged over a given period
    at all sites, with a Weertman-type fit across sites.
    """
    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
               0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]

    fig, ax = plt.subplots(figsize=(6, 5))

    vel_list, tau_list = [], []
    label_added = False

    for glacier_key, glacier_data in GLACIERS.items():
        for stake in glacier_data['xy_coords'].keys():

            if (stake == "Wheel") or (glacier_key == "Geb"):
                continue

            color = glacier_data['colors'][stake]

            try:
                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                    print(f"No data for {glacier_key} {stake}")
                    continue

                # Mean velocity and stress over the period
                df = pd.read_csv(proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv")
                mask = (df["date"] >= start_year) & (df["date"] < start_year + nb_years)

                if mask.sum() == 0:
                    print(f"No data for {glacier_key} {stake} in [{start_year}, {start_year + nb_years})")
                    continue

                mean_vel = df.loc[mask, "obs_u_bed"].mean(skipna=True)
                mean_tau = df.loc[mask, "obs_tau_b"].mean(skipna=True)

                ax.scatter(mean_vel, mean_tau, color=color, edgecolor='k', marker='o',
                           label=f"{start_year} - {start_year + nb_years}" if not label_added else None,
                           zorder=10)
                label_added = True

                # Stake labels, offset by hand to avoid overlaps
                label = f"{glacier_key} {stake}"
                if stake in ["101", "tac", "ech"]:
                    ax.text(0.84*mean_vel, 0.94*mean_tau, label, fontsize=9, ha='left')
                elif stake in ["4", "5", "B4"] and glacier_key != "Gie":
                    ax.text(1.1*mean_vel, 0.99*mean_tau, label, fontsize=9, ha='left')
                elif stake in ["102"]:
                    ax.text(0.55*mean_vel, 0.99*mean_tau, label, fontsize=9, ha='left')
                else:
                    ax.text(0.84*mean_vel, 1.03*mean_tau, label, fontsize=9, ha='left')

                vel_list.append(mean_vel)
                tau_list.append(mean_tau)

            except Exception as e:
                print(f"Skip {glacier_key} {stake} (data): {e}")
                continue

            # Fitted law at each site, for reference
            if glacier_key != "StSo":
                fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"
                if not fit_file.exists():
                    print(f"Missing fit file {glacier_key} {stake}")
                    continue
                df_fit = pd.read_csv(fit_file)
                ax.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, alpha=0.4, linewidth=1)

    vel_arr = np.asarray(vel_list)
    tau_arr = np.asarray(tau_list)
    mask = np.isfinite(vel_arr) & np.isfinite(tau_arr)
    vel_arr, tau_arr = vel_arr[mask], tau_arr[mask]

    if len(vel_arr) < 3:
        print("Not enough points for the fit across sites")
        return

    # Weertman-type fit across sites
    res = fit_weertman_law(vel=vel_arr, tau=tau_arr, initial_guess=[20000, 3])
    ax.plot(res["vel_fit"], res["tau_fit"], color='k', linewidth=1, label=f'm = {res["m"]:.0f}')

    ax.legend()
    ax.set_xlabel(VEL_LABEL)
    ax.set_ylabel(TAU_LABEL)
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

    fig.savefig(fig_dir / f"Fig_9_spatial_friction_laws_main_m{m}.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("spatial_friction_laws_main saved")


def print_friction_params(mw=3):
    """Print the fitted CN and As at each stake."""
    for glacier_key, glacier_data in GLACIERS.items():
        for stake in glacier_data['xy_coords'].keys():
            result = get_friclaw_params(glacier_key, stake, mw=mw)
            if result is None:
                print(f"[SKIP] {glacier_key} {stake}: no parameters")
                continue
            CN_value, q_value, As_value, m_value = result
            print(f"{glacier_key} {stake}  CN = {CN_value:.2f}  As = {round(As_value, -2):.0f}")


if __name__ == "__main__":
    plot_surface_vel_timeseries()
    plot_thk_changes_timeseries()
    plot_glaciers_longit_cs()
    plot_friction_laws(1)
    plot_friction_laws(3)
    plot_friction_laws(6)
    plot_uncertainties()
    plot_CN_vs_slope(1)
    plot_CN_vs_slope(3)
    plot_CN_vs_slope(6)
    plot_spatial_friction_law()
    print_friction_params()