"""
Supplementary figures: sensitivity to the friction law exponent m used in
Elmer/Ice, sensitivity to the averaging scale, and friction laws per stake.
"""
import sys
from pathlib import Path

# Make src/ importable
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, proc_data_dir, get_friclaw_params, plot_specs
from process_timeseries import process_glacier_stake
from friction_laws import calcul_normalised_friction_law, scaled_friction_law
from run_friction_fits import compile_vel_tau_timeseries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

VEL_LABEL = r'Basal sliding velocity (m yr$^{-1}$)'
TAU_LABEL = 'Basal shear stress (MPa)'
VEL_NORM_LABEL = r'Scaled sliding velocity $\frac{u_b}{A_s(CN)^m}$'
TAU_NORM_LABEL = r'Scaled shear stress $\left(\frac{\tau_b}{CN}\right)^m$'

X_TICKS = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
Y_TICKS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
           0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]


def _all_stakes():
    """All studied stakes, as (glacier, stake) pairs."""
    return [(gk, s)
            for gk, gd in GLACIERS.items()
            for s in gd['xy_coords'].keys()
            if s != "Wheel"]


def _scalar_log_axes(ax):
    """Log axes with plain decimal tick labels."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())


# ============================================================================
# SENSITIVITY TO m
# ============================================================================

def plot_friction_laws_comparison(m_values=(1, 3, 6)):
    """Friction laws (top) and normalised laws (bottom) for each value of m used in Elmer/Ice."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    legend_handles = {}  # one legend entry per stake

    for col, m in enumerate(m_values):
        ax1 = axes[0, col]
        ax2 = axes[1, col]

        for glacier_key, glacier_data in GLACIERS.items():
            for stake in glacier_data['xy_coords'].keys():
                if stake == "Wheel":
                    continue
                color = glacier_data['colors'][stake]
                marker = glacier_data['markers'][stake]
                label = f"{glacier_data['full_name']} {stake}".strip()

                # Data
                try:
                    date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                    if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                        continue
                    kwargs = dict(color=color, marker=marker, zorder=10, s=20)
                    if marker != '2':
                        kwargs['edgecolors'] = 'k'
                        kwargs['linewidths'] = 0.4
                    sc = ax1.scatter(vel, tau, **kwargs)
                    legend_handles.setdefault(label, sc)

                except Exception as e:
                    print(f"Skip {glacier_key} {stake} (m={m}): {e}")
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
                    ax2.scatter(vel_norm, tau_norm, color=color, marker=marker,
                                edgecolors='k', linewidths=0.4, s=20)
                except Exception as e:
                    print(f"Skip {glacier_key} {stake} (normalised, m={m}): {e}")

        # Theoretical laws
        V_values = np.arange(0.05, 50, 0.1)
        ax2.plot(V_values, [scaled_friction_law(u, 1) for u in V_values],
                 color='k', linewidth=1.2, label='Lliboutry-type law')
        ax2.plot(np.arange(0.05, 1.5, 0.1), np.arange(0.05, 1.5, 0.1),
                 'b--', linewidth=1.2, label='Weertman-type law')

        _scalar_log_axes(ax1)
        ax1.set_xlim(0.8, 200)
        ax1.set_ylim(0.01, 0.17)
        ax1.set_xticks([x for x in X_TICKS if ax1.get_xlim()[0] <= x <= ax1.get_xlim()[1]])
        ax1.set_yticks([y for y in Y_TICKS if ax1.get_ylim()[0] <= y <= ax1.get_ylim()[1]])
        ax1.grid(which='both', linestyle='dotted', alpha=0.5)
        ax1.set_title(f"m = {m}", fontsize=16, fontweight='bold')

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(which='both', linestyle='dotted', alpha=0.5)
        if col == 0:
            ax2.legend(fontsize=8, loc='upper left')

        # Axis labels on outer panels only
        if col == 0:
            ax1.set_ylabel(TAU_LABEL, fontsize=14, fontweight="bold")
            ax2.set_ylabel(TAU_NORM_LABEL, fontsize=14, fontweight="bold")
        ax1.set_xlabel(VEL_LABEL, fontsize=14, fontweight="bold")
        ax2.set_xlabel(VEL_NORM_LABEL, fontsize=14, fontweight="bold")

    fig.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()),
               loc='lower center', ncol=6, fontsize=15,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(fig_dir / "Fig_S7_friction_laws_comparison.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("friction_laws_comparison saved")


def plot_scatter_taub_mw():
    """Basal shear stress and sliding velocity obtained with m = 1 and m = 6, versus m = 3."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9),
                             gridspec_kw=dict(hspace=0.35, wspace=0.35))
    legend_handles = {}

    for col, m in enumerate([1, 6]):
        ax_tau = axes[0, col]
        ax_vel = axes[1, col]

        for glacier_key, glacier_data in GLACIERS.items():
            for stake in glacier_data['xy_coords'].keys():
                if stake == "Wheel":
                    continue
                color = glacier_data['colors'][stake]
                label = f"{glacier_data['full_name']} {stake}".strip()

                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                date_ref, vel_ref, tau_ref = compile_vel_tau_timeseries(glacier_key, stake, 3)

                if tau is None or tau.empty or tau_ref is None or tau_ref.empty:
                    continue

                # Match both runs on date
                d = pd.DataFrame({"date": date.values, "vel": vel.values, "tau": tau.values})
                d_ref = pd.DataFrame({"date": date_ref.values, "vel_ref": vel_ref.values, "tau_ref": tau_ref.values})
                d = d.merge(d_ref, on="date")

                sc = ax_tau.scatter(d["tau_ref"], d["tau"], c=color, s=15, alpha=0.7, label=label)
                ax_vel.scatter(d["vel_ref"], d["vel"], c=color, s=15, alpha=0.7)
                legend_handles.setdefault(label, sc)

        for ax in [ax_tau, ax_vel]:
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.6, zorder=0)  # 1:1 line
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='dotted', alpha=0.5)
            ax.tick_params(labelsize=9)
            ax.set_title(f'$m = {m}$ vs $m = 3$', fontsize=16, fontweight='bold')

        ax_tau.set_xlabel(r'$\tau_b$ (MPa), $m=3$', fontsize=14)
        ax_tau.set_ylabel(fr'$\tau_b$ (MPa), $m={m}$', fontsize=14)
        ax_vel.set_xlabel(r'$u_b$ (m yr$^{-1}$), $m=3$', fontsize=14)
        ax_vel.set_ylabel(fr'$u_b$ (m yr$^{{-1}}$), $m={m}$', fontsize=14)

    fig.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()),
               loc='lower center', ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, -0.1), frameon=True)

    fig.savefig(fig_dir / "Fig_S6_scatter_taub_ub.pdf", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("scatter_taub_ub saved")


# ============================================================================
# SENSITIVITY TO THE AVERAGING SCALE
# ============================================================================

SCALES_COMMON = [40, 200, 400, 600, 800, 1000]


def process_glacier_stake_multi_scale(glacier_name, stake_name, config, m, C,
                                      scales=SCALES_COMMON, Arg_simu=None, force=False):
    """
    Run process_glacier_stake for several averaging radii.

    Results are saved in processed_timeseries/mw{1/m}/scale_sensitivity/ and
    read from file if they already exist, unless force is True.

    Returns
    -------
    results : dict
        {radius: DataFrame}
    native_scale : float
        Averaging radius used in the study for this stake.
    """
    native_scale = config['avg_dist'][stake_name]
    all_scales = list(scales) + ([native_scale] if native_scale not in scales else [])
    results = {}

    output_dir = proc_data_dir / f'mw{1/m:.3f}' / 'scale_sensitivity'
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale in all_scales:
        csv_path = output_dir / f'{glacier_name}_{stake_name}_scale{scale}.csv'

        if csv_path.exists() and not force:
            results[scale] = pd.read_csv(csv_path)
            continue

        config_tmp = config.copy()
        config_tmp['avg_dist'] = config['avg_dist'].copy()
        config_tmp['avg_dist'][stake_name] = scale

        df = process_glacier_stake(glacier_name, stake_name, config_tmp, m, C,
                                   Arg_simu=Arg_simu, output_file=csv_path)
        if df is not None:
            results[scale] = df
        else:
            print(f"[SKIP] {glacier_name} {stake_name}, radius {scale} m: no result")

    return results, native_scale


def _scale_colors(native_scale):
    """Radii and colours: viridis for the common radii, red for the study radius if different."""
    cmap = plt.cm.viridis
    colors_common = [cmap(i / (len(SCALES_COMMON) - 1)) for i in range(len(SCALES_COMMON))]
    extra = native_scale not in SCALES_COMMON
    scale_list = SCALES_COMMON + ([native_scale] if extra else [])
    color_list = colors_common + (['red'] if extra else [])
    return scale_list, color_list


def _set_scale_log_axes(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=np.array([1., 2., 4., 6.]) * 0.1, numticks=10))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=np.array([1., 2., 3., 4., 5., 6., 8., 10., 12., 15., 20.]) * 0.1, numticks=10))
    ax.grid(True, linestyle='dotted', alpha=0.5)
    ax.tick_params(labelsize=14)


def _scale_legend_handles():
    cmap = plt.cm.viridis
    return [plt.scatter([], [], c=[cmap(i / (len(SCALES_COMMON) - 1))], s=60, label=f'{s} m')
            for i, s in enumerate(SCALES_COMMON)]


def plot_scale_friction_laws(m_index=1):
    """Reconstructed friction law at each stake for several averaging radii."""
    all_stakes = _all_stakes()
    ncols = 4
    nrows = int(np.ceil(len(all_stakes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 16),
                             gridspec_kw=dict(hspace=0.45, wspace=0.25))
    axes_flat = axes.ravel()

    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax = axes_flat[idx]
        config = GLACIERS[glacier_key]
        m, C = config['mval_Cval'][m_index]
        native_scale = config['avg_dist'][stake]

        results, _ = process_glacier_stake_multi_scale(glacier_key, stake, config, m, C)
        scale_list, color_list = _scale_colors(native_scale)

        for scale, color in zip(scale_list, color_list):
            df = results.get(scale)
            if df is None:
                continue
            valid = np.isfinite(df['obs_u_bed']) & np.isfinite(df['obs_tau_b'])
            if valid.sum() == 0:
                continue
            label = f"best radius = {native_scale} m" if scale == native_scale else None
            ax.scatter(df['obs_u_bed'][valid], df['obs_tau_b'][valid],
                       c=[color], s=40, alpha=0.7, label=label)

        _set_scale_log_axes(ax)
        ax.set_xlim(0.5, 200)
        ax.set_ylim(0.02, 0.20)
        ax.legend(fontsize=18)
        ax.set_title(f"{config['full_name']} {stake}", fontsize=18, fontweight='bold', pad=4)

    for ax in axes_flat[len(all_stakes):]:
        ax.set_visible(False)

    fig.legend(handles=_scale_legend_handles(), loc='lower center',
               ncol=len(SCALES_COMMON) // 2, fontsize=20,
               bbox_to_anchor=(0.5, -0.01), frameon=True,
               title='Averaging scale (m)', title_fontsize=20)

    fig.supxlabel(VEL_LABEL, fontsize=28, y=0.10, fontweight="bold")
    fig.supylabel(TAU_LABEL, fontsize=28, x=0.06, fontweight="bold")
    fig.subplots_adjust(bottom=0.16)

    fig.savefig(fig_dir / f"Fig_S4_scale_friction_laws_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("scale_friction_laws saved")


def plot_scale_influence(m_index=1):
    """
    (a) Elmer/Ice basal shear stress versus sliding velocity for several averaging radii;
    (b) stress components at the study radius.
    """
    ncols = 2
    nrows_grid = int(np.ceil(len(plot_specs) / ncols))

    fig = plt.figure(figsize=(22, 24))
    sf_left, sf_right = fig.subfigures(1, 2, wspace=0.1, width_ratios=[1, 1])
    axes_left = sf_left.subplots(nrows_grid, ncols, gridspec_kw=dict(hspace=0.45, wspace=0.25))
    axes_right = sf_right.subplots(nrows_grid, ncols, gridspec_kw=dict(hspace=0.45, wspace=0.25))

    for glacier_key, stake, r, c in plot_specs:
        ax_l = axes_left[r, c]
        ax_r = axes_right[r, c]
        config = GLACIERS[glacier_key]
        m, C = config['mval_Cval'][m_index]
        native_scale = config['avg_dist'][stake]
        title = f"{config['full_name']} {stake}"

        results, _ = process_glacier_stake_multi_scale(glacier_key, stake, config, m, C)
        scale_list, color_list = _scale_colors(native_scale)

        # (a) Elmer/Ice outputs for each radius
        for scale, color in zip(scale_list, color_list):
            df = results.get(scale)
            if df is None:
                continue
            valid = np.isfinite(df['u_bed_elmer']) & np.isfinite(df['tau_b_elmer'])
            label = f"best radius = {native_scale} m" if scale == native_scale else None
            ax_l.scatter(df['u_bed_elmer'][valid], df['tau_b_elmer'][valid],
                         c=[color], s=40, alpha=0.7, label=label)

        _set_scale_log_axes(ax_l)
        ax_l.set_xlim(1, 180)
        ax_l.set_ylim(0.02, 0.20)
        ax_l.legend(fontsize=18)
        ax_l.set_title(title, fontsize=18, fontweight='bold', pad=4)

        # (b) Stress components at the study radius
        df_native = results.get(native_scale)
        if df_native is None:
            continue
        df_native = df_native.sort_values('date')
        stake_color = config['colors'][stake]

        stress_vars = {
            'tau_b_elmer':      (r'$\tau_b$',        stake_color, 'o', ':'),
            'tau_d_elmer':      (r'$\tau_d$',        stake_color, '<', '-'),
            'sigma_elmer':      (r'$\sigma$',        stake_color, 'd', '--'),
            'sigma_plus_tau_b': (r'$\sigma+\tau_b$', 'red',       's', '-.'),
        }

        for col, (label, line_color, marker, ls) in stress_vars.items():
            valid = np.isfinite(df_native[col])
            if valid.sum() == 0:
                continue
            ax_r.plot(df_native['date'][valid], df_native[col][valid],
                      color=line_color, linestyle=ls, linewidth=1.2,
                      marker=marker, markersize=4, label=label)

        ax_r.grid(True, linestyle='dotted', alpha=0.5)
        ax_r.tick_params(labelsize=14)
        ax_r.set_title(title, fontsize=18, fontweight='bold', pad=4)

    sf_left.legend(handles=_scale_legend_handles(), loc='lower center',
                   ncol=len(SCALES_COMMON) // 2, fontsize=20,
                   bbox_to_anchor=(0.5, -0.01), frameon=True,
                   title='Averaging scale (m)', title_fontsize=20)

    legend_handles_right = [
        plt.plot([], [], color='black', linestyle=':',  marker='o', markersize=10, label=r'$\tau_b$')[0],
        plt.plot([], [], color='black', linestyle='-',  marker='<', markersize=10, label=r'$\tau_d$')[0],
        plt.plot([], [], color='black', linestyle='--', marker='d', markersize=10, label=r'$\sigma$')[0],
        plt.plot([], [], color='red',   linestyle='-.', marker='s', markersize=10, label=r'$\sigma+\tau_b$')[0],
    ]
    sf_right.legend(handles=legend_handles_right, loc='lower center', ncol=4,
                    fontsize=20, bbox_to_anchor=(0.5, -0.01), frameon=True)

    sf_left.supxlabel(VEL_LABEL, fontsize=28, y=0.06, fontweight="bold")
    sf_left.supylabel(TAU_LABEL, fontsize=28, fontweight="bold")
    sf_left.suptitle('(a)', fontsize=28, fontweight='bold', x=0.02, y=0.995, ha='left')
    sf_left.subplots_adjust(bottom=0.06, top=0.97, hspace=0.01, wspace=0.01)

    sf_right.supxlabel('Time', fontsize=28, y=0.06, fontweight="bold")
    sf_right.supylabel('Stress (MPa)', fontsize=28, fontweight="bold")
    sf_right.suptitle('(b)', fontsize=28, fontweight='bold', x=0.02, y=0.995, ha='left')
    sf_right.subplots_adjust(bottom=0.06, top=0.97, hspace=0.01, wspace=0.01)

    fig.subplots_adjust(bottom=0.10)
    fig.savefig(fig_dir / f"Fig_S1_scale_influence_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("scale_influence saved")


# ============================================================================
# FRICTION LAWS PER STAKE
# ============================================================================

def plot_friction_laws_per_glacier_panels(m=3):
    """One panel per stake: friction law (first figure) and normalised law (second figure)."""
    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 100, 200, 300, 400, 500]
    V_values = np.arange(0.05, 50, 0.1)

    all_stakes = _all_stakes()
    n = len(all_stakes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    # Normalised laws: hard-bedded sites only
    n_b = len([gk for gk, s in all_stakes if gk not in ["Geb", "StSo"]])
    nrows_b = int(np.ceil(n_b / ncols))

    fig_a, axes_a = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                                 gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_a_flat = axes_a.ravel()

    fig_b, axes_b = plt.subplots(nrows_b, ncols, figsize=(ncols * 5, nrows_b * 4),
                                 gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_b_flat = axes_b.ravel()

    idx_b = 0
    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax_a = axes_a_flat[idx]
        glacier_data = GLACIERS[glacier_key]
        color = glacier_data['colors'][stake]
        marker = glacier_data['markers'][stake]
        title = f"{glacier_data['full_name']} {stake}"

        # Data
        try:
            date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
            if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                ax_a.set_visible(False)
                continue

            scatter_kw = dict(color=color, marker=marker, s=30, zorder=10)
            if marker != '2':
                scatter_kw['edgecolors'] = 'k'
                scatter_kw['linewidths'] = 0.4
            ax_a.scatter(vel, tau, **scatter_kw)

        except Exception as e:
            print(f"Skip {glacier_key} {stake}: {e}")
            ax_a.set_visible(False)
            continue

        # Fitted law
        fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"
        if fit_file.exists():
            df_fit = pd.read_csv(fit_file)
            ax_a.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, linewidth=2)

        _scalar_log_axes(ax_a)
        ax_a.set_xlim(0.5, 200)
        ax_a.set_ylim(0.012, 0.25)
        ax_a.set_xticks([x for x in x_ticks if ax_a.get_xlim()[0] <= x <= ax_a.get_xlim()[1]])
        ax_a.set_yticks([y for y in Y_TICKS if ax_a.get_ylim()[0] <= y <= ax_a.get_ylim()[1]])
        ax_a.grid(which='both', linestyle='dotted', alpha=0.6)
        ax_a.tick_params(labelsize=8)
        ax_a.text(0.97, 0.05, title, transform=ax_a.transAxes, fontsize=10,
                  fontweight='bold', ha='right', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

        # Normalised law
        if glacier_key in ["Geb", "StSo"]:
            continue

        ax_b = axes_b_flat[idx_b]
        idx_b += 1

        CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake, m)

        try:
            vel_norm, tau_norm = calcul_normalised_friction_law(vel, tau, CN_value, As_value, m_value)
            if len(vel_norm) == 0:
                ax_b.set_visible(False)
                continue
            ax_b.scatter(vel_norm, tau_norm, color=color, edgecolors='k',
                         marker=marker, s=30, linewidths=0.4)
        except Exception as e:
            print(f"Skip {glacier_key} {stake} (normalised): {e}")
            ax_b.set_visible(False)
            continue

        # Lliboutry-type (solid) and Weertman-type (dashed) laws
        ax_b.plot(V_values, [scaled_friction_law(u, 1) for u in V_values], 'k-', linewidth=1.5)
        ax_b.plot(np.arange(0.05, 1.5, 0.1), np.arange(0.05, 1.5, 0.1), 'b--', linewidth=1.5)

        _scalar_log_axes(ax_b)
        ax_b.grid(which='both', linestyle='dotted', alpha=0.6)
        ax_b.tick_params(labelsize=8)
        ax_b.text(0.97, 0.05, title, transform=ax_b.transAxes, fontsize=10,
                  fontweight='bold', ha='right', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

    for ax in axes_a_flat[n:]:
        ax.set_visible(False)
    for ax in axes_b_flat[idx_b:]:
        ax.set_visible(False)

    fig_a.supxlabel(VEL_LABEL, fontsize=20, y=0.01, fontweight='bold')
    fig_a.supylabel(TAU_LABEL, fontsize=20, fontweight='bold')
    fig_a.savefig(fig_dir / f"Fig_S3_friction_laws_panels_raw_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig_a)
    print(f"friction_laws_panels_raw_m{m} saved")

    fig_b.supxlabel(VEL_NORM_LABEL, fontsize=20, y=0.01, fontweight='bold')
    fig_b.supylabel(TAU_NORM_LABEL, fontsize=20, fontweight='bold')
    fig_b.savefig(fig_dir / f"Fig_S5_friction_laws_panels_norm_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig_b)
    print(f"friction_laws_panels_norm_m{m} saved")


if __name__ == "__main__":
    plot_friction_laws_comparison()
    plot_scatter_taub_mw()
    plot_scale_friction_laws()
    plot_scale_influence()
    plot_friction_laws_per_glacier_panels()