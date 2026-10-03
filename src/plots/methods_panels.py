"""
Figures of the empirical relationships used to reconstruct basal shear stress
and deformation velocity timeseries.
"""
import sys
from pathlib import Path

# Make src/ importable
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, proc_data_dir, plot_specs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _stake_label(ax, title, x, y, ha, va, fontsize):
    """Stake name inside a panel."""
    ax.text(x, y, title, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))


def _two_subfigures():
    """Figure split into two subfigures of 8 x 2 panels."""
    n_rows = 8
    fig = plt.figure(figsize=(22, 24))
    sf_left, sf_right = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 1])

    # Space at the bottom for the common x label
    sf_left.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)
    sf_right.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)

    axes_left = sf_left.subplots(n_rows, 2)
    axes_right = sf_right.subplots(n_rows, 2)
    return fig, sf_left, sf_right, axes_left, axes_right


def _hide_empty_axes(*axes_arrays):
    for axes in axes_arrays:
        for ax in axes.ravel():
            if not ax.has_data():
                ax.set_visible(False)


def plot_reglin_taub_thk(m=3):
    """
    (a) Elmer/Ice basal shear stress versus thickness at DEM dates, with the linear fit;
    (b) Elmer/Ice and reconstructed basal shear stress timeseries.
    """
    def plot_panel_left(ax, df, color, title, glacier):
        if df is None or len(df) == 0:
            return
        # Glacier Blanc: thickness times surface slope angle, as in process_timeseries.py
        x = df["thickness"] * np.arctan(df["slope"]) if glacier == "GB" else df["thickness"]
        y = df["tau_b_elmer"]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return
        ax.scatter(x, y, color=color, marker='o', s=30)
        if len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.poly1d(p)(xx), '--', linewidth=1.2, color=color)
        _stake_label(ax, title, 0.03, 0.97, 'left', 'top', 20)
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.grid(True, linestyle='dotted')

    def plot_panel_right(ax, df, color, title, glacier):
        if df is None or len(df) == 0:
            return
        mask_elmer = np.isfinite(df["date"]) & np.isfinite(df["tau_b_elmer"])
        ax.scatter(df["date"][mask_elmer], df["tau_b_elmer"][mask_elmer],
                   color=color, marker='o', s=30)
        mask_obs = np.isfinite(df["date"]) & np.isfinite(df["obs_tau_b"])
        df_obs = df[mask_obs].sort_values("date")
        # Saint-Sorlin: show the thickness-based reconstruction
        col = "obs_tau_b_reglin" if glacier == "StSo" else "obs_tau_b"
        ax.plot(df_obs["date"], df_obs[col], '--', linewidth=1.2, color=color)
        _stake_label(ax, title, 0.97, 0.03, 'right', 'bottom', 20)
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.set_xlim(1900, 2025)
        ax.set_ylim(0, 0.14)
        ax.grid(True, linestyle='dotted')

    fig, sf_left, sf_right, axes_left, axes_right = _two_subfigures()

    for glacier, stake, r, c in plot_specs:
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing file: {glacier} {stake}")
            continue
        df = pd.read_csv(file)
        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"

        plot_panel_left(axes_left[r, c], df, color, title, glacier)
        if glacier == "GB":
            axes_left[r, c].set_xlabel(r"Thickness $\times$ slope angle (m)", fontsize=12)

        plot_panel_right(axes_right[r, c], df, color, title, glacier)

    _hide_empty_axes(axes_left, axes_right)

    sf_left.supxlabel('Thickness (m)', fontsize=28, y=0.01, fontweight='bold')
    sf_left.supylabel('Basal shear stress (MPa)', fontsize=28, fontweight='bold')
    sf_left.suptitle('(a)', fontsize=25, fontweight='bold', x=0.02, y=0.995, ha='left', va='top')

    sf_right.supxlabel('Time', fontsize=28, y=0.01, fontweight='bold')
    sf_right.supylabel('Basal shear stress (MPa)', fontsize=28, fontweight='bold')
    sf_right.suptitle('(b)', fontsize=25, fontweight='bold', x=0.02, y=0.995, ha='left', va='top')

    fig.savefig(fig_dir / f"Fig_6_reglin_taub_thk_m{m}.pdf", bbox_inches='tight')
    plt.close(fig)
    print("reglin_taub_thk saved")


def plot_reglin_udef_thk4(m=3):
    """
    (a) Elmer/Ice deformation velocity versus thickness^4 at DEM dates, with the linear fit;
    (b) Elmer/Ice and reconstructed deformation velocity timeseries.
    """
    def plot_panel_left(ax, df, color, title, glacier):
        if df is None or len(df) == 0:
            return
        # Glacier Blanc: thickness^4 times slope angle^3, as in process_timeseries.py
        if glacier == "GB":
            x = df["thickness"]**4 * np.arctan(df["slope"])**3
        else:
            x = df["thickness"]**4
        y = df["u_def_elmer"]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return
        ax.scatter(x, y, color=color, marker='o', s=30)
        if len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.poly1d(p)(xx), '--', linewidth=1.2, color=color)
        _stake_label(ax, title, 0.03, 0.97, 'left', 'top', 16)
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.grid(True, linestyle='dotted')

    def plot_panel_right(ax, df, color, title, glacier):
        if df is None or len(df) == 0:
            return
        mask_elmer = np.isfinite(df["date"]) & np.isfinite(df["u_def_elmer"])
        ax.scatter(df["date"][mask_elmer], df["u_def_elmer"][mask_elmer],
                   color=color, marker='o', s=30)
        mask_obs = np.isfinite(df["date"]) & np.isfinite(df["obs_u_def"])
        df_obs = df[mask_obs].sort_values("date")
        # Saint-Sorlin: show the thickness-based reconstruction
        col = "obs_u_def_reglin" if glacier == "StSo" else "obs_u_def"
        ax.plot(df_obs["date"], df_obs[col], '--', linewidth=1.2, color=color)
        _stake_label(ax, title, 0.97, 0.97, 'right', 'top', 16)
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.set_xlim(1900, 2025)
        ax.set_ylim(-5, 70)
        ax.grid(True, linestyle='dotted')

    fig, sf_left, sf_right, axes_left, axes_right = _two_subfigures()

    for glacier, stake, r, c in plot_specs:
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing file: {glacier} {stake}")
            continue
        df = pd.read_csv(file)
        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"

        plot_panel_left(axes_left[r, c], df, color, title, glacier)
        if glacier == "GB":
            axes_left[r, c].set_xlabel(r"Thickness$^4$ $\times$ slope angle$^3$", fontsize=12)

        plot_panel_right(axes_right[r, c], df, color, title, glacier)

    _hide_empty_axes(axes_left, axes_right)

    sf_left.supxlabel('Thickness$^4$ (m$^4$)', fontsize=28, y=0.01, fontweight='bold')
    sf_left.supylabel(r'Deformation velocity (m yr$^{-1}$)', fontsize=28, fontweight='bold')
    sf_left.suptitle('(a)', fontsize=20, fontweight='bold', x=0.02, y=0.995, ha='left', va='top')

    sf_right.supxlabel('Time', fontsize=28, y=0.01, fontweight='bold')
    sf_right.supylabel(r'Deformation velocity (m yr$^{-1}$)', fontsize=28, fontweight='bold')
    sf_right.suptitle('(b)', fontsize=20, fontweight='bold', x=0.02, y=0.995, ha='left', va='top')

    fig.savefig(fig_dir / f"Fig_7_reglin_udef_thk4_m{m}.pdf", bbox_inches='tight')
    plt.close(fig)
    print("reglin_udef_thk4 saved")


def _all_stakes():
    """All studied stakes, as (glacier, stake) pairs."""
    return [(gk, s)
            for gk, gd in GLACIERS.items()
            for s in gd['xy_coords'].keys()
            if s != "Wheel"]


def plot_all_stakes_reglin_thk_slope_taub(m=3):
    """Elmer/Ice basal shear stress versus thickness times surface slope, at all stakes."""
    def plot_panel(ax, df, color, title):
        x = df["thickness"] * df["slope"]
        y = df["tau_b_elmer"]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return
        ax.scatter(x, y, color=color, marker='o', s=40, edgecolors='k', linewidths=0.4)
        if len(x) > 1:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.poly1d(p)(xx), '--', linewidth=1.5, color=color)
        _stake_label(ax, title, 0.97, 0.05, 'right', 'bottom', 16)
        ax.grid(True, linestyle='dotted', alpha=0.6)
        ax.tick_params(labelsize=8)

    all_stakes = _all_stakes()
    n = len(all_stakes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5),
                             gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_flat = axes.ravel()

    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax = axes_flat[idx]
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing file: {glacier_key} {stake}")
            ax.set_visible(False)
            continue

        df = pd.read_csv(file)
        color = GLACIERS[glacier_key]["colors"][stake]
        title = f"{GLACIERS[glacier_key]['full_name']} {stake}"
        plot_panel(ax, df, color, title)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.supxlabel(r"Thickness $\times$ slope", fontsize=26, y=0.01, fontweight="bold")
    fig.supylabel("Basal shear stress, Elmer/Ice (MPa)", fontsize=26, fontweight="bold")

    fig.savefig(fig_dir / f"Fig_S2_reglin_all_taub_elmer_thk_slope_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("reglin_all_taub_elmer_thk_slope saved")


def plot_thick_elmer_vs_obs(m=3):
    """Elmer/Ice ice thickness versus observed thickness, at all stakes."""
    def plot_panel(ax, df, color, title):
        mask = np.isfinite(df['thick_elmer']) & np.isfinite(df['thickness'])
        x = df['thickness'][mask]
        y = df['thick_elmer'][mask]
        if len(x) == 0:
            return
        ax.scatter(x, y, color=color, marker='o', s=40, edgecolors='k', linewidths=0.4)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5)  # 1:1 line
        _stake_label(ax, title, 0.03, 0.97, 'left', 'top', 16)
        ax.grid(True, linestyle='dotted', alpha=0.6)
        ax.tick_params(labelsize=8)

    all_stakes = _all_stakes()
    ncols = 4
    nrows = int(np.ceil(len(all_stakes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_flat = axes.ravel()

    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax = axes_flat[idx]
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv"
        if not file.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(file)
        color = GLACIERS[glacier_key]["colors"][stake]
        title = f"{GLACIERS[glacier_key]['full_name']} {stake}"
        plot_panel(ax, df, color, title)

    for ax in axes_flat[len(all_stakes):]:
        ax.set_visible(False)

    fig.supxlabel("Observed thickness (m)", fontsize=26, y=0.01, fontweight="bold")
    fig.supylabel("Elmer/Ice thickness (m)", fontsize=26, fontweight="bold")

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(fig_dir / f"thick_elmer_vs_obs_m{m}.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("thick_elmer_vs_obs saved")


if __name__ == "__main__":
    plot_reglin_taub_thk(3)
    plot_reglin_udef_thk4(3)
    plot_all_stakes_reglin_thk_slope_taub()
    plot_thick_elmer_vs_obs()