import sys
from pathlib import Path

# Ajouter src au path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, get_friclaw_params, plot_specs
from friction_laws import *
from run_friction_fits import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_reglin_taub_thk(m=3):

    def plot_panel_left(ax, df, color, title, transform="thk"):
        if df is None or len(df) == 0:
            return
        x = df["thickness"] * df["slope"] if transform == "thk_slope" else df["thickness"]
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
        ax.set_title('')
        ax.text(0.03, 0.97, title,
            transform=ax.transAxes,
            fontsize=20, fontweight='bold',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.grid(True, linestyle='dotted')

    def plot_panel_right(ax, df, color, title):
        if df is None or len(df) == 0:
            return
        mask_elmer = np.isfinite(df["date"]) & np.isfinite(df["tau_b_elmer"])
        ax.scatter(df["date"][mask_elmer], df["tau_b_elmer"][mask_elmer],
                   color=color, marker='o', s=30, label='Elmer')
        mask_obs = np.isfinite(df["date"]) & np.isfinite(df["obs_tau_b"])
        df_obs = df[mask_obs].sort_values("date")
        if glacier=="StSo":
            ax.plot(df_obs["date"], df_obs["obs_tau_b_reglin"],
                '--', linewidth=1.2, color=color)
        else:
            ax.plot(df_obs["date"], df_obs["obs_tau_b"],
                '--', linewidth=1.2, color=color)
        ax.set_title('')
        ax.text(0.97, 0.03, title,
            transform=ax.transAxes,
            fontsize=20, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.set_xlim(1900, 2025)
        ax.set_ylim(0, 0.14)
        ax.grid(True, linestyle='dotted')

    n_rows = 8

    fig = plt.figure(figsize=(22, 24))

    # deux subfigures côte à côte avec marge entre elles
    sf_left, sf_right = fig.subfigures(
        1, 2,
        wspace=0.01,
        width_ratios=[1, 1]
    )

    # espace en bas de chaque subfigure pour le supxlabel
    sf_left.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)
    sf_right.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)

    axes_left  = sf_left.subplots(n_rows, 2)
    axes_right = sf_right.subplots(n_rows, 2)

    for glacier, stake, r, c in plot_specs:
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing file: {glacier}-{stake}")
            continue
        df    = pd.read_csv(file)
        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"
        trans = "thk_slope" if glacier == "GB" else "thk"

        plot_panel_left(axes_left[r, c], df, color, title, trans)
        if glacier == "GB":
            axes_left[r, c].set_xlabel("Thickness × Slope (m)", fontsize=12)

        plot_panel_right(axes_right[r, c], df, color, title)

    # labels des panels — y=0.0 colle au bas de la subfigure
    sf_left.supxlabel('Thickness (m)', fontsize=28, y=0.01, fontweight='bold')
    sf_left.supylabel('Basal shear stress (MPa)', fontsize=28, fontweight='bold')
    sf_left.suptitle('(a)', fontsize=25, fontweight='bold',
                     x=0.02, y=0.995, ha='left', va='top')

    sf_right.supxlabel('Time', fontsize=28, y=0.01, fontweight='bold')
    sf_right.supylabel('Basal shear stress (MPa)', fontsize=28, fontweight='bold')
    sf_right.suptitle('(b)', fontsize=25, fontweight='bold',
                      x=0.02, y=0.995, ha='left', va='top')

    fig.savefig(fig_dir / f"reglin_taub_thk_m{m}.pdf", bbox_inches='tight')
    print("reglin_taub_thk saved")





def plot_reglin_udef_thk4(m=3):

    def plot_panel_left(ax, df, color, title, transform="thk"):
        if df is None or len(df) == 0:
            return
        x = df["thickness"]**4 * df["slope"]**3 if transform == "thk_slope" else df["thickness"]**4
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
        ax.set_title('')
        ax.text(0.03, 0.97, title,
            transform=ax.transAxes,
            fontsize=16, fontweight='bold',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.grid(True, linestyle='dotted')

    def plot_panel_right(ax, df, color, title):
        if df is None or len(df) == 0:
            return
        mask_elmer = np.isfinite(df["date"]) & np.isfinite(df["u_def_elmer"])
        ax.scatter(df["date"][mask_elmer], df["u_def_elmer"][mask_elmer],
                   color=color, marker='o', s=30, label='Elmer')
        mask_obs = np.isfinite(df["date"]) & np.isfinite(df["obs_u_def"])
        df_obs = df[mask_obs].sort_values("date")
        if glacier=="StSo":
            ax.plot(df_obs["date"], df_obs["obs_u_def_reglin"],
                '--', linewidth=1.2, color=color)
        else:
            ax.plot(df_obs["date"], df_obs["obs_u_def"],
                '--', linewidth=1.2, color=color)
        ax.set_title('')
        ax.text(0.97, 0.97, title,
            transform=ax.transAxes,
            fontsize=16, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.tick_params(axis='both', labelsize=14, width=0.9)
        ax.set_xlim(1900, 2025)
        ax.set_ylim(-5, 70)
        ax.grid(True, linestyle='dotted')

    n_rows = 8

    fig = plt.figure(figsize=(22, 24))

    # deux subfigures côte à côte avec marge entre elles
    sf_left, sf_right = fig.subfigures(
        1, 2,
        wspace=0.01,
        width_ratios=[1, 1]
    )

    # espace en bas de chaque subfigure pour le supxlabel
    sf_left.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)
    sf_right.subplots_adjust(bottom=0.06, top=0.97, hspace=0.45, wspace=0.25)

    axes_left  = sf_left.subplots(n_rows, 2)
    axes_right = sf_right.subplots(n_rows, 2)

    for glacier, stake, r, c in plot_specs:
        file = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing file: {glacier}-{stake}")
            continue
        df    = pd.read_csv(file)
        color = GLACIERS[glacier]["colors"][stake]
        title = f"{GLACIERS[glacier]['full_name']} {stake}"
        trans = "thk_slope" if glacier == "GB" else "thk"

        plot_panel_left(axes_left[r, c], df, color, title, trans)
        if glacier == "GB":
            axes_left[r, c].set_xlabel(fr"Thickness$^4$ * Slope$^3$", fontsize=12)

        plot_panel_right(axes_right[r, c], df, color, title)

    # cacher axes vides
    for ax in axes_left.ravel():
        if not ax.has_data():
            ax.set_visible(False)
    for ax in axes_right.ravel():
        if not ax.has_data():
            ax.set_visible(False)

    # labels des panels — y=0.0 colle au bas de la subfigure
    sf_left.supxlabel('Thickness$^4$', fontsize=28, y=0.01, fontweight='bold')
    sf_left.supylabel('Deformation velocity $(m \cdot yr^{-1})$', fontsize=28, fontweight='bold')
    sf_left.suptitle('(a)', fontsize=20, fontweight='bold',
                     x=0.02, y=0.995, ha='left', va='top')

    sf_right.supxlabel('Time', fontsize=28, y=0.01, fontweight='bold')
    sf_right.supylabel('Deformation velocity $(m \cdot yr^{-1})$', fontsize=28, fontweight='bold')
    sf_right.suptitle('(b)', fontsize=20, fontweight='bold',
                      x=0.02, y=0.995, ha='left', va='top')

    fig.savefig(fig_dir / f"reglin_udef_thk4_m{m}.pdf", bbox_inches='tight')
    print("reglin_udef_thk4 saved")



def plot_all_stakes_reglin_thk_slope_taub(m=3):

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
            yy = np.poly1d(p)(xx)
            ax.plot(xx, yy, '--', linewidth=1.5, color=color)
        ax.text(0.97, 0.05, title,
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.grid(True, linestyle='dotted', alpha=0.6)
        ax.tick_params(labelsize=8)


    # --- collecter tous les stakes ---
    all_stakes = []
    for glacier_key, glacier_data in GLACIERS.items():
        for stake in glacier_data['xy_coords'].keys():
            if stake == "Wheel":
                continue
            all_stakes.append((glacier_key, stake))

    n      = len(all_stakes)
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5),
                             gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_flat = axes.ravel()

    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax    = axes_flat[idx]
        file  = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv"
        if not file.exists():
            print(f"[WARNING] missing: {glacier_key}-{stake}")
            ax.set_visible(False)
            continue

        df    = pd.read_csv(file)
        color = GLACIERS[glacier_key]["colors"][stake]
        title = f"{GLACIERS[glacier_key]['full_name']} {stake}"

        plot_panel(ax, df, color, title)

    # cacher axes vides
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.supxlabel(r"Thickness $\times$ Slope", fontsize=26, y=0.01, fontweight = "bold")
    fig.supylabel("Basal shear stress — Elmer (MPa)", fontsize=26, fontweight = "bold")

    fig.savefig(fig_dir / f"reglin_all_taub_elmer_thk_slope_m{m}.pdf",
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("plot_all_stakes_reglin_thk_slope_taub saved")


def plot_thick_elmer_vs_obs(m=3):

    def plot_panel(ax, df, color, title):
        mask = np.isfinite(df['thick_elmer']) & np.isfinite(df['thickness'])
        x = df['thickness'][mask]
        y = df['thick_elmer'][mask]
        if len(x) == 0:
            return
        ax.scatter(x, y, color=color, marker='o', s=40,
                   edgecolors='k', linewidths=0.4)
        # ligne 1:1
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='1:1')
        ax.text(0.03, 0.97, title,
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
        ax.grid(True, linestyle='dotted', alpha=0.6)
        ax.tick_params(labelsize=8)

    all_stakes = [(gk, s)
                  for gk, gd in GLACIERS.items()
                  for s in gd['xy_coords'].keys()
                  if s != "Wheel"]

    ncols = 4
    nrows = int(np.ceil(len(all_stakes) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             gridspec_kw=dict(hspace=0.45, wspace=0.35))
    axes_flat = axes.ravel()

    for idx, (glacier_key, stake) in enumerate(all_stakes):
        ax    = axes_flat[idx]
        file  = proc_data_dir / f"mw{1/m:.3f}" / f"{glacier_key}_all_data_{stake}.csv"
        if not file.exists():
            ax.set_visible(False)
            continue
        df    = pd.read_csv(file)
        color = GLACIERS[glacier_key]["colors"][stake]
        title = f"{GLACIERS[glacier_key]['full_name']} {stake}"
        plot_panel(ax, df, color, title)

    for ax in axes_flat[len(all_stakes):]:
        ax.set_visible(False)

    fig.supxlabel("Observed thickness (m)", fontsize=26, y=0.01, fontweight="bold")
    fig.supylabel("Elmer thickness (m)",     fontsize=26, fontweight="bold")

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(fig_dir / f"thick_elmer_vs_obs_m{m}.pdf",
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("plot_thick_elmer_vs_obs saved")


if __name__ == "__main__":
   
    plot_reglin_taub_thk(3)
    plot_reglin_udef_thk4(3)

    plot_all_stakes_reglin_thk_slope_taub()
    plot_thick_elmer_vs_obs()
