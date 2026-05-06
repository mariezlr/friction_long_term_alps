import sys
from pathlib import Path

# Ajouter src au path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import GLACIERS, fig_dir, get_friclaw_params
from friction_laws import *
from run_friction_fits import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_friction_laws_comparison(m_values=[1, 3, 6]):
    x_ticks = [1, 2, 4, 6, 10, 20, 30, 50, 80, 100, 200, 300, 400, 500]
    y_ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
               0.11, 0.12, 0.13, 0.14, 0.16, 0.2, 0.3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # axes[0, :] = raw friction laws (ax1 style)
    # axes[1, :] = normalized (ax2 style)

    legend_handles = {}  # label → handle, pour légende unique

    for col, m in enumerate(m_values):
        ax1 = axes[0, col]
        ax2 = axes[1, col]

        for glacier_key, glacier_data in GLACIERS.items():
            for stake in glacier_data['xy_coords'].keys():
                if stake == "Wheel":
                    continue
                color  = glacier_data['colors'][stake]
                marker = glacier_data['markers'][stake]
                label  = f"{glacier_data['full_name']} {stake}".strip()

                # --- RAW DATA ---
                try:
                    date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, m)
                    if vel is None or tau is None or len(vel) == 0 or len(tau) == 0:
                        continue
                    kwargs = dict(color=color, marker=marker, zorder=10, s=20)
                    if marker != '2':
                        kwargs['edgecolors'] = 'k'
                        kwargs['linewidths'] = 0.4
                    sc = ax1.scatter(vel, tau, **kwargs)
                    if label not in legend_handles:
                        legend_handles[label] = sc

                except Exception as e:
                    print(f"Skip {glacier_key}-{stake} raw (m={m}): {e}")
                    continue

                # --- FIT DATA ---
                if glacier_key not in ["Geb", "StSo"]:
                    fit_file = proc_data_dir / f"mw{1/m:.3f}" / "friction_fits" / f"{glacier_key}_{stake}_friclaw_ts.csv"
                    if not Path(fit_file).exists():
                        print(f"Missing fit file {glacier_key}-{stake}")
                        continue
                    df_fit = pd.read_csv(fit_file)
                    ax1.plot(df_fit['vel_fit'], df_fit['tau_fit'], color=color, linewidth=2)


                # --- PARAMS ---
                CN_value, q_value, As_value, m_value = get_friclaw_params(glacier_key, stake, m)

                # if glacier_key == "Geb":
                #     ax1.axhline(y=CN_value, color=color, linestyle='--', linewidth=0.8)

                # --- NORMALIZED ---
                if (glacier_key == "Geb") or (glacier_key == "StSo"):
                    continue

                try:
                    vel_norm, tau_norm = calcul_normalised_friction_law(
                        vel, tau, CN_value, As_value, m_value)
                    if len(vel_norm) == 0:
                        continue
                    ax2.scatter(vel_norm, tau_norm, color=color, marker=marker,
                                edgecolors='k', linewidths=0.4, s=20)
                except Exception as e:
                    print(f"Skip normalized {glacier_key}-{stake} (m={m}): {e}")

        # --- lois théoriques panel bas ---
        V_values = np.arange(0.05, 50, 0.1)
        ax2.plot(V_values, [scaled_friction_law(u, 1) for u in V_values],
                 color='k', linewidth=1.2, label='Cavitation law')
        ax2.plot(np.arange(0.05, 1.5, 0.1), np.arange(0.05, 1.5, 0.1),
                 'b--', linewidth=1.2, label='Weertman-type law')

        # --- style ax1 ---
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim(0.02, 0.17)
        ax1.set_xticks([x for x in x_ticks if ax1.get_xlim()[0] <= x <= ax1.get_xlim()[1]])
        ax1.set_yticks([y for y in y_ticks if ax1.get_ylim()[0] <= y <= ax1.get_ylim()[1]])
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax1.grid(which='both', linestyle='dotted', alpha=0.5)
        ax1.set_title(f"m = {m}", fontsize=16, fontweight='bold')

        # --- style ax2 ---
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(which='both', linestyle='dotted', alpha=0.5)
        if col == 0:
            ax2.legend(fontsize=8, loc='upper left')  # cavitation / weertman une fois

        # --- labels axes externes seulement ---
        if col == 0:
            ax1.set_ylabel(r'Basal shear stress (MPa)', fontsize=14, fontweight="bold")
            ax2.set_ylabel(r'Scaled shear stress $\left(\frac{\tau_b}{CN}\right)^m$', fontsize=14, fontweight="bold")
        ax1.set_xlabel(r'Basal sliding velocity $(m \cdot yr^{-1})$', fontsize=14, fontweight="bold")
        ax2.set_xlabel(r'Scaled sliding velocity $\frac{u_b}{A_s(CN)^m}$', fontsize=14, fontweight="bold")

    # --- légende globale unique ---
    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='lower center',
        ncol=6,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(fig_dir / "friction_laws_comparison.pdf",
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("friction_laws_comparison saved")


def plot_scatter_taub_mw():
    mw_list=[1, 3, 6]
    color_list=["purple", "crimson", "green"]
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for (mw, color) in zip(mw_list, color_list):
        first = True

        for glacier_key, glacier_data in GLACIERS.items():
            if glacier_key in ["Geb"]:
                continue      
            for stake in glacier_data['xy_coords'].keys():
                date, vel, tau = compile_vel_tau_timeseries(glacier_key, stake, mw)
                if tau is None or tau.empty:
                    continue

                label = f"m = {mw}" if first else None
                ax.scatter(vel, tau,
                    c=color, s=10, alpha=0.6, label=label
                )

                first = False 

        ax.set_xlabel(f"Basal sliding velocity (m/yr)")
        ax.grid(True, linestyle='dotted')
    
    ax.legend()
    ax.set_ylabel(fr"$\tau_b$ (MPa)")
    plt.tight_layout()
    fig.savefig(fig_dir / f"scatter_taub_mw{'_'.join(str(m) for m in mw_list)}.pdf", bbox_inches='tight')
    plt.close(fig)
    print("scatter_taub saved")



if __name__ == "__main__":
    plot_friction_laws_comparison()
    plot_scatter_taub_mw()