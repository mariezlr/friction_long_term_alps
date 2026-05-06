"""
methods_figure.py
=================
Flowchart "methods synthesis" — modifiable facilement.
"""
import sys
from pathlib import Path

# Ajouter src au path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

from utils import fig_dir
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ─────────────────────────────────────────────────────────────
# 1. PALETTE & STYLES
# ─────────────────────────────────────────────────────────────

C = {
    "input"  : "#a8392b",   # rouge  — données d'entrée
    "model"  : "#1a5f8a",   # bleu   — sorties de modèle
    "obs"    : "#5e3a7a",   # violet — séries observées continues
    "interp" : "#c06010",   # orange — interpolées
    "result" : "#0d5c42",   # vert   — résultat final
    "arrow"  : "#5a5248",   # gris   — flèches neutres
    "bg"     : "#fdfbf8",
}

ALPHA_FILL  = 0.12    # transparence du fond des boîtes
LW_BOX      = 1.6     # épaisseur du contour des boîtes
LW_ARROW    = 1.3     # épaisseur des flèches
FONT_TITLE  = 13      # taille du titre dans les boîtes
FONT_SUB    = 10        # taille des sous-titres

plt.rcParams.update({
    "font.family"    : "serif",
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": C["bg"],
    "axes.facecolor"  : C["bg"],
})


# ─────────────────────────────────────────────────────────────
# 2. BOÎTES  (x, y, w, h tous en axes fraction)
# ─────────────────────────────────────────────────────────────
#
# Format d'une boîte :
#   "id": dict(
#       x, y, w, h      — position & taille
#       color           — clé dans C{}
#       title           — texte principal (LaTeX ok)
#       sub             — liste de sous-titres ([] si aucun)
#       style           — "solid" | "dashed"
#   )
#
BOXES = {

    # ── Colonne gauche : dates DEM ──────────────────────────
    "dem": dict(
        x=0.04, y=0.87, w=0.42, h=0.08,
        color="input",
        title=r"Surface & Bed DEMs",
        sub=[r"$z_s(x,y,\,t_\mathrm{DEM})$,  $z_b(x,y)$"],
        style="solid",
    ),
    "stokes": dict(
        x=0.04, y=0.73, w=0.42, h=0.08,
        color="model",
        title=r"Force Balance  ·  Elmer/Ice",
        sub=[r"Full Stokes solved at each $t_\mathrm{DEM}$"],
        style="solid",
    ),
    "tau_disc": dict(
        x=0.04, y=0.57, w=0.18, h=0.10,
        color="model",
        title=r"$\tau_b(t_\mathrm{DEM})$",
        sub=[r"$\approx$", 
             r"$g\left(H(t_\mathrm{DEM})\right)$"],
        style="solid",
    ),
    "udef_disc": dict(
        x=0.28, y=0.57, w=0.18, h=0.10,
        color="model",
        title=r"$u_\mathrm{def}(t_\mathrm{DEM})$",
        sub=[r"$\approx$",  
             r"$h\left(H^{n+1}(t_\mathrm{DEM})\right)$"],
        style="solid",
    ),

    # ── Colonne droite : dates Stake ────────────────────────
    "obs": dict(
        x=0.54, y=0.57, w=0.42, h=0.10,
        color="input",
        title=r"Field Observations",
        sub=[r"In-situ stake readings"],
        style="solid",
    ),
    "H_cont": dict(
        x=0.54, y=0.41, w=0.18, h=0.10,
        color="obs",
        title=r"$H(t_\mathrm{Stake})$",
        sub=[r"Ice thickness"],
        style="solid",
    ),
    "usurf_cont": dict(
        x=0.78, y=0.41, w=0.18, h=0.10,
        color="obs",
        title=r"$u_\mathrm{surf}(t_\mathrm{Stake})$",
        sub=[r"Surface velocity"],
        style="solid",
    ),

    # ── Rangée interpolation ────────────────────────────────
    "tau_cont": dict(
        x=0.04, y=0.21, w=0.30, h=0.10,
        color="interp",
        title=r"$\mathrm{\tau_b(t_\mathrm{Stake})}$",
        sub=[r"$\approx g\left(H(t_\mathrm{Stake})\right)$",
             r"Basal shear stress"],
    style="solid",
    ),
    "udef_cont": dict(
        x=0.39, y=0.21, w=0.22, h=0.10,
        color="interp",
        title=r"$\mathrm{u_\mathrm{def}(t_\mathrm{Stake})}$",
        sub=[r"$\approx h\left(H^{n+1}(t_\mathrm{Stake})\right)$",
             r"Deformation velocity"],
    style="dashed",
    ),
    "ubed_cont": dict(
        x=0.66, y=0.21, w=0.30, h=0.10,
        color="interp",
        title=r"$\mathrm{u_\mathrm{bed}(t_\mathrm{Stake})}$",
        sub=[r"$= u_\mathrm{surf}(t_\mathrm{Stake}) - u_\mathrm{def}(t_\mathrm{Stake})$",
             r"Basal sliding velocity"],
        style="solid",
    ),

    # ── Résultat final ──────────────────────────────────────
    "law": dict(
        x=0.10, y=0.03, w=0.80, h=0.12,
        color="result",
        title=r"Friction Law :  $\tau_b = f(u_\mathrm{bed})$",
        sub=[r"Fit $\tau_b(t_\mathrm{Stake})$ vs $u_\mathrm{bed}(t_\mathrm{Stake})$ — Weertman- & Lliboutry-type laws"],
        style="solid",
    ),
}


# ─────────────────────────────────────────────────────────────
# 3. FLÈCHES
# ─────────────────────────────────────────────────────────────
#
# Format :
#   (x0, y0, x1, y1, color_key, style, label)
#
#   style : "solid" | "dashed"
#   label : "" si pas de label
#
# Les coordonnées sont en axes fraction.
# Astuce : (box["x"] + box["w"]/2) = centre horizontal d'une boîte
#          (box["y"])               = bord bas,  (box["y"]+box["h"]) = bord haut

def cx(k): return BOXES[k]["x"] + BOXES[k]["w"] / 2   # centre x
def cy(k): return BOXES[k]["y"] + BOXES[k]["h"] / 2   # centre y
def top(k): return BOXES[k]["y"] + BOXES[k]["h"]
def bot(k): return BOXES[k]["y"]
def left(k): return BOXES[k]["x"]
def right(k): return BOXES[k]["x"] + BOXES[k]["w"]

ARROWS = [
    # (x0,       y0,          x1,        y1,         color,     style,    label)
    # DEM → Stokes
    (cx("dem"),  bot("dem"),   cx("stokes"), top("stokes"),   "input",  "solid",  ""),
    # Stokes → tau_disc
    (cx("stokes")-0.08, bot("stokes"), cx("tau_disc"),  top("tau_disc"),   "model",  "solid",  ""),
    # Stokes → udef_disc
    (cx("stokes")+0.08, bot("stokes"), cx("udef_disc"), top("udef_disc"),  "model",  "solid",  ""),
    # Obs → H_cont
    (cx("obs")-0.08, bot("obs"), cx("H_cont"),   top("H_cont"),   "input",  "solid",  ""),
    # Obs → usurf_cont
    (cx("obs")+0.08, bot("obs"), cx("usurf_cont"), top("usurf_cont"), "input", "solid", ""),
    # tau_disc → tau_cont
    (cx("tau_disc"),  bot("tau_disc"),  cx("tau_cont"),  top("tau_cont"),  "model",  "solid",  ""),
    # H_cont → tau_cont
    (left("H_cont"),  cy("H_cont"),     cx("tau_cont"), top("tau_cont"),  "obs",   "dashed", ""),
    # udef_disc → udef_cont
    (cx("udef_disc"), bot("udef_disc"), cx("udef_cont"), top("udef_cont"), "model",  "solid",  ""),
    # H_cont → udef_cont
    (cx("H_cont"),    bot("H_cont"),    cx("udef_cont"), top("udef_cont"), "obs",   "dashed", ""),
    # udef_cont → ubed_cont
    (right("udef_cont"), cy("udef_cont"), left("ubed_cont"), cy("ubed_cont"), "interp","solid",""),
    # usurf_cont → ubed_cont
    (cx("usurf_cont"), bot("usurf_cont"), right("ubed_cont")-0.08, top("ubed_cont"), "obs","solid",""),
    # tau_cont → law
    (cx("tau_cont"),  bot("tau_cont"),  cx("law")-0.15, top("law"),        "result", "solid",  ""),
    # ubed_cont → law
    (cx("ubed_cont"), bot("ubed_cont"), cx("law")+0.15, top("law"),        "result", "solid",  ""),
]


# ─────────────────────────────────────────────────────────────
# 4. ANNOTATIONS  (badges de colonnes, label SIA)
# ─────────────────────────────────────────────────────────────

PATCHES=[
    (0.02, 0.55, 0.46, 0.42, "model"),
    (0.52, 0.39, 0.46, 0.30, "obs")]


BADGES = [
    # (x_centre, y_centre, texte, color_key)
    (0.25, 0.99, r"Few dates  $t_\mathrm{DEM}$",   "model"),
    (0.75, 0.71, r"Many dates  $t_\mathrm{Stake}$", "obs"),
]

# Texte libre au centre (SIA)
FREE_TEXTS = [
    # (x, y, texte, color_key, fontsize, bold)
    (0.25, 0.62, "S\nI\nA", "model", 14, True),
    (0.50, 0.62, "+", "arrow", 24, False), 
    (0.25, 0.695, "Empirical relationships", "model", 10, False), 
    (0.50, 0.35, "TEMPORAL INTERPOLATION via empirical SIA relationships", "interp", 14, False), 
]


# ─────────────────────────────────────────────────────────────
# 5. RENDU
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── dessin des boîtes ─────────────────────────────────────────

def draw_box(ax, b):
    col   = C[b["color"]]
    alpha = ALPHA_FILL
    ls    = "--" if b.get("style") == "dashed" else "-"

    # fond + contour
    rect = FancyBboxPatch(
        (b["x"], b["y"]), b["w"], b["h"],
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor=col + f"{int(alpha*255):02x}",
        edgecolor=col, linewidth=LW_BOX,
        linestyle=ls, clip_on=False,
    )
    ax.add_patch(rect)

    # barre gauche décorative
    bar = FancyBboxPatch(
        (b["x"], b["y"]), 0.006, b["h"],
        boxstyle="round,pad=0.001",
        transform=ax.transAxes,
        facecolor=col, edgecolor="none",
        clip_on=False,
    )
    ax.add_patch(bar)

    # positions verticales selon nombre de sous-titres
    n = len(b.get("sub", []))
    if   n == 0: ty = b["y"] + b["h"] * 0.50
    elif n == 1: ty = b["y"] + b["h"] * 0.68
    else         : ty = b["y"] + b["h"] * 0.78

    ax.text(b["x"] + b["w"]/2, ty, b["title"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold", color=col,
            clip_on=False)

    sub_ys = []
    if n == 1:
        sub_ys = [b["y"] + b["h"] * 0.33]
    elif n == 2:
        sub_ys = [b["y"] + b["h"] * 0.47, b["y"] + b["h"] * 0.18]

    for i, (sy, stxt) in enumerate(zip(sub_ys, b.get("sub", []))):
        ax.text(b["x"] + b["w"]/2, sy, stxt,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SUB, color=col, alpha=0.88,
                clip_on=False)


for b in BOXES.values():
    draw_box(ax, b)

# ── dessin des flèches ────────────────────────────────────────

def draw_arrow(ax, x0, y0, x1, y1, color, style, label=""):
    col = C[color]
    ls  = (0, (5, 3)) if style == "dashed" else "solid"

    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=col,
            lw=LW_ARROW,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        clip_on=False,
    )
    if label:
        mx, my = (x0+x1)/2 + 0.025, (y0+y1)/2
        ax.text(mx, my, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color=col, alpha=0.85, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))


for arr in ARROWS:
    draw_arrow(ax, *arr)


# ─────────────────────────────────────────────────────────────
# 3. PATCHES
# ─────────────────────────────────────────────────────────────
def draw_patch(ax, x0, y0, lx, ly, color):
    col = C[color]

    ax.add_patch(Rectangle((x0, y0), lx, ly,
                            transform=ax.transAxes,
                            facecolor=col, alpha=0.1, edgecolor=col,
                            linewidth=1.2, linestyle=":", clip_on=False, zorder=0))

for patch in PATCHES:
    draw_patch(ax, *patch)

# ── badges ────────────────────────────────────────────────────

for (bx, by, txt, ckey) in BADGES:
    ax.text(bx, by, txt,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.35", fc=C[ckey], ec="none"))

# ── textes libres ─────────────────────────────────────────────

for (tx, ty, txt, ckey, fs, bold) in FREE_TEXTS:
    fw = "bold" if bold else "normal"
    ax.text(tx, ty, txt,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, color=C[ckey], fontweight=fw,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=C[ckey], alpha=0.85, lw=0.8))

# ── zone d'interpolation (ligne pointillée) ───────────────────

# ax.plot([0.01, 0.99], [0.44, 0.46],
#         transform=ax.transAxes,
#         color=C["interp"], lw=0.8, ls="--", alpha=0.45, clip_on=False)
# ax.text(0.995, 0.445, r"$\longleftarrow$ interpolation",
#         transform=ax.transAxes, ha="right", va="bottom",
#         fontsize=7.5, color=C["interp"], alpha=0.7, clip_on=False)

# ── légende ───────────────────────────────────────────────────

legend_items = [
    mpatches.Patch(facecolor=C["input"]  + "28", edgecolor=C["input"],  label="Input data"),
    mpatches.Patch(facecolor=C["model"]  + "28", edgecolor=C["model"],  label="Model output"),
    mpatches.Patch(facecolor=C["obs"]    + "28", edgecolor=C["obs"],    label="Observed timeseries"),
    mpatches.Patch(facecolor=C["interp"] + "28", edgecolor=C["interp"], label="Interpolated timeseries"),
    mpatches.Patch(facecolor=C["result"] + "28", edgecolor=C["result"], label="Final result"),
]
ax.legend(handles=legend_items,
          loc="lower center", ncol=5,
          bbox_to_anchor=(0.5, -0.04),
          frameon=True, framealpha=0.95,
          fontsize=8.5,
          edgecolor="#c8bfb2")

plt.tight_layout(pad=0.5)

# ── sauvegarde ────────────────────────────────────────────────

fig.savefig(fig_dir / "methods_synthesis.pdf", bbox_inches="tight", dpi=200)
print(f"Figure saved → methods_synthesis")
plt.show()
















C = {
    "box"    : "#5a5248",   # rouge  — données d'entrée
    "interp" : "#c06010",   # orange — interpolées
    "feedback"  : "#5e3a7a",
    "result" : "#0d5c42",   # vert   — résultat final
    "arrow"  : "#1a5f8a",   # gris   — flèches neutres
}

BOXES = {

    # ── Colonne gauche : dates DEM ──────────────────────────
    "dem": dict(
        x=0.30, y=0.84, w=0.40, h=0.08,
        color="box",
        title=r"Surface & Bed DEMs",
        sub=[],
        style="solid",
    ),
    "geom": dict(
        x=0.30, y=0.68, w=0.40, h=0.08,
        color="box",
        title=r"Observed geometry",
        sub=[],
        style="solid",
    ),
    "tau_vel_fields": dict(
        x=0.30, y=0.52, w=0.40, h=0.08,
        color="box",
        title="Stress & deformation velocity \n fields",
        sub=[],
        style="solid",
    ),


    # ── Empirical relationship ────────────────────────
    "rel": dict(
        x=0.30, y=0.36, w=0.40, h=0.08,
        color="box",
        title=r"Empirical relationship",
        sub=[r"$\tau_b(t_\mathrm{DEM}) = f_1(H(t_\mathrm{DEM})) \quad & \quad u_{def} = f_2(H(t_\mathrm{DEM}))$"],
        style="solid",
    ),

    # ── Interpolation ────────────────────────────────
    "interp": dict(
        x=0.30, y=0.20, w=0.40, h=0.08,
        color="box",
        title=r"Temporal interpolation",
        sub=[r"$\tau_b(t_\mathrm{Stake}) = f_1(H(t_\mathrm{Stake})) \quad & \quad u_b = u_s - f_2(H(t_\mathrm{Stake}))$"],
    style="solid",
    ),

    # ── Résultat final ──────────────────────────────────────
    "law": dict(
        x=0.30, y=0.04, w=0.40, h=0.08,
        color="box",
        title=r"Friction Law",
        sub=[r"$\tau_b = f(u_b)$"],
        style="solid",
    ),
}


# ─────────────────────────────────────────────────────────────
# 3. FLÈCHES
# ─────────────────────────────────────────────────────────────
#

def cx(k): return BOXES[k]["x"] + BOXES[k]["w"] / 2   # centre x
def cy(k): return BOXES[k]["y"] + BOXES[k]["h"] / 2   # centre y
def top(k): return BOXES[k]["y"] + BOXES[k]["h"]
def bot(k): return BOXES[k]["y"]
def left(k): return BOXES[k]["x"]
def right(k): return BOXES[k]["x"] + BOXES[k]["w"]

ARROWS = [
    # (x0,       y0,          x1,        y1,         color,     style,    label)
    (cx("dem"),  bot("dem"),   cx("geom"), top("geom"),   "arrow",  "solid",  ""),
    (cx("geom"), bot("geom"),  cx("tau_vel_fields"),  top("tau_vel_fields"),   "feedback",  "solid",  r"Elmer/Ice (with $A_s \tau_b^m = u_s$)"),
    (cx("tau_vel_fields"), bot("tau_vel_fields"), cx("rel"), top("rel"),  "arrow",  "solid",  ""),
    (cx("rel"), bot("rel"), cx("interp"),   top("interp"),   "interp",  "solid",  r"Thickness $H$ and surface velocity $u_s$ observed at stakes"),
    (cx("interp"), bot("interp"), cx("law"), top("law"), "arrow", "solid", ""),
    (left("law"), cy("law"), 0.35, (cy("geom") + cy("tau_vel_fields"))/2, "feedback", "solid", "", "left"),
]


# ─────────────────────────────────────────────────────────────
# 4. ANNOTATIONS  (badges de colonnes, label SIA)
# ─────────────────────────────────────────────────────────────

# Texte libre au centre (SIA)
FREE_TEXTS = []


# ─────────────────────────────────────────────────────────────
# 5. RENDU
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── dessin des boîtes ─────────────────────────────────────────

def draw_box(ax, b):
    col   = C[b["color"]]
    alpha = ALPHA_FILL
    ls    = "--" if b.get("style") == "dashed" else "-"

    # fond + contour
    rect = FancyBboxPatch(
        (b["x"], b["y"]), b["w"], b["h"],
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor=col + f"{int(alpha*255):02x}",
        edgecolor=col, linewidth=LW_BOX,
        linestyle=ls, clip_on=False,
    )
    ax.add_patch(rect)

    # barre gauche décorative
    # bar = FancyBboxPatch(
    #     (b["x"], b["y"]), 0.006, b["h"],
    #     boxstyle="round,pad=0.001",
    #     transform=ax.transAxes,
    #     facecolor=col, edgecolor="none",
    #     clip_on=False,
    # )
    # ax.add_patch(bar)

    # positions verticales selon nombre de sous-titres
    n = len(b.get("sub", []))
    if   n == 0: ty = b["y"] + b["h"] * 0.50
    elif n == 1: ty = b["y"] + b["h"] * 0.68
    else         : ty = b["y"] + b["h"] * 0.78

    ax.text(b["x"] + b["w"]/2, ty, b["title"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold", color=col,
            clip_on=False)

    sub_ys = []
    if n == 1:
        sub_ys = [b["y"] + b["h"] * 0.33]
    elif n == 2:
        sub_ys = [b["y"] + b["h"] * 0.47, b["y"] + b["h"] * 0.18]

    for i, (sy, stxt) in enumerate(zip(sub_ys, b.get("sub", []))):
        ax.text(b["x"] + b["w"]/2, sy, stxt,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SUB, color=col, alpha=0.88,
                clip_on=False)


for b in BOXES.values():
    draw_box(ax, b)

# ── dessin des flèches ────────────────────────────────────────

def draw_arrow(ax, x0, y0, x1, y1, color, style, label="", curve=0.0):
    """
    curve = 0.0       -> droite
    curve = -0.4      -> arc courbé vers la gauche
    curve = "left"    -> trois segments passant par la gauche
    curve = "right"   -> trois segments passant par la droite
    """
    col = C[color]
    ls  = (0, (5, 3)) if style == "dashed" else "solid"

    if curve == "left":
        connstyle = "bar,angle=90,fraction=-0.3"
    elif curve == "right":
        connstyle = "bar,angle=90,fraction=0.3"
    else:
        connstyle = f"arc3,rad={curve}"

    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=col,
            lw=LW_ARROW,
            linestyle=ls,
            connectionstyle=connstyle,
        ),
        clip_on=False,
    )
    if label:
        # Décaler le label vers la gauche si la flèche passe à gauche
        offset_x = -0.12 if (isinstance(curve, str) and "left" in curve) \
                         or (isinstance(curve, float) and curve < 0) \
                   else 0.025
        mx = (x0 + x1) / 2 + offset_x
        my = (y0 + y1) / 2
        ax.text(mx, my, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=col, alpha=0.85, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))


for arr in ARROWS:
    draw_arrow(ax, *arr)


# ── textes libres ─────────────────────────────────────────────

for (tx, ty, txt, ckey, fs, bold) in FREE_TEXTS:
    fw = "bold" if bold else "normal"
    ax.text(tx, ty, txt,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, color=C[ckey], fontweight=fw,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=C[ckey], alpha=0.85, lw=0.8))

# ── zone d'interpolation (ligne pointillée) ───────────────────

# ax.plot([0.01, 0.99], [0.44, 0.46],
#         transform=ax.transAxes,
#         color=C["interp"], lw=0.8, ls="--", alpha=0.45, clip_on=False)
# ax.text(0.995, 0.445, r"$\longleftarrow$ interpolation",
#         transform=ax.transAxes, ha="right", va="bottom",
#         fontsize=7.5, color=C["interp"], alpha=0.7, clip_on=False)

# ── légende ───────────────────────────────────────────────────

# legend_items = [
#     mpatches.Patch(facecolor=C["input"]  + "28", edgecolor=C["input"],  label="Input data"),
#     mpatches.Patch(facecolor=C["model"]  + "28", edgecolor=C["model"],  label="Model output"),
#     mpatches.Patch(facecolor=C["obs"]    + "28", edgecolor=C["obs"],    label="Observed timeseries"),
#     mpatches.Patch(facecolor=C["interp"] + "28", edgecolor=C["interp"], label="Interpolated timeseries"),
#     mpatches.Patch(facecolor=C["result"] + "28", edgecolor=C["result"], label="Final result"),
# ]
# ax.legend(handles=legend_items,
#           loc="lower center", ncol=5,
#           bbox_to_anchor=(0.5, -0.04),
#           frameon=True, framealpha=0.95,
#           fontsize=8.5,
#           edgecolor="#c8bfb2")

plt.tight_layout(pad=0.5)

# ── sauvegarde ────────────────────────────────────────────────

fig.savefig(fig_dir / "methods_synthesis_simple.pdf", bbox_inches="tight", dpi=200)
print(f"Figure saved → methods_synthesis")
plt.show()














C = {
    "input"  : "#a8392b",   # rouge  — données d'entrée
    "model"  : "#1a5f8a",   # bleu   — sorties de modèle
    "obs"    : "#c06010",   # violet — séries observées continues
    "interp" : "#5e3a7a",   # orange — interpolées
    "result" : "#0d5c42",   # vert   — résultat final
    "arrow"  : "#5a5248",   # gris   — flèches neutres
    "bg"     : "#fdfbf8",
}

ALPHA_FILL  = 0.12    # transparence du fond des boîtes
LW_BOX      = 1.6     # épaisseur du contour des boîtes
LW_ARROW    = 1.3     # épaisseur des flèches
FONT_TITLE  = 13      # taille du titre dans les boîtes
FONT_SUB    = 12        # taille des sous-titres

plt.rcParams.update({
    "font.family"    : "serif",
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": C["bg"],
    "axes.facecolor"  : C["bg"],
})


# ─────────────────────────────────────────────────────────────
# 2. BOÎTES  (x, y, w, h tous en axes fraction)
# ─────────────────────────────────────────────────────────────
#
# Format d'une boîte :
#   "id": dict(
#       x, y, w, h      — position & taille
#       color           — clé dans C{}
#       title           — texte principal (LaTeX ok)
#       sub             — liste de sous-titres ([] si aucun)
#       style           — "solid" | "dashed"
#   )
#
BOXES = {

    # ── Colonne gauche : dates DEM ──────────────────────────
    "dem": dict(
        x=0.04, y=0.84, w=0.42, h=0.10,
        color="model",
        title=r"Surface & Bed DEMs",
        sub=[],
        style="solid",
    ),
    "stokes": dict(
        x=0.04, y=0.64, w=0.42, h=0.10,
        color="model",
        title=r"Stress & velocity fields",
        sub=[],
        style="solid",
    ),
    "emp_rel": dict(
        x=0.04, y=0.44, w=0.42, h=0.10,
        color="model",
        title=r"Empirical relationships",
        sub=[r"$\tau_b(t_\mathrm{DEM}) = f_1(H(t_\mathrm{DEM})) \quad & \quad u_{def}(t_\mathrm{DEM}) = f_2(H(t_\mathrm{DEM}))$"],
        style="solid",
    ),


    # ── Colonne droite : dates Stake ────────────────────────
    "obs": dict(
        x=0.54, y=0.44, w=0.42, h=0.12,
        color="obs",
        title="Field observations",
        sub=["thickness " + r"$H(t_\mathrm{Stake})$" + "& surface velocity " + r"$u_s(t_\mathrm{Stake})$"],
        style="solid",
    ),


    # ── Interpolation ────────────────────────────────
    "interp": dict(
        x=0.10, y=0.24, w=0.80, h=0.10,
        color="interp",
        title=r"Temporal interpolation of basal shear stress & sliding velocity",
        sub=[r"$\tau_b(t_\mathrm{Stake}) = f_1(H(t_\mathrm{Stake})) \quad & \quad u_b(t_\mathrm{Stake}) = u_s(t_\mathrm{Stake}) - f_2(H(t_\mathrm{Stake}))$"],
    style="solid",
    ),

    # ── Résultat final ──────────────────────────────────────
    "law": dict(
        x=0.35, y=0.04, w=0.30, h=0.10,
        color="result",
        title=r"Friction Law :  $\tau_b = f(u_\mathrm{bed})$",
        sub=[],
        style="solid",
    ),
}


# ─────────────────────────────────────────────────────────────
# 3. FLÈCHES
# ─────────────────────────────────────────────────────────────
#
# Format :
#   (x0, y0, x1, y1, color_key, style, label)
#
#   style : "solid" | "dashed"
#   label : "" si pas de label
#
# Les coordonnées sont en axes fraction.
# Astuce : (box["x"] + box["w"]/2) = centre horizontal d'une boîte
#          (box["y"])               = bord bas,  (box["y"]+box["h"]) = bord haut

def cx(k): return BOXES[k]["x"] + BOXES[k]["w"] / 2   # centre x
def cy(k): return BOXES[k]["y"] + BOXES[k]["h"] / 2   # centre y
def top(k): return BOXES[k]["y"] + BOXES[k]["h"]
def bot(k): return BOXES[k]["y"]
def left(k): return BOXES[k]["x"]
def right(k): return BOXES[k]["x"] + BOXES[k]["w"]

ARROWS = [
    # (x0,       y0,          x1,        y1,         color,     style,    label)
    # DEM → Stokes
    (cx("dem"),  bot("dem"),   cx("stokes"), top("stokes"),   "model",  "solid",  r"Elmer/Ice (with $A_s \tau_b^m = u_s$)"),
    # Stokes → emp_rel
    (cx("stokes"), bot("stokes"), cx("emp_rel"),  top("emp_rel"),   "model",  "solid",  ""),
    # Emp_rel → interp
    (cx("emp_rel"), bot("emp_rel"), cx("interp"),  top("interp"),   "model",  "solid",  ""),
    # Obs → interp
    (cx("obs"), bot("obs"), cx("interp"),  top("interp"),   "obs",  "solid",  ""),
    # Interp → Law
    (cx("interp"), bot("interp"), cx("law"),  top("law"),   "result",  "solid",  ""),
]


# ─────────────────────────────────────────────────────────────
# 4. ANNOTATIONS  (badges de colonnes, label SIA)
# ─────────────────────────────────────────────────────────────

PATCHES=[
    (0.02, 0.42, 0.46, 0.54, "model"),
    (0.52, 0.42, 0.46, 0.16, "obs")]


BADGES = [
    # (x_centre, y_centre, texte, color_key)
    (0.25, 0.98, r"Specific dates  $t_\mathrm{DEM}$",   "model"),
    (0.75, 0.60, r"Annual timeseries  $t_\mathrm{Stake}$", "obs"),
]

# Texte libre au centre (SIA)
FREE_TEXTS = [
    # (x, y, texte, color_key, fontsize, bold, box)
    (0.50, 0.49, "+", "arrow", 24, False, True), 
    (0.355, 0.265, "__", "model", 12, True, False), 
    (0.40, 0.265, "_______", "obs", 12, True, False), 
    (0.595, 0.265, "________", "obs", 12, True, False), 
    (0.655, 0.265, "__", "model", 12, True, False), 
    (0.70, 0.265, "_______", "obs", 12, True, False), 
    ]


# ─────────────────────────────────────────────────────────────
# 5. RENDU
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── dessin des boîtes ─────────────────────────────────────────

def draw_box(ax, b):
    col   = C[b["color"]]
    alpha = ALPHA_FILL
    ls    = "--" if b.get("style") == "dashed" else "-"

    # fond + contour
    rect = FancyBboxPatch(
        (b["x"], b["y"]), b["w"], b["h"],
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor=col + f"{int(alpha*255):02x}",
        edgecolor=col, linewidth=LW_BOX,
        linestyle=ls, clip_on=False,
    )
    ax.add_patch(rect)

    # barre gauche décorative
    bar = FancyBboxPatch(
        (b["x"]-0.006, b["y"]), 0.006, b["h"],
        boxstyle="round,pad=0.001",
        transform=ax.transAxes,
        facecolor=col, edgecolor="none",
        clip_on=False,
    )
    ax.add_patch(bar)

    # positions verticales selon nombre de sous-titres
    n = len(b.get("sub", []))
    if   n == 0: ty = b["y"] + b["h"] * 0.50
    elif n == 1: ty = b["y"] + b["h"] * 0.68
    else         : ty = b["y"] + b["h"] * 0.78

    ax.text(b["x"] + b["w"]/2, ty, b["title"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold", color=col,
            clip_on=False)

    sub_ys = []
    if n == 1:
        sub_ys = [b["y"] + b["h"] * 0.33]
    elif n == 2:
        sub_ys = [b["y"] + b["h"] * 0.47, b["y"] + b["h"] * 0.18]

    for i, (sy, stxt) in enumerate(zip(sub_ys, b.get("sub", []))):
        ax.text(b["x"] + b["w"]/2, sy, stxt,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SUB, color=col, alpha=0.88,
                clip_on=False)


for b in BOXES.values():
    draw_box(ax, b)

# ── dessin des flèches ────────────────────────────────────────

def draw_arrow(ax, x0, y0, x1, y1, color, style, label=""):
    col = C[color]
    ls  = (0, (5, 3)) if style == "dashed" else "solid"

    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=col,
            lw=LW_ARROW,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        clip_on=False,
    )
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=col, alpha=0.85, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))


for arr in ARROWS:
    draw_arrow(ax, *arr)


# ─────────────────────────────────────────────────────────────
# 3. PATCHES
# ─────────────────────────────────────────────────────────────
def draw_patch(ax, x0, y0, lx, ly, color):
    col = C[color]

    ax.add_patch(Rectangle((x0, y0), lx, ly,
                            transform=ax.transAxes,
                            facecolor=col, alpha=0.2, edgecolor=col,
                            linewidth=1.2, linestyle=":", clip_on=False, zorder=0))

for patch in PATCHES:
    draw_patch(ax, *patch)

# ── badges ────────────────────────────────────────────────────

for (bx, by, txt, ckey) in BADGES:
    ax.text(bx, by, txt,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.35", fc=C[ckey], ec="none"))

# ── textes libres ─────────────────────────────────────────────

for (tx, ty, txt, ckey, fs, bold, box) in FREE_TEXTS:
    fw = "bold" if bold else "normal"
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec=C[ckey], alpha=0.85, lw=0.8) if box else None
    ax.text(tx, ty, txt,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, color=C[ckey], fontweight=fw,
            clip_on=False, bbox=bbox)


plt.tight_layout(pad=0.5)

# ── sauvegarde ────────────────────────────────────────────────

fig.savefig(fig_dir / "methods_workflow.pdf", bbox_inches="tight", dpi=200)
print(f"Figure saved → methods_synthesis")
plt.show()
