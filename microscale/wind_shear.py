"""
Wind shear gradient visualisation — three separate figures.

Model: V_wy(h) = V_ref * (h / 10)^0.143   (power-law boundary-layer profile)

Each figure highlights one V_ref with horizontal shear arrows from x=0 to the
curve; the other two V_refs are overlaid at very low alpha for context.
"""

import numpy as np
import matplotlib.pyplot as plt

h = np.linspace(0.0, 25, 500)
h_ref = 10.0
p = 0.143

V_refs = [5, 10, 15]
colors  = ["#2196F3", "#FF5722", "#4CAF50"]   # blue, deep-orange, green

# Heights at which to draw shear arrows (evenly spaced, avoid h=0)
arrow_heights = np.linspace(1, 24, 16)

for v_ref_main, color_main in zip(V_refs, colors):

    fig, ax = plt.subplots(figsize=(5, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- ghost profiles ---
    for v_ref, color in zip(V_refs, colors):
        if v_ref != v_ref_main:
            V_w = v_ref * (h / h_ref) ** p
            ax.plot(V_w, h, color=color, linewidth=1.4, alpha=0.4)

    # --- active profile ---
    V_w_main = v_ref_main * (h / h_ref) ** p
    ax.plot(V_w_main, h, color=color_main, linewidth=2.2, zorder=3)

    # --- shear arrows (left edge → curve) ---
    x_left = 0.0
    for h_a in arrow_heights:
        x_tip = float(v_ref_main * (h_a / h_ref) ** p)
        length = x_tip - x_left
        if length < 0.05:
            continue
        ax.annotate(
            "",
            xy=(x_tip, h_a),
            xytext=(x_left, h_a),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color_main,
                alpha=0.55,
                lw=0.9,
                mutation_scale=7,
            ),
            zorder=2,
        )

    # --- reference height dotted line ---
    x_max = ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else v_ref_main * 1.15
    ax.axhline(h_ref, color="grey", linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    ax.set_xlabel("Wind speed $V_w$ [m/s]", fontsize=11)
    ax.set_ylabel("Height $h$ [m]", fontsize=11)
    ax.set_ylim(0, 25)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25, color="grey")

    plt.tight_layout()
    fname = f"figures/microscale/wind_shear_vref{v_ref_main}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {fname}")
