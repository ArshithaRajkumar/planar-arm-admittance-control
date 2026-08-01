"""
plot_results.py - Generate result plots for the Chopstick Crane simulation
===========================================================================
Loads results/data.npz and produces 5 publication-quality figures.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kinematics import (
    BOARD_HINGE_POS, SWEEP_OFFSET, SWEEP_AMPLITUDE,
    F_MIN, F_MAX, F_TARGET, target_curve,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA_PATH   = os.path.join(RESULTS_DIR, "data.npz")

# Dark style
plt.rcParams.update({
    "figure.facecolor":  "#12121e",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#3a3a5c",
    "axes.labelcolor":   "#c8c8d8",
    "axes.titlecolor":   "#e8e8f0",
    "axes.grid":          True,
    "grid.color":        "#2a2a4a",
    "xtick.color":       "#a0a0c0",
    "ytick.color":       "#a0a0c0",
    "text.color":        "#c8c8d8",
    "lines.linewidth":    2.0,
    "font.size":         11,
})

PHASE_COLORS = {
    1: "#2244aa",   # WARMUP  - blue
    2: "#aa4422",   # SWEEP   - orange
    3: "#226644",   # RETURN  - green
    4: "#444444",   # DONE    - grey
}
PHASE_NAMES = {1: "Warm-up", 2: "Sweep", 3: "Return", 4: "Done"}


def phase_spans(t, phase):
    """Return list of (t_start, t_end, phase_val) for shading."""
    spans = []
    if len(t) == 0:
        return spans
    cur_p = phase[0]; t0 = t[0]
    for i in range(1, len(t)):
        if phase[i] != cur_p:
            spans.append((t0, t[i], cur_p))
            cur_p = phase[i]; t0 = t[i]
    spans.append((t0, t[-1], cur_p))
    return spans


def add_phase_shading(ax, t, phase):
    for (ts, te, p) in phase_spans(t, phase):
        ax.axvspan(ts, te, alpha=0.15, color=PHASE_COLORS.get(p, "#444444"), lw=0)


def plot_all(logs: dict, out_dir: str = RESULTS_DIR):
    t      = logs["t"]
    tip    = logs["tip"]
    target = logs["target"]
    phi    = logs["phi"]
    Fn     = logs["Fn"]
    phase  = logs["phase"]
    theta  = logs["theta"]

    sweep  = (phase == 2)
    err    = np.linalg.norm(tip - target, axis=1)

    os.makedirs(out_dir, exist_ok=True)

    # ── Fig 1: Trajectory ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Pen-Tip Trajectory vs. Target Curve")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")

    # Board region
    bx = BOARD_HINGE_POS[0]
    bz = BOARD_HINGE_POS[1]
    board_rect = plt.Rectangle((bx - 0.05, bz - 0.015), 0.32, 0.015,
                                color="#332200", alpha=0.5, label="Board region")
    ax.add_patch(board_rect)

    # Target curve at phi=0
    sv   = np.linspace(0, 1, 200)
    tc   = np.array([target_curve(s, 0.0) for s in sv])
    ax.plot(tc[:, 0], tc[:, 1], "g--", lw=2, label="Target curve")

    # Actual pen tip
    ax.plot(tip[:, 0], tip[:, 1], color="#ff6633", lw=1.5, label="Pen tip (actual)")
    ax.scatter(*tip[0],  color="#6699ff", s=80, zorder=5, label="Start")
    ax.scatter(*tip[-1], color="#ff33aa", s=80, marker="x", zorder=5, linewidths=2, label="End")

    ax.scatter(*BOARD_HINGE_POS, color="#ffdd44", s=100, zorder=6, label="Board hinge A")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_trajectory.png"), dpi=120)
    plt.close(fig)

    # ── Fig 2: Board tilt ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Board Tilt phi(t)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("phi [deg]")
    add_phase_shading(ax, t, phase)
    ax.plot(t, np.degrees(phi), color="#66aaff", lw=2)
    ax.axhline(0, color="#555577", ls=":", lw=1)
    # Legend for phases
    patches = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.5, label=PHASE_NAMES[p])
               for p in PHASE_COLORS]
    ax.legend(handles=patches, fontsize=9, title="Phase")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_board_tilt.png"), dpi=120)
    plt.close(fig)

    # ── Fig 3: Contact force ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Contact Normal Force  Fn(t)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Fn [N]")
    add_phase_shading(ax, t, phase)
    ax.axhspan(F_MIN, F_MAX, alpha=0.15, color="#44ff99", label=f"Target band [{F_MIN}, {F_MAX}] N")
    ax.axhline(F_TARGET, color="#44ff99", ls="--", lw=1.5, label=f"F* = {F_TARGET} N")
    ax.plot(t, Fn, color="#ff9944", lw=2)
    patches = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.5, label=PHASE_NAMES[p])
               for p in PHASE_COLORS]
    ax.legend(handles=patches + [
        mpatches.Patch(color="#44ff99", alpha=0.3, label=f"Band [{F_MIN},{F_MAX}]N")
    ], fontsize=9, title="Phase")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_contact_force.png"), dpi=120)
    plt.close(fig)

    # ── Fig 4: Tracking error ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Tracking Error  e(t) = ||tip - target||")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("e [mm]")
    add_phase_shading(ax, t, phase)
    ax.plot(t, err * 1000, color="#cc88ff", lw=2)
    if sweep.sum() > 0:
        ax.axhline(err[sweep].mean() * 1000, color="#cc88ff", ls="--", lw=1.5,
                   label=f"Sweep mean = {err[sweep].mean()*1000:.1f}mm")
    patches = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.5, label=PHASE_NAMES[p])
               for p in PHASE_COLORS]
    ax.legend(handles=patches, fontsize=9, title="Phase")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_tracking_error.png"), dpi=120)
    plt.close(fig)

    # ── Fig 5: Joint angles ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Joint Angles theta(t)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angle [deg]")
    add_phase_shading(ax, t, phase)
    colors = ["#ff6688", "#66aaff", "#88ff99"]
    for i, (c, lbl) in enumerate(zip(colors, ["theta1 (shoulder)", "theta2 (elbow)", "theta3 (wrist)"])):
        ax.plot(t, np.degrees(theta[:, i]), color=c, lw=2, label=lbl)
    patches = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.5, label=PHASE_NAMES[p])
               for p in PHASE_COLORS]
    ax.legend(handles=patches, fontsize=9, title="Phase")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_joint_angles.png"), dpi=120)
    plt.close(fig)

    print(f"[Plots]  Saved 5 figures to {out_dir}/")

    # Print sweep stats
    print(f"\n-- Sweep Phase Statistics --")
    if sweep.sum() > 0:
        print(f"  Tracking error: mean={err[sweep].mean()*1000:.1f}mm  max={err[sweep].max()*1000:.1f}mm")
        print(f"  Contact force:  mean={Fn[sweep].mean():.2f}N  in-band={100*np.mean((Fn[sweep]>=F_MIN)&(Fn[sweep]<=F_MAX)):.1f}%")
        print(f"  Board tilt:     mean={np.degrees(phi[sweep]).mean():.1f} deg  "
              f"range=[{np.degrees(phi[sweep]).min():.1f} deg, {np.degrees(phi[sweep]).max():.1f} deg]")
    else:
        print("  No sweep data found.")


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        print("Run simulate.py first.")
    else:
        logs = {k: v for k, v in np.load(DATA_PATH, allow_pickle=True).items()}
        plot_all(logs)
