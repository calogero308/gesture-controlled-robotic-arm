"""
Simulazione del braccio RRR con matplotlib.

Una sequenza di gesti (scritta a mano, al posto del video) viene tradotta in
comandi; ogni comando sposta la punta di un passo, la cinematica inversa
calcola gli angoli e il braccio viene disegnato in 3D e di profilo.

Se un comando porta la punta in un punto non raggiungibile, il braccio resta fermo.

Esecuzione (da python/control):
    python3 simulate.py                  finestra interattiva
    python3 simulate.py --gif out.gif    salva una GIF (utile se la finestra non si apre)
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# il modulo del braccio sta in python/arm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arm"))

from arm_config import H, L1, L2
from kinematics import forward_kinematics, inverse_kinematics
from viz import joint_positions
from gesture_to_command import gesture_to_command, apply_command

# --- Parametri della simulazione ---
HOME_ANGLES = (0.0, math.radians(45), math.radians(-90))   # posizione di partenza
SUBSTEPS = 3          # fotogrammi di interpolazione tra un comando e il successivo
FPS = 15

# Sequenza di gesti: ciascuno vale un comando (un passo). None = nessun gesto stabile.
SEQUENCE = (
    ["like"] * 8        # su
    + ["rock"] * 10     # destra
    + ["fist"] * 8      # giù
    + ["peace"] * 20    # sinistra
    + [None] * 3        # nessun gesto: il braccio resta fermo
    + ["like"] * 22     # su, fino a toccare il limite dello spazio di lavoro
    + ["palm"] * 3      # stop
)


def run_sequence(gestures, home_angles):
    """Applica i gesti uno alla volta. Restituisce la lista degli stati, senza disegnare."""
    angles = home_angles
    pos = forward_kinematics(*angles)
    states = [dict(gesture=None, command=None, pos=pos, angles=angles, status="posizione iniziale")]

    for g in gestures:
        cmd = gesture_to_command(g)
        if cmd is None:
            status = "nessun comando"
        elif cmd == (0, 0, 0):
            status = "stop"
        else:
            target = apply_command(pos, cmd)
            soluzione = inverse_kinematics(*target)
            if soluzione is None:
                status = "BLOCCATO: punto non raggiungibile"
            else:
                pos, angles, status = target, soluzione, "ok"
        states.append(dict(gesture=g, command=cmd, pos=pos, angles=angles, status=status))
    return states


def interpolate(states):
    """Fotogrammi intermedi tra due stati consecutivi (solo per fluidità del disegno)."""
    frames = []
    for prev, cur in zip(states, states[1:]):
        for k in range(1, SUBSTEPS + 1):
            t = k / SUBSTEPS
            angles = tuple(a + (b - a) * t for a, b in zip(prev["angles"], cur["angles"]))
            frames.append((angles, cur))
    return frames


def build_figure():
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_side = fig.add_subplot(gs[1])

    # --- vista 3D ---
    ax3d.set_xlim(-0.05, 0.32)
    ax3d.set_ylim(-0.32, 0.32)
    ax3d.set_zlim(0, 0.40)
    ax3d.set_box_aspect((0.37, 0.64, 0.40))
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.plot([-0.05, 0.32, 0.32, -0.05, -0.05], [-0.32, -0.32, 0.32, 0.32, -0.32], [0] * 5,
              color="0.6", lw=1)  # bordo del tavolo

    # --- vista di profilo: distanza orizzontale r contro altezza z ---
    ax_side.set_xlim(-0.05, 0.34)
    ax_side.set_ylim(-0.02, 0.42)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("distanza dall'asse della base r (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.axhline(0, color="0.6", lw=1)
    ax_side.grid(alpha=0.25)
    ax_side.set_title("vista di profilo")

    return fig, ax3d, ax_side


def main():
    states = run_sequence(SEQUENCE, HOME_ANGLES)
    frames = interpolate(states)
    fig, ax3d, ax_side = build_figure()

    arm3d, = ax3d.plot([], [], [], "-o", color="tab:blue", lw=4, ms=7)
    trail3d, = ax3d.plot([], [], [], "-", color="tab:orange", lw=1.2)
    arm_side, = ax_side.plot([], [], "-o", color="tab:blue", lw=4, ms=7)
    hud = fig.text(0.02, 0.02, "", family="monospace", fontsize=9, va="bottom")
    trail = []

    def update(i):
        angles, state = frames[i]
        pts = joint_positions(*angles)
        xs, ys, zs = zip(*pts)
        arm3d.set_data(xs, ys)
        arm3d.set_3d_properties(zs)

        trail.append(pts[-1])
        tx, ty, tz = zip(*trail)
        trail3d.set_data(tx, ty)
        trail3d.set_3d_properties(tz)

        rs = [math.hypot(x, y) for x, y, _ in pts]
        arm_side.set_data(rs, zs)

        t1, t2, t3 = (math.degrees(a) for a in angles)
        x, y, z = pts[-1]
        hud.set_text(
            f"gesto: {str(state['gesture']):<6}  comando: {str(state['command']):<11}  {state['status']}\n"
            f"theta1={t1:6.1f}  theta2={t2:6.1f}  theta3={t3:6.1f} gradi    "
            f"punta: x={x:.3f} y={y:.3f} z={z:.3f} m"
        )
        return arm3d, trail3d, arm_side, hud

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False, repeat=False)
    fig.subplots_adjust(bottom=0.20, top=0.92, left=0.0, right=0.97, wspace=0.08)

    if "--gif" in sys.argv:
        out = sys.argv[sys.argv.index("--gif") + 1]
        anim.save(out, writer=PillowWriter(fps=FPS), dpi=70)
        print("GIF salvata in", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
