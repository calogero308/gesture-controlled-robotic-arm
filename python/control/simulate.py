"""
Simulazione del braccio RRR con matplotlib.

Una sequenza di gesti (scritta a mano, o presa da un video) viene tradotta in
comandi; ogni comando sposta la punta di un passo, la cinematica inversa
calcola gli angoli e il braccio viene disegnato in 3D e di profilo.

Se un comando porta la punta in un punto non raggiungibile, il braccio resta fermo.

Esecuzione (da python/control):
    python3 simulate.py                  finestra interattiva
    python3 simulate.py --gif out.gif    salva una GIF (utile se la finestra non si apre)
"""

import math
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from arm_state import HOME_ANGLES, initial_state, step
from viz import joint_positions

# --- Parametri della simulazione ---
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


def run_sequence(gestures, home_angles=HOME_ANGLES):
    """Applica i gesti uno alla volta. Restituisce la lista degli stati, senza disegnare."""
    state = initial_state(home_angles)
    states = [state]
    for g in gestures:
        state = step(state, g)
        states.append(state)
    return states


def interpolate(states, substeps=SUBSTEPS):
    """Fotogrammi intermedi tra due stati consecutivi (solo per fluidità del disegno)."""
    frames = []
    for prev, cur in zip(states, states[1:]):
        for k in range(1, substeps + 1):
            t = k / substeps
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

    fig.subplots_adjust(bottom=0.20, top=0.92, left=0.0, right=0.97, wspace=0.08)
    return fig, ax3d, ax_side


class ArmView:
    """Finestra con il braccio: draw(angoli, stato) lo ridisegna. Serve sia all'animazione sia al dal vivo."""

    def __init__(self):
        self.fig, self.ax3d, self.ax_side = build_figure()
        self.arm3d, = self.ax3d.plot([], [], [], "-o", color="tab:blue", lw=4, ms=7)
        self.trail3d, = self.ax3d.plot([], [], [], "-", color="tab:orange", lw=1.2)
        self.arm_side, = self.ax_side.plot([], [], "-o", color="tab:blue", lw=4, ms=7)
        self.hud = self.fig.text(0.02, 0.02, "", family="monospace", fontsize=9, va="bottom")
        self.trail = []

    def draw(self, angles, state):
        pts = joint_positions(*angles)
        xs, ys, zs = zip(*pts)
        self.arm3d.set_data(xs, ys)
        self.arm3d.set_3d_properties(zs)

        self.trail.append(pts[-1])
        tx, ty, tz = zip(*self.trail)
        self.trail3d.set_data(tx, ty)
        self.trail3d.set_3d_properties(tz)

        rs = [math.hypot(x, y) for x, y, _ in pts]
        self.arm_side.set_data(rs, zs)

        t1, t2, t3 = (math.degrees(a) for a in angles)
        x, y, z = pts[-1]
        self.hud.set_text(
            f"gesto: {str(state['gesture']):<6}  comando: {str(state['command']):<11}  {state['status']}\n"
            f"theta1={t1:6.1f}  theta2={t2:6.1f}  theta3={t3:6.1f} gradi    "
            f"punta: x={x:.3f} y={y:.3f} z={z:.3f} m"
        )
        return self.arm3d, self.trail3d, self.arm_side, self.hud


def animate(sequence=SEQUENCE, gif_path=None, substeps=SUBSTEPS):
    """Esegue la sequenza di gesti e disegna il braccio. Con gif_path salva una GIF, altrimenti apre la finestra.

    substeps = fotogrammi di interpolazione per comando: per sequenze lunghe (video) usa 1,
    altrimenti la GIF diventa enorme.
    """
    states = run_sequence(sequence)
    frames = interpolate(states, substeps)
    view = ArmView()

    anim = FuncAnimation(view.fig, lambda i: view.draw(*frames[i]), frames=len(frames),
                         interval=1000 / FPS, blit=False, repeat=False)

    if gif_path:
        anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=70)
        print("GIF salvata in", gif_path)
    else:
        plt.show()


def main():
    gif_path = sys.argv[sys.argv.index("--gif") + 1] if "--gif" in sys.argv else None
    animate(SEQUENCE, gif_path)


if __name__ == "__main__":
    main()
