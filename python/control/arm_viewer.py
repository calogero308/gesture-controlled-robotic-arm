"""
Finestra del braccio, in un processo a parte.

Legge da stdin una riga JSON per ogni nuovo stato (angoli, gesto, comando, stato) e ridisegna il braccio.
Lo avvia live_arm.py; si può provare anche a mano:
    echo '{"angles": [0.3, 0.8, -1.0], "gesture": "like", "command": [0,0,1], "status": "ok"}' | python3 arm_viewer.py

Il processo è separato perché OpenCV e matplotlib, nello stesso programma, possono litigare
sulle librerie grafiche (Qt); così ciascuno ha la sua finestra e il suo ciclo.

Opzione: --png file.png  salva l'ultima immagine alla chiusura.
"""

import json
import queue
import signal
import sys
import threading

import matplotlib.pyplot as plt

from arm_state import initial_state
from simulate import ArmView


def read_stdin(q):
    for line in sys.stdin:
        line = line.strip()
        if line:
            q.put(json.loads(line))
    q.put(None)                                   # fine dell'input: il programma principale ha chiuso


def main():
    # Ctrl+C va a tutto il gruppo di processi: il viewer lo ignora e si chiude quando finisce l'input
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None

    view = ArmView()
    state = initial_state()
    view.draw(state["angles"], state)

    q = queue.Queue()
    threading.Thread(target=read_stdin, args=(q,), daemon=True).start()

    plt.show(block=False)
    finished = False
    while plt.fignum_exists(view.fig.number) and not finished:
        new_state = None
        while True:                                # prende solo l'ultimo messaggio arrivato
            try:
                msg = q.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                finished = True
                break
            new_state = msg
        if new_state is not None:
            if new_state["command"] is not None:
                new_state["command"] = tuple(new_state["command"])
            state = new_state
            view.draw(state["angles"], state)
        plt.pause(0.03)

    if png:
        view.fig.savefig(png, dpi=100)


if __name__ == "__main__":
    main()
