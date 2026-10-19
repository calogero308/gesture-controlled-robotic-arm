"""
Stato del braccio e passo di controllo, senza grafica.

step() prende lo stato attuale e un gesto stabile e restituisce il nuovo stato:
gesto -> comando -> nuova posizione della punta -> cinematica inversa -> angoli.
Se il punto richiesto non è raggiungibile, il braccio resta dov'è.
"""

import math
import sys
from pathlib import Path

# il modulo del braccio sta in python/arm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arm"))

from kinematics import forward_kinematics, inverse_kinematics
from gesture_to_command import gesture_to_command, apply_command

HOME_ANGLES = (0.0, math.radians(45), math.radians(-90))   # posizione di partenza


def initial_state(home_angles=HOME_ANGLES):
    return dict(gesture=None, command=None, pos=forward_kinematics(*home_angles),
                angles=home_angles, status="posizione iniziale")


def step(state, gesture):
    """Applica un gesto allo stato e restituisce il nuovo stato (quello vecchio non viene toccato)."""
    pos, angles = state["pos"], state["angles"]
    command = gesture_to_command(gesture)

    if command is None:
        status = "nessun comando"
    elif command == (0, 0, 0):
        status = "stop"
    else:
        target = apply_command(pos, command)
        solution = inverse_kinematics(*target)
        if solution is None:
            status = "BLOCCATO: punto non raggiungibile"
        else:
            pos, angles, status = target, solution, "ok"

    return dict(gesture=gesture, command=command, pos=pos, angles=angles, status=status)
