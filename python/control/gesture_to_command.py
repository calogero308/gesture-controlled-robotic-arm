from arm_config import STEP


# Dizionario che mantiene la mappatura tra gesto-movimento
GESTURE_COMMANDS = {
    "like": (0, 0, 1), # SU
    "fist": (0, 0, -1), # GIU
    "peace": (0, 1, 0), # SINISTRA
    "rock": (0, -1, 0), # DESTRA
    "palm": (0, 0, 0), # STOP
}


# Funzione che da un determinato gesto restituisce un movimento (x, y, z)
# Inizialmente implementata come uno switch case ma sostituito da un dizionario
# per facilità di ampliazione futura
def gesture_to_command(gesture):
    # Preleva e restituisce se presente il movimento da effettuare
    return GESTURE_COMMANDS.get(gesture)


# Funzione che non fa altro che applicare la modifica di posizione al vettore attuale
def apply_command(position, command):
    if command is None:
        return position

    x, y, z = position
    dx, dy, dz = command
    new_position = (x + STEP * dx, y + STEP * dy, z + STEP * dz)
    return new_position 