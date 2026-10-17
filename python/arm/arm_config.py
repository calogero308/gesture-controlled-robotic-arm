"""
Configurazione centrale del braccio robotico RRR (3 gradi di libertà).

Tutti i parametri del braccio stanno qui: nessun'altra funzione deve
contenere numeri scritti a mano.

Unità: lunghezze in metri, angoli in radianti.

Geometria:
    H  -> altezza dell'asse della spalla rispetto al piano del tavolo
    L1 -> link 1: dalla spalla al gomito
    L2 -> link 2: dal gomito alla punta

Convenzioni degli angoli:
    theta1 -> rotazione della base attorno all'asse verticale;
              0 = verso l'asse x, positivo in senso antiorario visto dall'alto
    theta2 -> angolo del link 1 rispetto all'orizzontale (spalla);
              positivo verso l'alto
    theta3 -> angolo del link 2 rispetto al link 1 (gomito);
              negativo = il gomito si piega verso il basso
              (unica soluzione ammessa: "gomito in alto")
"""

import math

# --- Geometria (m) ---
# Dimensioni delle componenti del braccio tutto in metri
H = 0.09
L1 = 0.16
L2 = 0.14

# --- Limiti dei giunti (rad) ---
# Vincoli di rotazione per gli angoli
THETA1_MIN = math.radians(-90)
THETA1_MAX = math.radians(90)

THETA2_MIN = math.radians(0)
THETA2_MAX = math.radians(90)

THETA3_MIN = math.radians(-150)
# Non arriva a 0: a theta3 = 0 il braccio è teso e il Jacobiano è singolare
# (det J = L1*L2*sin(theta3)). Il margine di 5° evita soluzioni troppo sensibili.
THETA3_MAX = math.radians(-5)

# --- Margini di sicurezza (m) ---
# Altezza minima della punta (e del gomito) sopra il piano del tavolo
Z_MIN = 0.01
