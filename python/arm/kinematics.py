from arm_config import H, L1, L2
from arm_config import THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX, THETA3_MIN, THETA3_MAX, Z_MIN
import math


# Funzione che controlla se gli angoli rispettano i vincoli imposti
def is_pose_valid(theta1, theta2, theta3, z):

    # Controllo sui vincoli per gli angoli
    return (THETA1_MIN <= theta1 <= THETA1_MAX
            and THETA2_MIN <= theta2 <= THETA2_MAX
            and THETA3_MIN <= theta3 <= THETA3_MAX
            and z >= Z_MIN)



# Funzione per il calcolo della cinematica diretta
def forward_kinematics(theta1, theta2, theta3):

    """
    Cinematica diretta: dagli angoli (rad) alla posizione della punta (m).
    Restituisce (x, y, z). Non controlla i limiti: usare within_limits().
    """

    # Calcolo della cinematica diretta
    # r è la distanza orizzontale della punta dall’asse della base
    r = L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)
    
    # Posizione sull'asse delle z
    z = H + L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3)

    # Posizione sull'asse delle x
    x = r * math.cos(theta1)

    # Posizione sull'asse delle y
    y = r * math.sin(theta1)

    return x, y, z



# Funzione per il calcolo della cinematica inversa
def inverse_kinematics(x, y, z):

    # Calcolo del primo angolo theta1
    theta1 = math.atan2(y, x)

    # Calcolo di r distanza orizzontale della punta dall’asse della base
    r = math.sqrt(x**2 + y**2)

    # Calcolo di zz ovvero z - h
    zz = z - H

    # Calcolo di D 
    D = (r**2 + zz**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # Controllo che il valore D sia minore in valore assoluto a 1 (Punto fuori portata)
    if abs(D) > 1: 
        return None
    
    # Calcolo del terzo angolo theta3
    theta3 = -math.acos(D)

    # Calcolo del secondo angolo theta2 (somma dell'angolo alpha e beta)
    theta2 = math.atan2(zz, r) + math.atan2(L2 * abs(math.sin(theta3)), L1 + L2 * math.cos(theta3))

    # Chiamata alla funzione per il controllo dei vincoli
    if not is_pose_valid(theta1, theta2, theta3, z):
        return None

    return theta1, theta2, theta3

    
    