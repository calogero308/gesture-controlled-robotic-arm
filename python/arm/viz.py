"""
Geometria e disegno del braccio RRR (solo visualizzazione).

joint_positions() restituisce i punti da disegnare: base, spalla, gomito, punta.
La punta coincide con forward_kinematics(); il gomito serve solo qui.
"""

import math

from arm_config import H, L1, L2


def joint_positions(theta1, theta2, theta3):
    """Base, spalla, gomito e punta come terne (x, y, z) in metri."""
    c1, s1 = math.cos(theta1), math.sin(theta1)

    r_gomito = L1 * math.cos(theta2)
    z_gomito = H + L1 * math.sin(theta2)

    r_punta = r_gomito + L2 * math.cos(theta2 + theta3)
    z_punta = z_gomito + L2 * math.sin(theta2 + theta3)

    return [
        (0.0, 0.0, 0.0),                          # base, sul tavolo
        (0.0, 0.0, H),                            # spalla
        (r_gomito * c1, r_gomito * s1, z_gomito), # gomito
        (r_punta * c1, r_punta * s1, z_punta),    # punta
    ]
