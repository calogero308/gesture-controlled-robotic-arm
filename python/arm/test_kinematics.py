"""
Test della cinematica del braccio RRR.

1. forward_kinematics su tre configurazioni di cui si conosce il risultato.
2. Giro completo: angoli casuali nei limiti -> punta -> inverse_kinematics -> stessi angoli.
3. Punti non raggiungibili (o non ammessi): inverse_kinematics deve restituire None.

Esecuzione (da python/arm):  python3 test_kinematics.py
"""

import math
import random

from arm_config import (THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX,
                        THETA3_MIN, THETA3_MAX, Z_MIN, H, L1, L2)
from kinematics import forward_kinematics, inverse_kinematics

TOL = 1e-6          # tolleranza sui confronti (m o rad)
N_CASI = 1000


def test_forward_noti():
    """Tre configurazioni con risultato calcolato a mano."""
    casi = [
        ((0, 0, 0), (L1 + L2, 0, H)),                      # braccio teso in orizzontale
        ((0, math.pi / 2, 0), (0, 0, H + L1 + L2)),        # braccio teso in verticale
        ((0, 0, -math.pi / 2), (L1, 0, H - L2)),           # avambraccio verso il basso
    ]
    for angoli, atteso in casi:
        ottenuto = forward_kinematics(*angoli)
        for o, a in zip(ottenuto, atteso):
            assert math.isclose(o, a, abs_tol=TOL), (angoli, ottenuto, atteso)
    print("forward_kinematics: 3/3 casi noti corretti")


def test_giro_completo():
    """angoli -> punta -> angoli, su punti casuali nello spazio di lavoro."""
    random.seed(0)
    provati = saltati = falliti = 0
    for _ in range(N_CASI):
        t1 = random.uniform(THETA1_MIN, THETA1_MAX)
        t2 = random.uniform(THETA2_MIN, THETA2_MAX)
        t3 = random.uniform(THETA3_MIN, THETA3_MAX)
        x, y, z = forward_kinematics(t1, t2, t3)

        if z < Z_MIN:           # punta troppo bassa: l'inversa deve rifiutarla
            saltati += 1
            assert inverse_kinematics(x, y, z) is None, (t1, t2, t3, x, y, z)
            continue

        provati += 1
        risultato = inverse_kinematics(x, y, z)
        if risultato is None:
            falliti += 1
            print("  FALLITO (None):", t1, t2, t3)
            continue
        for ottenuto, atteso in zip(risultato, (t1, t2, t3)):
            if not math.isclose(ottenuto, atteso, abs_tol=TOL):
                falliti += 1
                print("  FALLITO (angoli diversi):", (t1, t2, t3), "->", risultato)
                break

    print(f"giro completo: provati {provati}, saltati (z < Z_MIN) {saltati}, falliti {falliti}")
    assert falliti == 0


def test_punti_non_raggiungibili():
    """Punti che inverse_kinematics deve rifiutare."""
    casi = {
        "troppo lontano": (1.0, 0.0, 0.09),
        "sull'asse della base": (0.0, 0.0, 0.09),
        "dietro la base": (-0.15, 0.0, 0.15),
        "sotto il tavolo": (0.15, 0.0, -0.05),
    }
    for nome, punto in casi.items():
        assert inverse_kinematics(*punto) is None, (nome, punto)
    print(f"punti non raggiungibili: {len(casi)}/{len(casi)} rifiutati")


if __name__ == "__main__":
    test_forward_noti()
    test_giro_completo()
    test_punti_non_raggiungibili()
    print("TUTTI I TEST PASSATI")