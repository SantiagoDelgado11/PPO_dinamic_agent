import numpy as np

# --- Funciones Base de Hadamard (antes en utils.py) ---


def hadamard_matrix(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard_matrix(n // 2)
        return np.block([[h, h], [h, -h]])
