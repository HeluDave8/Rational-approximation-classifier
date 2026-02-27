import numpy as np
from scipy.optimize import linprog

def solve_lp(z, G_matrix, H_matrix, y, delta=1e-5):
    y = np.asarray(y)  # Ensure y is a NumPy array
    length_dataset = len(y)
    num_alpha = G_matrix.shape[1]
    num_beta = H_matrix.shape[1]

    c = [1] + [0] * (num_alpha + num_beta)
    A_ub, b_ub = [], []

    for i in range(length_dataset):
        G_row = G_matrix[i]
        H_row = H_matrix[i]
        y_i = y[i]

        # First constraint
        A_ub.append([-1] + [-g for g in G_row] + [(y_i - z) * h for h in H_row])
        b_ub.append(0)

        # Second constraint
        A_ub.append([-1] + [g for g in G_row] + [-(y_i - z) * h for h in H_row])
        b_ub.append(0)

        # Positivity constraint
        A_ub.append([0] + [0] * num_alpha + [-h for h in H_row])
        b_ub.append(-delta)

    bounds = [(0, None)] + [(None, None)] * (num_alpha + num_beta)
    result = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    return result

def bisection_method(uL, uH, G_matrix, H_matrix, y, precision=1e-6):
    y = np.asarray(y)  # Ensure y is a NumPy array

    while (uH - uL) > precision:
        z = (uH + uL) / 2
        result = solve_lp(z, G_matrix, H_matrix, y)

        if result.success:
            uH = z
        else:
            uL = z

    return uH



