import numpy as np
from itertools import chain


# Helper functions to generate rational function matrices
def generate_multi_indices(n, d):
    from itertools import chain
    return list(chain(*[list(r_multi_indices(n, _)) for _ in range(d + 1)]))

def generate_rational_function_matrix(numerator_degree, denominator_degree, n_components, dataset):
    G_indices = generate_multi_indices(n_components, numerator_degree)
    H_indices = generate_multi_indices(n_components, denominator_degree)

    G_matrix = np.zeros((len(dataset), len(G_indices)))
    H_matrix = np.zeros((len(dataset), len(H_indices)))

    for i, data_point in enumerate(dataset):
        G_matrix[i] = [np.prod([data_point[k] ** idx[k] for k in range(len(data_point))]) for idx in G_indices]
        H_matrix[i] = [np.prod([data_point[k] ** idx[k] for k in range(len(data_point))]) for idx in H_indices]

    return G_matrix, H_matrix

def r_multi_indices(n, d):
    if n == 1:
        yield (d,)
    else:
        for k in range(d + 1):
            for c in r_multi_indices(n - 1, k):
                yield (d - k, *c)
