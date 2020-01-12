import numpy as np
import TextExtraction
import scipy.special as sp

"""Contains the implemented Variational Inference algorithm for LDA and associated functions"""

NUMBER_OF_TOPICS_K = 17
VI_ITERATIONS = 10000

phi = 0
gamma = 0

# Pseudocode for VI, LDA
""" (1) initialize φ0 ni := 1/k for all i and n 
    (2) initialize γi := αi+N/k for all i 
    (3) repeat 
    (4)     for n = 1 to N 
    (5)         for i = 1 to k 
    (6)             φt+1 ni := βiwn exp(Ψ(γt i)) 
    (7)         normalize φt+1 n to sum to 1. 
    (8)     γt+1 := α+∑N n=1φt+1 n 
    (9) until convergence"""


def variational_Inference(N, K, alfa, beta):
    """
    Runs the VI algorithm according to the pseudocode above.
    :param
    :return
    """

    phi = init_phi(N, K)
    gamma = init_gamma(K, alfa, N)

    has_not_converged = True
    t = 0

    while has_not_converged:
        for n in range(N):
            for i in range(K):
                phi = beta[i, n] * np.exp(sp.digamma(gamma[i]))
            phi[n] = normalize_row(phi)
        gamma = sum_columns_alfa(alfa, phi)
        has_not_converged = convergence_check(t)


def sum_columns_alfa(alfa, phi):
    return alfa + np.sum(phi, axis=1)


def convergence_check(t):
    t += 1
    if t == VI_ITERATIONS:
        return False
    else:
        return True


def normalize_row(row):
    """
    Takes in a matrix row and normalizes it.
    :param A row of a matrix
    :return The normalized matrix
    """
    row_sum = np.sum(row)
    return row / row_sum


def init_phi(n, i):
    phi0 = np.zeros((n, i))
    return phi0 + 1 / NUMBER_OF_TOPICS_K


def init_gamma(i, alfa, number_of_words):
    return alfa + number_of_words
