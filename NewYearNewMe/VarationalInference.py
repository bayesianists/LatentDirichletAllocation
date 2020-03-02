import numpy as np
import scipy.special as sp


def initializeDoc(alpha, N, K):
    phi = np.ones((N, K)) / K
    gamma = alpha + N / K
    return phi, gamma


def inference(alpha, beta, N, doc, K, numIterations):
    phi, gamma = initializeDoc(alpha, N, K)
    t = 0
    while t < numIterations:
        for n in range(N):
            for i in range(K):
                # BETA INDEXING MAY BE A PROBLEM - Smoothing!!!!!!!!
                phi[n][i] = beta[i][doc[n]] * np.exp(sp.digamma(gamma[i]))
            phi[n] /= np.sum(phi[n])
        gamma = alpha + np.sum(phi, axis=0)
        t += 1
    return phi, gamma
