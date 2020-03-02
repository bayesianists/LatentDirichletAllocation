import NewYearNewMe.EstimateParams as EP
import numpy as np
import scipy.special as sp


def initializeDoc(alpha, N):
    phi = np.ones((N, EP.NUM_TOPICS_K)) / EP.NUM_TOPICS_K
    gamma = alpha + N / EP.NUM_TOPICS_K
    return phi, gamma


def inference(alpha, beta, N, doc):
    phi, gamma = initializeDoc(alpha, N)
    iterations = 0
    while EP.VI_ITERATIONS > iterations:
        for n in range(N):
            for i in range(EP.NUM_TOPICS_K):
                # BETA INDEXING MAY BE A PROBLEM - Smoothing!!!!!!!!
                phi[n][i] = beta[i][doc[n]] * np.exp(sp.digamma(gamma[i]))
            phi[n] /= np.sum(phi[n])
        gamma = alpha + np.sum(phi, axis=0)
        iterations += 1
    return phi, gamma
