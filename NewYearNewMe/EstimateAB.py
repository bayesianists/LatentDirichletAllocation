import numpy as np


# once for each document
def initBeta(V, K):
    return np.ones((K, V)) / V


def initAlphaBeta(V, K):
    alpha = np.array([1.0] * K)
    beta = initBeta(V, K)
    return alpha, beta


# Might be able to use numpy here =)
def betaIndex(i, j, phi, corpus):
    sum = 0
    M = len(corpus)
    for d in range(M):
        N = corpus[d].size
        for n in range(N):
            if corpus[d][n] == j:
                sum += phi[d][n][i] * corpus[d][n]
    return sum


def maximizationStep(corpus, V, alpha, beta, phi, K):
    # beta = np.zeros(K, V))
    for i in range(K):
        for j in range(V):
            beta[i][j] = betaIndex(i, j, phi, corpus)
