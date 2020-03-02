import NewYearNewMe.EstimateParams as EP
import numpy as np


# once for each document
def initBeta(V):
    return np.ones((EP.NUM_TOPICS_K, V)) / V


def initAlphaBeta(V):
    alpha = np.array([1.0] * EP.NUM_TOPICS_K)
    beta = initBeta(V)
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


def maximizationStep(corpus, V, alpha, beta, phi):
    # beta = np.zeros((EP.NUM_TOPICS_K, V))
    for i in range(EP.NUM_TOPICS_K):
        for j in range(V):
            beta[i][j] = betaIndex(i, j, phi, corpus)
