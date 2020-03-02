import NewYearNewMe.EstimateParams as EP
import numpy as np


def beta_index(i, j, phi, corpus):
    sum = 0
    for d in range(M):
        N = corpus[d].size
        for n in range(N):
            if corpus[d][n] == j:
                sum += phi[d][n][i] * corpus[d][n]
    return sum


def maximization_step(corpus):
    beta = np.zeros((EP.NUM_TOPICS_K, EP.VOCAB_SIZE))
    for i in range(EP.NUM_TOPICS_K):
        for j in range(EP.VOCAB_SIZE):
            beta[i][j] = beta_index(i, j, phi, corpus)
