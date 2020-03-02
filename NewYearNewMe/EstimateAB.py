import numpy as np
import scipy.special as sp


# once for each document
def initBeta(V, K):
    return np.ones((K, V)) / V


def initAlphaBeta(V, K):
    alpha = np.array([1.0] * K)
    beta = initBeta(V, K)
    return alpha, beta


# Might be able to use numpy here =)
def betaIndex(i, j, phi, corpus):
    betaSum = 0
    M = len(corpus)
    for d in range(M):
        N = corpus[d].size
        for n in range(N):
            if corpus[d][n] == j:
                betaSum += phi[d][n][i]

    return betaSum


# Written in the context of a single document
# Uses natural logarithm
# Maybe try some other logarithm
def gradient_k(alpha_k, alpha_sum, N):
    return N * (sp.digamma(alpha_sum) - sp.digamma(alpha_k) + np.log(alpha_k / alpha_sum))


def hessian_inverse_gradient(alpha, M, K):
    alphaSum = np.sum(alpha)
    z = M * sp.polygamma(1, alphaSum)

    Q = np.zeros(K)
    gradient = np.zeros(K)

    if alphaSum == 0:
        print("alpha:", alpha)

    # TODO: Check what j is in the formulas
    for k in range(K):
        Q[k] = -M * sp.polygamma(1, alpha[k])
        gradient[k] = gradient_k(alpha[k], alphaSum, M)

    inv_Q = np.reciprocal(Q)
    b = np.sum(gradient * inv_Q) / (np.sum(inv_Q) + (1 / z))
    # print("z:", z)
    # print("sum of inv_Q:", np.sum(inv_Q))

    return (gradient - b) / Q


def maximizationStep(corpus, V, alpha, beta, phi, K):
    import time
    timeTaken = time.time()

    for i in range(K):
        for j in range(V):
            beta[i][j] = betaIndex(i, j, phi, corpus)

    print("BetaTime:", time.time() - timeTaken)

    alpha = alpha - hessian_inverse_gradient(alpha, len(corpus), K)
    return alpha, beta
