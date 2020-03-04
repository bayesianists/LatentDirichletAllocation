import numpy as np
import scipy.special as sp


# once for each document
def initBeta(V, K):
    eta = np.ones(V)
    beta = np.random.dirichlet(eta, K)
    return beta
    # return np.ones((K, V)) / V


def initAlphaBeta(V, K):
    alpha = np.array([1.0] * K)
    beta = initBeta(V, K)
    return alpha, beta


# Might be able to use numpy here =)
def betaIndex(i, j, phi, corpus):
    M = len(corpus)
    betaSum = 0

    '''
    for d in range(M):
        N = corpus[d].size
        for n in range(N):
            if corpus[d][n] == j:
                betaSum += phi[d][n][i]
    '''

    for d in range(M):
        idxs = np.where(corpus[d] == j)[0]
        # print(idxs)
        # print(np.where(corpus[d] == j))
        # print("len:", len(idxs))
        if len(idxs) > 0:
            # if len(idxs) == 1:
            # idxs = idxs[0]
            # print(phi[d].shape)
            # print(phi[d][:, i].shape)
            # print(phi[d][:, i][idxs].shape)
            # print("Getting phi[" + str(d) + "][" + str(idxs) + "][" + str(i) + "]:", phi[d, idxs, i])
            # print(phi[d][:, i][idxs].shape)
            betaSum += np.sum(phi[d][:, i][idxs])

    # print(betaSum1)
    # print(betaSum2)
    # assert betaSum1 == betaSum2
    # print(np.where(np.asarray(corpus) == j))
    if np.isnan(betaSum):
        print(betaSum)
        assert False

    return betaSum


# Written in the context of a single document
# Uses natural logarithm
# Maybe try some other logarithm
def gradient_k(alpha_k, alpha_sum, N, expected_phi):
    return N * (sp.digamma(alpha_sum) - sp.digamma(alpha_k) + np.log(expected_phi))  # np.log(alpha_k / alpha_sum))


def hessian_inverse_gradient(alpha, M, K, gamma):
    alphaSum = np.sum(alpha)
    gammaSum = np.sum(gamma)
    z = M * sp.polygamma(1, alphaSum)

    Q = np.zeros(K)
    gradient = np.zeros(K)

    expected_phi = np.exp(np.mean(sp.digamma(gamma) - np.expand_dims(sp.digamma(np.sum(gamma, axis=1)), 1), axis=0))


    if alphaSum == 0:
        print("alpha:", alpha)

    # TODO: Check what j is in the formulas
    for k in range(K):
        Q[k] = -M * sp.polygamma(1, alpha[k])
        gradient[k] = gradient_k(alpha[k], alphaSum, M, expected_phi[k])

    inv_Q = np.reciprocal(Q)
    b = np.sum(gradient * inv_Q) / (np.sum(inv_Q) + (1 / z))
    # print("z:", z)
    # print("sum of inv_Q:", np.sum(inv_Q))

    return (gradient - b) / Q


def maximizationStep(corpus, V, alpha, beta, phi, K, gamma):
    import time
    timeTaken = time.time()
    for i in range(K):
        for j in range(V):
            beta[i][j] = betaIndex(i, j, phi, corpus)

    #print("BetaTime:", time.time() - timeTaken)

    # Paper uses subtraction here, but a student in slack derived that this is incorrect and should be an addition
    alpha = alpha + hessian_inverse_gradient(alpha, len(corpus), K, gamma)
    return alpha, beta


if __name__ == '__main__':
    testList = np.array(["ab", "lol", "sup"])
    print(testList[0])
    print(testList[[0, 1]])
    print(testList[[0]])
