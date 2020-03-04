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


'''
# Might be able to use numpy here =)
def betaIndex(i, j, phi, corpus):
    M = len(corpus)
    betaSum = 0

    #for d in range(M):
    #     N = corpus[d].size
    #     for n in range(N):
    #         if corpus[d][n] == j:
    #             betaSum += phi[d][n][i]

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
'''


# Written in the context of a single document
# Uses natural logarithm
# Maybe try some other logarithm
def gradient_k(alpha_k, alpha_sum, M, log_exp_phi):
    if alpha_sum == 0:
        print ("ALPHASUM IS 0")
    if alpha_k == 0:
        print ("ALPHA_K IS 0")
    if log_exp_phi == 0:
        print ("EXP_PHI IS 0")
    return M * (sp.digamma(alpha_sum) - sp.digamma(alpha_k) + log_exp_phi)  # np.log(alpha_k / alpha_sum))


def hessian_inverse_gradient(alpha, M, K, gamma):
    alphaSum = np.sum(alpha)
    # gammaSum = np.sum(gamma)
    z = M * sp.polygamma(1, alphaSum)  # c

    Q = np.zeros(K)
    gradient = np.zeros(K)

    # Expected dirichlet
    gammaSumDigamma = sp.digamma(np.sum(gamma, axis=1))
    gammaSumDigamma = np.expand_dims(gammaSumDigamma, 1)  # (M, 1)
    diffVector = sp.digamma(gamma) - gammaSumDigamma  # (M, K)
    meanVector = np.mean(diffVector, axis=0)  # (1, K)

    log_expected_phi = meanVector  # (1, K)

    if alphaSum == 0:
        print("alpha:", alpha)

    # K was M
    for k in range(K):
        Q[k] = -M * sp.polygamma(1, alpha[k])
        gradient[k] = gradient_k(alpha[k], alphaSum, M, log_expected_phi[k])

    inv_Q = np.reciprocal(Q)
    if z == 0:
        print("z:", z)

    # print("*")
    # print (inv_Q)
    b = np.sum(gradient * inv_Q)
    b /= (np.sum(inv_Q) + (1 / z))
    # print("z:", z)
    # print("sum of inv_Q:", np.sum(inv_Q))

    return (gradient - b) / Q


def maximizationStep(corpus, V, alpha, beta, phi, K, gamma):

    # Paper uses subtraction here, but a student in slack derived that this is incorrect and should be an addition
    # for i in range(K):
    alpha = alpha + hessian_inverse_gradient(alpha, len(corpus), K, gamma)

    # import time
    # timeTaken = time.time()
    M = len(corpus)
    for i in range(K):
        for j in range(V):

            betaSum = 0
            # idxs = np.array([np.where(corpus[d] == j)[0] for d in range(M)])
            # betaSums = np.sum(phi[:][:, i])
            # assert False

            for d in range(M):
                idxs = np.where(corpus[d] == j)[0]
                # print(idxs)
                if len(idxs) > 0:
                    betaSum += np.sum(phi[d][:, i][idxs])

            # assert False
            beta[i][j] = betaSum

    # print("BetaTime:", time.time() - timeTaken)

    return alpha, beta


def normalizeMatrix(mat):
    return mat / np.expand_dims(np.sum(mat, axis=1), 1)


def getTFIDF(beta, K):
    prod = np.prod(beta, axis=0)
    products = np.power(prod, (1.0 / K))
    return beta * np.log(beta / products)


def getMostPopularWordsPerTopic(beta, K, vocab):
    betaNorm = normalizeMatrix(beta)
    tfidf = getTFIDF(betaNorm, K)
    # tfidf = betaNorm
    V = len(tfidf[0])

    tuples = []
    for i in range(K):
        tuples.append([(tfidf[i][j], vocab[j]) for j in range(V)])

    # sortedTuples = np.sort(tuples, key=lambda x: x[0], axis=1)
    for i in range(K):
        tuples[i] = sorted(tuples[i], key=lambda x: float(x[0]))
        tuples[i] = np.array(tuples[i])

    for i in range(K):
        print(tuples[i][-5:, 1])
        # print(tuples[i][-10:, 0])


if __name__ == '__main__':
    pass
    # testArr = np.ones((5, 3)) * np.arange(1, 4) * 72
    # testArr[0][1] += 100
    # testArr = normalizeMatrix(testArr)
    # print(testArr)
    # tfidf = getTFIDF(testArr, len(testArr))
    # print(tfidf)
