import numpy as np
import scipy.special as sp


def initializeDoc(alpha, N, K):
    phi = np.ones((N, K)) / K
    gamma = alpha + N / K
    return phi, gamma


# Context of one document, phi is 2D-Matrix and gamma is vector
def inference(alpha, beta, N, doc, K, numIterations):
    phi, gamma = initializeDoc(alpha, N, K)
    t = 0
    while t < numIterations:
        gammaSumDigamma = sp.digamma(np.sum(gamma))
        for n in range(N):
            word = doc[n]
            # betaVec = beta[:, word]
            # expVec = np.exp(sp.digamma(gamma))

            for i in range(K):
                # BETA INDEXING MAY BE A PROBLEM - Smoothing!!!!!!!!
                phi[n][i] = beta[i][word] * (np.exp(sp.digamma(gamma[i])) - gammaSumDigamma) + 1e-20
                if phi[n][i] == 0:
                    print("Beta:", beta[i][doc[n]])
                    print("DiGamma:", np.exp(sp.digamma(gamma[i])))
            # phi[n] /= np.sum(phi[n])
        phi = phi / np.expand_dims(np.sum(phi, axis=1), 1)  # (153, ) => (153, 1)
        gamma = alpha + np.sum(phi, axis=0)
        t += 1
    return phi, gamma
