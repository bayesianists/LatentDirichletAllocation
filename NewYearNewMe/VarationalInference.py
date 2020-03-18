import numpy as np
import scipy.special as sp

#Youssef
def initializeDoc(alpha, N, K):
    phi = np.ones((N, K)) / K
    gamma = alpha + N / K
    return phi, gamma

#Youssef & Joey
# Context of one document, phi is 2D-Matrix and gamma is vector
def inference(alpha, beta, N, doc, K, numIterations, gammaPrev):
    phi, gamma = initializeDoc(alpha, N, K)
    # if gammaPrev is not None:
    #     gamma = gammaPrev

    t = 0
    gammaPrev = gamma
    phiPrev = phi
    while t < numIterations:
        gammaSumDigamma = sp.digamma(np.sum(gamma))
        # phi = (M, N, K) => (N, K)
        # doc = (1, N)
        # beta = (K, V)
        # gamma = (M, K) => (1, K)
        words = doc[np.arange(N)]
        # print(words)
        expVec = np.exp(sp.digamma(gamma) - gammaSumDigamma)
        if len(words) > 0:
            betaMat = beta[:, words]  # (K, N)
            phi = betaMat.T * expVec + 1e-100  # (N, K) * (1, K) + (1, 1) = (N, K) + (N, K) = (N, K)

            # for n in range(N):
                # word = doc[n]
                # betaVec = beta[:, word]
                # expVec = np.exp(sp.digamma(gamma) - gammaSumDigamma)
                # phi[n] = betaVec * expVec

                # for i in range(K):
                    # BETA INDEXING MAY BE A PROBLEM - Smoothing!!!!!!!!
                    # phi[n][i] = beta[i][word] * (np.exp(sp.digamma(gamma[i])) - gammaSumDigamma) + 1e-20
                    # if phi[n][i] == 0:
                        # print("Beta:", beta[i][doc[n]])
                        # print("DiGamma:", np.exp(sp.digamma(gamma[i])))
                # phi[n] /= np.sum(phi[n])
            # phiMat = phiMat / np.expand_dims(np.sum(phiMat, axis=1), 1)  # (153, ) => (153, 1)
            phi = phi / np.expand_dims(np.sum(phi, axis=1), 1)  # (153, ) => (153, 1)
            gamma = alpha + np.sum(phi, axis=0)
            t += 1

        '''
        diffGamma = np.sum(np.abs(gamma - gammaPrev))
        diffPhi = np.sum(np.abs(phi - phiPrev))
        print(t, diffGamma, diffPhi)
        gammaPrev = gamma
        phiPrev = phi
        '''
    return phi, gamma
