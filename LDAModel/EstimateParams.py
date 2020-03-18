import time
import numpy as np

import LDAModel.VarationalInference as VI
from LDAModel.ExperimentsNew.Classification import accuracy
from LDAModel import PreProcess, EstimateAB

# Everyone worked on this part

NUM_TOPICS_K = 25
VI_ITERATIONS = 250
EM_ITERATIONS = 50

# Everyone worked on this part
def expectationMaximization(corpus, V):
    alpha, beta = EstimateAB.initAlphaBeta(V, NUM_TOPICS_K)
    phi = None
    gamma = []
    phiPrev = None
    gammaPrev = None

    for i in range(EM_ITERATIONS):
        print("EM iteration:", i)
        phi = []
        gamma = [None for _ in range(len(corpus))]
        # E: VI
        timeTaken = time.time()
        # print(beta)
        for idx, doc in enumerate(corpus):
            N = len(doc)
            phiDoc, gammaDoc = VI.inference(alpha, beta, N, doc, NUM_TOPICS_K, VI_ITERATIONS, gamma[idx])
            phi.append(phiDoc)
            gamma[idx] = gammaDoc


        print("Time taken E:", time.time() - timeTaken)
        timeTaken = time.time()
        alpha, beta = EstimateAB.maximizationStep(corpus, V, alpha, beta, phi, NUM_TOPICS_K, gamma)
        print("Time taken M:", time.time() - timeTaken)

        EstimateAB.getMostPopularWordsPerTopic(beta, NUM_TOPICS_K, vocab)

        '''
        if phiPrev is not None:
            diffGamma = np.sum(np.abs(gamma - gammaPrev))
            phiSum = 0
            for j in range(len(corpus)):
                phiSum += np.sum(np.abs(phi[j] - phiPrev[j]))
            print(i, diffGamma, phiSum)

        phiPrev = phi
        gammaPrev = np.copy(gamma)
        print(alpha)
        print(np.random.dirichlet(alpha, 10))
        '''

    return alpha, beta, phi, gamma

def estimateParams(vocab, corpus):
    V = len(vocab)
    a, b, phi, gamma = expectationMaximization(corpus, V)
    return a, b, phi, gamma

if __name__ == '__main__':
    vocab, corpus, topics = PreProcess.preProcess(numFilesToImport=1, loadFromFile=False, reuters=False)

    # only estimate params if this is false, otherwise load old params
    LOAD_PARAMS = False

    if LOAD_PARAMS:
        print("Loading parameters!")
        a = np.load("Parameters/a.npy")
        b = np.load("Parameters/b.npy")
        phi = np.load("Parameters/phi.npy", allow_pickle=True)
        gamma = np.load("Parameters/gamma.npy")
    else:
        print("Estimating parameters!")
        a, b, phi, gamma = estimateParams(vocab, corpus)
        np.save("Parameters/a", a)
        np.save("Parameters/b", b)
        np.save("Parameters/phi", phi)
        np.save("Parameters/gamma", gamma)

    # Data and parameters are ready to be used for experiments beneath!

    print("--------")
    EstimateAB.getMostPopularWordsPerTopic(b, NUM_TOPICS_K, vocab)
    gamma = np.array(gamma)

