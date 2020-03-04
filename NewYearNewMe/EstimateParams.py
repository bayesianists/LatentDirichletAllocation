import time
import numpy as np

import NewYearNewMe.VarationalInference as VI
from NewYearNewMe.ExperimentsNew.Classification import accuracy
from NewYearNewMe import PreProcess, EstimateAB

NUM_TOPICS_K = 8
VI_ITERATIONS = 20
EM_ITERATIONS = 10


def expectationMaximization(corpus, V):
    alpha, beta = EstimateAB.initAlphaBeta(V, NUM_TOPICS_K)
    phi = None
    gamma = None

    for i in range(EM_ITERATIONS):
        #print("EM iteration:", i)
        phi = []
        gamma = []
        # E: VI
        timeTaken = time.time()
        # print(beta)
        for idx, doc in enumerate(corpus):
            # print("VI on document:", idx)
            N = len(doc)
            phiDoc, gammaDoc = VI.inference(alpha, beta, N, doc, NUM_TOPICS_K, VI_ITERATIONS)
            phi.append(phiDoc)
            gamma.append(gammaDoc)

            # print(phiDoc)
        gamma = np.array(gamma)
        #print("Time taken E:", time.time() - timeTaken)
        timeTaken = time.time()
        # M: EstimateAB
        alpha, beta = EstimateAB.maximizationStep(corpus, V, alpha, beta, phi, NUM_TOPICS_K,gamma)
        #print("Time taken M:", time.time() - timeTaken)
        accLDA = accuracy(gamma, topics)

        print("Topic Features (LDA): " + str(accLDA))
    return alpha, beta, phi, gamma


def estimateParams(vocab, corpus):
    V = len(vocab)
    a, b, phi, gamma = expectationMaximization(corpus, V)
    return a, b, phi, gamma


if __name__ == '__main__':
    np.random.seed(13)
    vocab, corpus, topics = PreProcess.preProcess(numFilesToImport=1, loadFromFile=False)
    print("ACCURACY")
    freqList = PreProcess.generateFreqList(corpus, len(vocab))
    acc = accuracy(freqList, topics)
    print("Word Features: " + str(acc))
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

    gamma = np.array(gamma)


