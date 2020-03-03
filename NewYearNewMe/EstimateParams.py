import time
import numpy as np

import NewYearNewMe.VarationalInference as VI
from NewYearNewMe.ExperimentsNew.Classification import accuracy
from NewYearNewMe import PreProcess, EstimateAB

NUM_TOPICS_K = 8
VI_ITERATIONS = 20
EM_ITERATIONS = 50


def expectationMaximization(corpus, V):
    alpha, beta = EstimateAB.initAlphaBeta(V, NUM_TOPICS_K)
    phi = None
    gamma = None

    for i in range(EM_ITERATIONS):
        print("EM iteration:", i)
        phi = []
        gamma = []
        # E: VI
        timeTaken = time.time()
        for idx, doc in enumerate(corpus):
            # print("VI on document:", idx)
            N = len(doc)
            phiDoc, gammaDoc = VI.inference(alpha, beta, N, doc, NUM_TOPICS_K, VI_ITERATIONS)
            phi.append(phiDoc)
            gamma.append(gammaDoc)

        print("Time taken E:", time.time() - timeTaken)
        timeTaken = time.time()
        # M: EstimateAB
        alpha, beta = EstimateAB.maximizationStep(corpus, V, alpha, beta, phi, NUM_TOPICS_K)
        print("Time taken M:", time.time() - timeTaken)

    return alpha, beta, phi, gamma


def estimateParams(vocab, corpus):
    V = len(vocab)
    a, b, phi, gamma = expectationMaximization(corpus, V)
    return a, b, phi, gamma


if __name__ == '__main__':
    np.random.seed(13)
    vocab, corpus, topics = PreProcess.preProcess(2)
    '''
    a, b, phi, gamma = estimateParams(vocab, corpus)
    np.save("Parameters/a", a)
    np.save("Parameters/b", b)
    np.save("Parameters/phi", phi)
    np.save("Parameters/gamma", gamma)
    '''
    a, b, phi, gamma = estimateParams(vocab, corpus)
    print("--------")
    print(gamma)
    gamma = np.array(gamma)
    freqList = PreProcess.generateFreqList(corpus, len(vocab))
    print(freqList.shape)
    print(gamma.shape)

    acc = accuracy(freqList, topics)
    accLDA = accuracy(gamma, topics)
    print("ACCURACY")
    print("Word Features: " + str(acc))
    print("Topic Features (LDA): " + str(accLDA))
