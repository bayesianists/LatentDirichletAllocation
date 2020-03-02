import NewYearNewMe.VarationalInference as VI
from NewYearNewMe import PreProcess, EstimateAB

NUM_TOPICS_K = 3
VI_ITERATIONS = 3
EM_ITERATIONS = 5


def expectationMaximization(corpus, V):
    alpha, beta = EstimateAB.initAlphaBeta(V)
    phi = None
    gamma = None

    for i in range(EM_ITERATIONS):
        phi = []
        gamma = []
        # E: VI
        for idx, doc in enumerate(corpus):
            N = len(doc)
            phiDoc, gammaDoc = VI.inference(alpha, beta, N, doc)
            phi.append(phiDoc)
            gamma.append(gammaDoc)

        # M: EstimateAB
        EstimateAB.maximizationStep(corpus, V, alpha, beta, phi)

    return alpha, beta, phi, gamma


def estimateParams(vocab, corpus):
    V = len(vocab)
    a, b, phi, gamma = expectationMaximization(corpus, V)
    return a, b, phi, gamma


if __name__ == '__main__':
    vocab, corpus = PreProcess.preProcess(1)
    estimateParams(vocab, corpus)
