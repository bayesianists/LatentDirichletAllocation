#from LatentDirichletAllocation.NewYearNewMe import PreProcess
import NewYearNewMe.PreProcess as PP
import NewYearNewMe.VarationalInference as VI
NUM_TOPICS_K = 3
VI_ITERATIONS = 3
EM_ITERATIONS = 5
VOCAB_SIZE = 1000

def expectationMaximization():
    for i in range(EM_ITERATIONS):
        pass
        # E: VI
        phi, gamma = VI.inference(alpha,beta, N, doc)
        # M: EstimateAB
        EM.maximization(phi, gamma)
    pass


def estimateParams():
    expectationMaximization()


if __name__ == '__main__':
    vocab, corpus = PreProcess.preProcess(1)
    estimateParams()
