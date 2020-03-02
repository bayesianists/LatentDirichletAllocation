#from LatentDirichletAllocation.NewYearNewMe import PreProcess
import NewYearNewMe.PreProcess as PP

NUM_TOPICS_K = 3
VI_ITERATIONS = 3
EM_ITERATIONS = 5


def expectationMaximization():
    for i in range(EM_ITERATIONS):
        pass
        # E: VI
        # M: EstimateAB
    pass


def estimateParams():
    expectationMaximization()


if __name__ == '__main__':
    vocab, corpus = PreProcess.preProcess()
    estimateParams()
