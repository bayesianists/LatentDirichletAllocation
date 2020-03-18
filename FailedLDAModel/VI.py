import numpy as np
# import TextExtraction
from FailedLDAModel.TextExtraction import get_vocab, make_corpus, one_hot_to_string
import scipy.special as sp

"""Contains the implemented Variational Inference algorithm for LDA and associated functions"""

NUM_TOPICS_K = 3
VI_ITERATIONS = 3
EM_ITERATIONS = 5

# Pseudocode for VI, LDA
""" (1) initialize φ0 ni := 1/k for all i and n 
    (2) initialize γi := αi+N/k for all i 
    (3) repeat 
    (4)     for n = 1 to N 
    (5)         for i = 1 to k 
    (6)             φt+1 ni := βiwn exp(Ψ(γt i)) 
    (7)         normalize φt+1 n to sum to 1. 
    (8)     γt+1 := α+∑N n=1φt+1 n 
    (9) until convergence"""


# Called once per doc
def variational_inference(N, alpha, beta, doc, phi=None, gamma=None):
    """
    Runs the VI algorithm according to the pseudocode above.
    :param
    :return
    """

    phi, gamma = init_phi_gamma(N, alpha, phi, gamma)

    has_converged = False
    t = 0

    while not has_converged:
        for n in range(N):
            for i in range(NUM_TOPICS_K):
                nonZeroes = np.nonzero(doc[n])
                betaVal = 0.00001  # joey smoothing
                if np.size(nonZeroes) > 0:
                    betaVal = beta[i][nonZeroes[0]]

                phi[n][i] = betaVal * np.exp(sp.digamma(gamma[i]))
            phi[n] /= np.sum(phi[n])
        gamma = alpha + np.sum(phi, axis=0)

        t += 1
        has_converged = t == VI_ITERATIONS

    return phi, gamma


# for one document
def init_phi(N, phi=None):
    if phi is None:
        phi = np.empty((N, NUM_TOPICS_K))

    phi.fill(1 / NUM_TOPICS_K)
    return phi


def init_gamma(alpha, N, gamma=None):
    if gamma is None:
        return alpha + N / NUM_TOPICS_K
    else:
        for i in range(NUM_TOPICS_K):
            gamma.fill(alpha[i] + N / NUM_TOPICS_K)
        return gamma


# once for each document
def init_beta(V):
    beta = []
    for i in range(NUM_TOPICS_K):
        beta.append(np.array([1 / V] * V))
    return np.array(beta)


def init_alpha_beta(V):
    alpha = np.array([1.0] * NUM_TOPICS_K)
    beta = init_beta(V)
    return alpha, beta


def init_phi_gamma(N, alpha, phi=None, gamma=None):
    phi = init_phi(N, phi)
    gamma = init_gamma(alpha, N, gamma)
    return phi, gamma


# Written in the context of a single document
# Uses natural logarithm
# Maybe try some other logarithm
def gradient_k(alpha_k, alpha_sum, N):
    return N * (sp.digamma(alpha_sum) - sp.digamma(alpha_k) + np.log(alpha_k / alpha_sum))


def hessian_inverse_gradient(alpha, M):
    alpha_sum = np.sum(alpha)
    z = M * sp.polygamma(1, alpha_sum)

    Q = np.zeros(NUM_TOPICS_K)
    gradient = np.zeros(NUM_TOPICS_K)

    if alpha_sum == 0:
        print("alpha:", alpha)

    # TODO: Check what j is in the formulas
    for k in range(NUM_TOPICS_K):
        Q[k] = -M * sp.polygamma(1, alpha[k])
        gradient[k] = gradient_k(alpha[k], alpha_sum, M)

    inv_Q = np.reciprocal(Q)
    b = np.sum(gradient * inv_Q) / (np.sum(inv_Q) + (1 / z))
    # print("z:", z)
    # print("sum of inv_Q:", np.sum(inv_Q))

    return (gradient - b) / Q


def beta_i_j(phi, documents, i, j):
    s = 0
    for d in range(len(documents)):
        for n in range(len(documents[d])):
            # if d > 19000:  # 19042 gave error
            # print(documents[d][n])
            # s += phi[d][n][i] * (documents[d][n] ** j)
            s += phi[d][n][i] * documents[d][n][j]

    return s


def max_step(alpha, beta, phi, V, corpus):
    for i in range(NUM_TOPICS_K):
        for j in range(V):
            beta[i][j] = beta_i_j(phi, corpus, i, j)

    # theta_averages = row_averages(theta, len(corpus))
    # for docIdx in range(len(corpus)):
    alpha = alpha - hessian_inverse_gradient(alpha, len(corpus))

    return alpha, beta


def variational_expectation_maximization():
    vocab = get_vocab()
    V = len(vocab)
    corpus = make_corpus(get_vocab())
    NUM_DOCS = len(corpus)
    # phi, gamma, alpha, beta = init_params(corpus, V)
    alpha, beta = init_alpha_beta(V)
    # theta = init_theta(NUM_DOCS)
    phi = [None] * NUM_DOCS
    gamma = [None] * NUM_DOCS

    for emStep in range(EM_ITERATIONS):
        print("EM-step:", emStep)
        # phi, gamma, for each doc in list
        print("E-step...")
        for docIdx in range(NUM_DOCS):
            doc = corpus[docIdx]
            N = np.size(doc, axis=0)
            phi[docIdx], gamma[docIdx] = variational_inference(N, alpha, beta, doc, phi[docIdx], gamma[docIdx])

        # Maximization step
        print("M-step...")
        alpha, beta = max_step(alpha, beta, phi, V, corpus)

    return phi, gamma, alpha, beta


def find_topic_words(beta, topic, vocab, numWords=5):
    wordDist = beta[topic]
    topWords = np.sort(wordDist)[::-1]
    extractedWords = []
    for i in range(numWords):
        extractedWords.append(one_hot_to_string(topWords[i], vocab))
    return extractedWords


def joeys_algorithm():
    phi, gamma, alpha, beta = variational_expectation_maximization()
    print("Phi:", phi)
    print("Gamma:", gamma)
    print("alpha:", alpha)
    print("beta:", beta)
    vocab = get_vocab()
    topicWords = []
    for i in range(NUM_TOPICS_K):
        topicWords.append(find_topic_words(beta, i, vocab, 5))
    print("Topics and words")
    print(np.array(topicWords))
    np.save("params", np.array([phi, gamma, alpha, beta]))


if __name__ == "__main__":
    joeys_algorithm()
