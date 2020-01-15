import numpy as np
# import TextExtraction
from TextExtraction import get_vocab, make_corpus
import scipy.special as sp

"""Contains the implemented Variational Inference algorithm for LDA and associated functions"""

NUM_TOPICS_K = 2
VI_ITERATIONS = 1
EM_ITERATIONS = 3


def beta_i_j(phi, documents, i, j):
    s = 0
    # print(np.array(phi).shape)
    # TODO GETTING THEM ERRORS!!! OOGABOOGA
    for d in range(len(documents)):
        for n in range(len(documents[d])):
            # if d > 19000:  # 19042 gave error
            # print(documents[d][n])
            # s += phi[d][n][i] * (documents[d][n] ** j)
            s += phi[d][n][i] * documents[d][n][j]

    return s


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
def variational_inference(N, alpha, beta, doc):
    """
    Runs the VI algorithm according to the pseudocode above.
    :param
    :return
    """

    phi, gamma = init_phi_gamma(N, alpha)

    has_converged = False
    t = 0

    while not has_converged:
        for n in range(N):
            for i in range(NUM_TOPICS_K):
                nonZeroes = np.nonzero(doc[n])
                betaVal = 0.0001
                if np.size(nonZeroes) > 0:
                    betaVal = beta[i][nonZeroes[0]]

                phi[n][i] = betaVal * np.exp(sp.digamma(gamma[i]))
            phi[n] /= np.sum(phi[n])
        gamma = sum_rows_phi_plus_alpha(alpha, phi)

        t += 1
        has_converged = convergence_check(t)

    return phi, gamma


def sum_rows_phi_plus_alpha(alpha, phi):
    return alpha + np.sum(phi, axis=0)


def convergence_check(t):
    return t == VI_ITERATIONS


# for one document
def init_phi(N):
    phi0 = np.zeros((N, NUM_TOPICS_K))
    return phi0 + 1 / NUM_TOPICS_K


def init_gamma(alpha, N):
    return alpha + N / NUM_TOPICS_K


# once for each document
def init_beta(V):
    beta = []
    for i in range(NUM_TOPICS_K):
        beta.append(np.array([1 / V] * V))
    return np.array(beta)


def init_alpha_beta(V):
    alpha = np.array([0.5] * NUM_TOPICS_K)
    beta = init_beta(V)
    return alpha, beta


def init_phi_gamma(N, alpha):
    phi = init_phi(N)
    gamma = init_gamma(alpha, N)
    return phi, gamma


# Written in the context of a single document
# Uses natural logarithm
# Maybe try some other logarithm
def gradient_k(alpha_k, alpha_sum, N):
    return N * (sp.digamma(alpha_sum) - sp.digamma(alpha_k) + np.log(alpha_k / alpha_sum))


def hessian_inverse_gradient(alpha, document, M):
    N = len(document)

    alpha_sum = np.sum(alpha)
    z = N * sp.polygamma(1, alpha_sum)

    Q = np.zeros(NUM_TOPICS_K)
    gradient = np.zeros(NUM_TOPICS_K)

    if alpha_sum == 0:
        print("alpha:", alpha)

    # TODO: Check what j is in the formulas
    for k in range(NUM_TOPICS_K):
        Q[k] = -N * sp.polygamma(1, alpha[k])
        gradient[k] = gradient_k(alpha[k], alpha_sum, N)

    inv_Q = np.reciprocal(Q)
    b = np.sum(gradient * inv_Q) / (np.sum(inv_Q) + (1 / z))

    return (gradient - b) / Q


def max_step(alpha, beta, phi, V, corpus):
    for i in range(NUM_TOPICS_K):
        for j in range(V):
            beta[i][j] = beta_i_j(phi, corpus, i, j)

    # theta_averages = row_averages(theta, len(corpus))
    for docIdx in range(len(corpus)):
        alpha = alpha - hessian_inverse_gradient(alpha, corpus[docIdx], len(corpus))

    return alpha, beta


def variational_expectation_maximization():
    vocab = get_vocab()
    V = len(vocab)
    corpus = make_corpus(get_vocab())
    NUM_DOCS = len(corpus)
    # phi, gamma, alpha, beta = init_params(corpus, V)
    alpha, beta = init_alpha_beta(V)
    # theta = init_theta(NUM_DOCS)
    phi = []
    gamma = []

    for emStep in range(EM_ITERATIONS):
        print("EM-step:", emStep)
        # phi, gamma, for each doc in list
        phi = []
        gamma = []
        print("E-step...")
        for docIdx in range(NUM_DOCS):
            doc = corpus[docIdx]
            N = np.size(doc, axis=0)
            phi_doc, gamma_doc = variational_inference(N, alpha, beta, doc)
            phi.append(phi_doc)
            gamma.append(gamma_doc)

        # Maximization step
        print("M-step...")
        alpha, beta = max_step(alpha, beta, phi, V, corpus)

    return phi, gamma, alpha, beta


def runStuff():
    phi, gamma, alpha, beta = variational_expectation_maximization()
    print("Phi:", phi)
    print("Gamma:", gamma)
    print("alpha:", alpha)
    print("beta:", beta)


if __name__ == "__main__":
    runStuff()
