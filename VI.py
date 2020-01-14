import numpy as np
# import TextExtraction
from TextExtraction import get_vocab, make_corpus
import scipy.special as sp
import scipy.signal as ss

"""Contains the implemented Variational Inference algorithm for LDA and associated functions"""

NUM_TOPICS_K = 17
VI_ITERATIONS = 10000

phi = 0
gamma = 0


def beta_i_j(phi, documents, i, j):
    s = 0
    for d in range(len(documents)):
        for n in range(len(documents[0])):
            s += phi[d][n][i] * (documents[d][n] ** j)

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
def variational_inference(N, alfa, beta):
    """
    Runs the VI algorithm according to the pseudocode above.
    :param
    :return
    """

    '''
    phi = init_phi(N)
    gamma = init_gamma(NUM_TOPICS_K, alfa, N)
    '''

    vocab = get_vocab()
    V = len(vocab)
    corpus = make_corpus(get_vocab())
    NUM_DOCS = len(corpus)
    phi, gamma, alfa, beta = init_params(corpus, V)

    has_not_converged = True
    t = 0

    while has_not_converged:
        for n in range(N):
            for i in range(NUM_TOPICS_K):
                phi = beta[i, n] * np.exp(sp.digamma(gamma[i]))
            phi[n] = normalize_row(phi)
        gamma = sum_columns_alfa(alfa, phi)
        has_not_converged = convergence_check(t)

    return phi, gamma


def sum_columns_alfa(alfa, phi):
    return alfa + np.sum(phi, axis=1)


def convergence_check(t):
    t += 1
    if t == VI_ITERATIONS:
        return False
    else:
        return True


def normalize_row(row):
    """
    Takes in a matrix row and normalizes it.
    :param A row of a matrix
    :return The normalized matrix
    """
    row_sum = np.sum(row)
    return row / row_sum


# for one document
def init_phi(N):
    phi0 = np.zeros((N, NUM_TOPICS_K))
    return phi0 + 1 / NUM_TOPICS_K


def init_gamma(alfa, N):
    return alfa + N / NUM_TOPICS_K


# once for each document
def init_beta(V):
    beta = []
    for i in range(NUM_TOPICS_K):
        beta.append([1 / V] * V)
    return beta


def init_alfa_beta(V):
    alfa = [0.5] * NUM_TOPICS_K
    beta = init_beta(V)
    return alfa, beta


def init_phi_gamma(N, alfa):
    phi = init_phi(N)
    gamma = init_gamma(alfa, N)
    return phi, gamma


'''
def init_params_one_doc(doc, V):
    N = len(doc)
    phi = init_phi(N)
    alfa = [0.5] * NUM_TOPICS_K
    beta = init_beta(V)
    gamma = init_gamma(alfa, N)

    return phi, gamma, alfa, beta
'''

'''
def init_params(corpus, V):
    NUM_DOCS = len(corpus)
    # Initialization
    phi = []  # M * N * k
    gamma = []  # M * k
    alfa = []  # M * k
    beta = []  # M * k * V
    for docIdx in range(NUM_DOCS):
        doc = corpus[docIdx]
        N = len(doc)
        phi.append(init_phi(N))
        alfa.append([0.5] * NUM_TOPICS_K)
        beta.append(init_beta(V))
        gamma.append(init_gamma(alfa[docIdx], N))

    return phi, gamma, alfa, beta
'''

#Written in the context of a single document
#Uses natural logarithm
def gradient_k(theta_average_k, alfa_k, alfa_sum, N):
    return N*(sp.psi(alfa_sum) - sp.psi(alfa_k) + np.log(theta_average_k))

def hessian_inverse_gradient(alfa, document, theta_average_k, M):
    N = len(document)

    z = N * sp.polygamma(1, np.sum(alfa))

    Q = np.zeros(NUM_TOPICS_K)
    for k in range(NUM_TOPICS_K):
        Q[k] = -N*sp.polygamma(1, np.sum(alfa))

    #id_matrix = np.diag(np.array(NUM_TOPICS_K))
    #inv_Q = np.linalg.inv(Q)
    hessian_inverse_gradient = np.array([])
    alfa_sum = np.sum(alfa)
    denominator = 0
    numerator = 0
    for j in range(NUM_TOPICS_K):
        denominator += gradient_k(theta_average_k, alfa[j], alfa_sum, M)/Q[j]
        numerator += 1/Q[j]
    b = numerator/denominator

    for j in range(NUM_TOPICS_K):
        hessian_inverse_gradient[j] = (gradient_k(theta_average_k, alfa[j], alfa_sum, M) - b)/Q[j]
    
    return hessian_inverse_gradient


def max_step(alfa, beta, phi, V, corpus):
    for i in range(NUM_TOPICS_K):
        pass
        for j in range(V):
            beta[i][j] = beta_i_j(phi, corpus, i, j)

    for docIdx in range(len(corpus)):
        alfa[docIdx] = alfa[docIdx] - hessian_inverse_gradient(alfa[docIdx], corpus[docIdx], theta_average_k, len(corpus))

    return alfa, beta


def variational_expectation_maximization():
    vocab = get_vocab()
    V = len(vocab)
    corpus = make_corpus(get_vocab())
    NUM_DOCS = len(corpus)
    # phi, gamma, alfa, beta = init_params(corpus, V)
    alfa, beta = init_alfa_beta(V)

    # phi, gamma, for each doc in list
    phi = []
    gamma = []
    for docIdx in range(NUM_DOCS):
        doc = corpus[docIdx]
        N = np.size(doc, axis=0)  # TODO DOUBLE CHECK
        phi_doc, gamma_doc = variational_inference(N, alfa, beta)
        phi.append(phi_doc)
        gamma.append(gamma_doc)

    # Maximization step
    alfa, beta = max_step(alfa, beta, phi, V)
