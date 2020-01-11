import numpy as np

"""Contains the implemented Variational Inference algorithm for LDA and associated functions"""

NUMBER_OF_TOPICS_K = 17


#Pseudocode for VI, LDA
""" (1) initialize φ0 ni := 1/k for all i and n 
    (2) initialize γi := αi+N/k for all i 
    (3) repeat 
    (4)     for n = 1 to N 
    (5)         for i = 1 to k 
    (6)             φt+1 ni := βiwn exp(Ψ(γt i)) 
    (7)         normalize φt+1 n to sum to 1. 
    (8)     γt+1 := α+∑N n=1φt+1 n 
    (9) until convergence"""

def variational_Inference(n , i, alfa, number_of_words):
    """
    Runs the VI algorithm according to the pseudocode above.
    :param
    :return
    """

    phi0 = init_phi(n, i)
    gamma = init_gamma(i, alfa, number_of_words)

    has_not_convergence = True

    while has_not_convergence:
        for n in range(n):
            for i in range(k):



def init_phi(n, i):
    phi0 = np.zeros((n, i))
    return phi0 + 1 / NUMBER_OF_TOPICS_K

def init_gamma(i, alfa, number_of_words):
    return alfa + number_of_words