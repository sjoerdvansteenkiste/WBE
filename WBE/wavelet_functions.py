import numpy as np
from WBE import utilities


def daubechies(order, polyphase=True):
    """
    returns the lowpass and highpass filter coefficients for a Daubechies wavelet
    :param order: the order of the wavelet function
    :param polyphase: whether the output should be return in polyphase form
    :return: the Daubechies-<order> coefficients
    """
    if order is 1:
        c = [0.7071067812, 0.7071067812]
        d = [-0.7071067812, 0.7071067812]
    elif order is 2:
        c = [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131]
        d = [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226]
    elif order is 3:
        c = [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569]
        d = [-0.3326705529509569, 0.8068915093133388, -0.4598775021193313, -0.13501102001039084, 0.08544127388224149, 0.035226291882100656]
    else:
        raise ValueError('order is not supported')

    if polyphase:
        return utilities.classic2polyphase(c, d)

    return c, d


def lattice_decomposition(in_theta, p=1):
    """
    Returns a polyphase matrix containing the wavelet filters. Order of the returned filter is len(in_theta)+1, i.e.
    for p = 1 and in_theta [.455] this yields a zeroth-order balanced wavelet of degree n = 2 (i.e. 4 coefficients)
    :param in_theta: a list containing the coefficients of the parametrization
    :param p: the number of p vanishing moments to include (currently only supports p=1)
    :return: a polyphase filter matrix containing the coefficients
    """

    if p > 1:
        raise ValueError("p > 1 is currently unsupported, use p = 1 for a single vanishing moment")

    H = np.eye(2)
    n_coeffs = 1
    n = len(in_theta)+p

    theta = in_theta
    if p == 1:
        theta = np.concatenate((np.array([np.pi / 4 - np.mod(sum(theta), 2 * np.pi)]), theta))

    for k in range(n-1, 0, -1):
        rot = np.array([[np.cos(theta[k]), 0, 0, -np.sin(theta[k])], [np.sin(theta[k]), 0, 0, np.cos(theta[k])]])
        H = polynomial_matrix_multiplication(H, np.array(rot), n_coeffs, 2)
        n_coeffs += 1

    rot = np.array([[np.cos(theta[0]), np.sin(theta[0])], [np.sin(theta[0]), -np.cos(theta[0])]])

    H = polynomial_matrix_multiplication(H, np.array(rot), n_coeffs, 1)

    return H


# assume coefficients are stored as Ez0 Oz0 Ez-1 Oz-2
def polynomial_matrix_multiplication(M1, M2, n_coeffs_M1, n_coeffs_M2):
    """
    Helper function to perform matrix multiplication where each matrix entry contains the coefficients of a polynomial
    matrix multiplication is of the form M = M1 x M2
    :param M1: The left input matrix
    :param M2: The right input matrix
    :param n_coeffs_M1: The number of coefficients (of the polynomial) stored in M1
    :param n_coeffs_M2: The number of coefficients (of the polynomial) stored in M2
    :return: the matrix product M=M1M2
    """
    new_n_coeffs = n_coeffs_M1+n_coeffs_M2-1
    M = np.zeros((2, 2*new_n_coeffs))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_coefficients = np.convolve(M1[i, k:M1.shape[1]:2], M2[k, j:M2.shape[1]:2])
                for coeff in range(new_n_coeffs):
                    M[i, j+(2*coeff)] += new_coefficients[new_n_coeffs-coeff-1]
    return M