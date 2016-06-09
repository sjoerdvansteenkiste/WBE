import numpy as np


def classic2polyphase(c, d):
    """
    Transforms the coefficients of the low- and high-pass filter to a polyphase form
    :param c: list of low-pass coefficients
    :param d: list of high-pass coefficients
    :return: polyphase format [2, 2, n-1] containing c and d
    """

    H = np.zeros((2, 2, len(c)/2))
    for i in range(0, len(c), 2):
        H[..., i/2] = np.array([c[i:i+2], d[i:i+2]])

    return H
