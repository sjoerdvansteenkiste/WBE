import numpy as np


def classic2polyphase(c, d):
    """
    Transforms the coefficients of the low- and high-pass filter to a polyphase form
    :param c: list of low-pass coefficients
    :param d: list of high-pass coefficients
    :return: polyphase format containing c and d
    """
    return np.concatenate((np.array(c)[np.newaxis, :], np.array(d)[np.newaxis, :]), axis=0)


def get_gene_total(tensor_dim, l):
    """
    Compute the total number of genes required to encode a list of weight tensors as in the wavelet domain
    :param tensor_dim: list of tensor dimensions of the weight tensors to be encoded
    :param l: level of compression
    :return: number of coefficients
    """
    n_coeff = 0
    for dim in tensor_dim:
        assert np.max(np.mod(np.array(dim), 2 ** l)) == 0, 'tensor dimension ({}) does not allow for a {}-level encoding'\
            .format(dim, l)

        n_coeff += np.prod(np.array(dim)/(2**l))

    return n_coeff