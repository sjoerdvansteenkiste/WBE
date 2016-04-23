import numpy as np
import wavelet_transformations as wav


def decode(poly, genes, tensor_dim, l):
    """
    Implements the WBE and decodes a list of approximation wavelet coefficients -- genes into corresponding
    weight-tensors via a lossy inverse wavelet transformation
    :param poly: the polyphase filter matrix that corresponds to the high-pass and low-pass filters of a wavelet filter bank
    :param genes: Numpy array of approximation wavelet coefficients -- genes to be decoded
    :param tensor_dim: list of tensor dimensions for the genes to be encoded into
    :param l: corresponds to the number of inverse wavelet transformations taken
    """

    tensor_list = list()
    for dim in tensor_dim:
        if len(dim) > 3:
            raise ValueError("N-dimensional tensors are only supported up to n = 3 ")

        # select genes
        n_coeff = np.prod(np.array(dim)/(2**l))
        wavelet_coeff, genes = np.array_split(genes, [n_coeff])

        # embed coefficients
        embedding = embed_coefficients(wavelet_coeff, dim, l)

        # obtain weight representation
        if len(dim) == 1:
            a, d = embedding
            tensor_list.append(wav.idwt(a, d, poly, l)[0, :])
        elif len(dim) == 2:
            tensor_list.append(wav.idwt_2d(embedding, poly, l))
        elif len(dim) == 3:
            tensor_list.append(wav.idwt_3d(embedding, poly, l))

    return tensor_list


def embed_coefficients(coefficients, dimension, l):
    """
    Compute an embedding of the given dimension with the approximation coefficients structured along the same number of
    dimensions in the top left corner
    :param coefficients: the approximation wavelet coefficients to be embedded
    :param dimension: the dimension of the embedding
    :param l: the level of the approximation coefficients
    :return: the embedding shaped as [al, 0, ...., 0] along each dimension
    """
    compressed_dimension = np.array(dimension)/(2**l)
    embedding = np.zeros(dimension)

    if len(dimension) == 1:
        a = coefficients
        d = [np.zeros(len(a) * (2**level)) for level in reversed(range(l))]

        return a, d

    elif len(dimension) == 2:
        embedding[:compressed_dimension[0], :compressed_dimension[1]] = coefficients.reshape(
            compressed_dimension[0], compressed_dimension[1])
    elif len(dimension) == 3:
        embedding[:compressed_dimension[0], :compressed_dimension[1], :compressed_dimension[2]] = coefficients.reshape(
            compressed_dimension[0], compressed_dimension[1], compressed_dimension[2])

    return embedding