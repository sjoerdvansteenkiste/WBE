from numpy import max, concatenate, dot, zeros_like, array, newaxis, mod
import numpy as np
import utilities as utils


def dwt(s, poly, l=1):
    """
    Computes the discrete wavelet transform for a 1D signal
    :param s: the signal to be processed
    :param poly: polyphase filter matrix cointing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed signal
    """
    assert len(s) % (2**l) == 0, 'signal length ({}) does not allow for a {}-level decomposition'.format(len(s), l)

    detail = []
    approximation = array(s)
    for level in range(l):
        s = approximation.reshape((approximation.shape[0]/2, 2)).transpose()

        decomposition = zeros_like(s, dtype=float)
        for z in range(poly.shape[1]/2):
            decomposition += dot(poly[:, 2*z:2*z+2], concatenate((s[:, z:], s[:, :z]), axis=1))

        approximation = decomposition[0, :]
        detail.append(decomposition[1, :])

    return approximation, detail


def idwt(a, d, poly, l=1):
    """
    Computes the inverse discrete wavelet transform for a 1D signal
    :param a: the approximation coefficients at the deepest level
    :param d: a list of detail coefficients for each level
    :param poly: polyphase filter matrix cointing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed signal
    """
    assert len(d) == l, 'insufficient detail coefficients provided for reconstruction depth {}'.format(l)

    if len(a.shape) == 1:
        a = a[newaxis, :]

    for level in reversed(range(l)):
        decomposition = concatenate((a, d[level][newaxis, :]), axis=0)

        reconstruction = zeros_like(decomposition, dtype=float)
        for z in range(poly.shape[1]/2):
            reconstruction += dot(poly[:, 2*z:2*z+2].transpose(), concatenate(
                (decomposition[:, decomposition.shape[1]-z:], decomposition[:, :decomposition.shape[1]-z]), axis=1))

        a = reconstruction.transpose().reshape(1, 2*a.shape[1])

    return a


def dwt_2d(image, poly, l=1):
    """
    Computes the discrete wavelet transform for a 2D input image
    :param image: input image to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed image
    """
    assert max(mod(image.shape, 2**l)) == 0, 'image dimension ({}) does not allow for a {}-level decomposition'.format(image.shape, l)

    image_ = image.copy()
    for level in range(l):
        sub_image = image_[:(image.shape[0]/(2**level)), :(image.shape[1]/(2**level))]

        for row in range(sub_image.shape[0]):
            s = sub_image[row, :]
            a, d = dwt(s, poly)

            sub_image[row, :] = concatenate((a[newaxis, :], d[0][newaxis, :]), axis=1)

        for col in range(sub_image.shape[1]):
            s = sub_image[:, col]
            a, d = dwt(s, poly)

            sub_image[:, col] = concatenate((a, d[0]), axis=0)

    return image_


def idwt_2d(image, poly, l=1):
    """
    Computes the inverse discrete wavelet transform for a 2D input image
    :param image: input image to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed image
    """
    assert max(mod(image.shape, 2**l)) == 0, 'image dimension ({}) does not allow for a {}-level reconstruction'.format(image.shape, l)

    image_ = image.copy()
    for level in reversed(range(l)):

        sub_image = image_[:(image.shape[0]/(2**level)), :(image.shape[1]/(2**level))]

        for col in range(sub_image.shape[1]):
            a = sub_image[:sub_image.shape[0]/2, col]
            d = sub_image[sub_image.shape[0]/2:, col]

            sub_image[:, col] = idwt(a, [d], poly)

        for row in range(sub_image.shape[0]):
            a = sub_image[row, :sub_image.shape[1]/2]
            d = sub_image[row, sub_image.shape[1]/2:]

            sub_image[row, :] = idwt(a, [d], poly)

    return image_


def dwt_3d(cube, poly, l=1):
    """
    Computes the discrete wavelet transform for a 3D input cube
    :param cube: input cube to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed cube
    """
    assert max(mod(cube.shape, 2**l)) == 0, 'cube dimension ({}) does not allow for a {}-level decomposition'.format(cube.shape, l)

    cube_ = cube.copy()
    for level in range(l):
        sub_cube = cube_[:(cube.shape[0]/(2**level)), :(cube.shape[1]/(2**level)), :(cube.shape[2]/(2**level))]

        # iterate over the last dimension
        for i in range(sub_cube.shape[2]):
            sub_cube[:, :, i] = dwt_2d(sub_cube[:, :, i], poly)

        # perform 1-d transform over the first two dimensions
        for i in range(sub_cube.shape[0]):
            for j in range(sub_cube.shape[1]):
                a, d = dwt(sub_cube[i, j, :], poly)
                sub_cube[i, j, :] = concatenate((a, d[0]), axis=0)

    return cube_


def idwt_3d(cube, poly, l=1):
    """
    Computes the inverse discrete wavelet transform for a 3D input cube
    :param cube: input cube to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed cube
    """
    assert max(mod(cube.shape, 2**l)) == 0, 'cube dimension ({}) does not allow for a {}-level reconstruction'.format(cube.shape, l)

    cube_ = cube.copy()
    for level in reversed(range(l)):

        sub_cube = cube_[:(cube.shape[0]/(2**level)), :(cube.shape[1]/(2**level)), :(cube.shape[2]/(2**level))]

        # reverse over the first two dimensions
        for i in range(sub_cube.shape[0]):
            for j in range(sub_cube.shape[1]):
                a = sub_cube[i, j, :sub_cube.shape[2]/2]
                d = sub_cube[i, j, sub_cube.shape[2]/2:]
                sub_cube[i, j, :] = idwt(a, [d], poly)

        # iterate over the last dimension
        for i in range(sub_cube.shape[2]):
            sub_cube[:, :, i] = idwt_2d(sub_cube[:, :, i], poly)

    return cube_
