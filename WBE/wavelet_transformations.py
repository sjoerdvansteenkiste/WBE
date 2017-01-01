#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import numpy as np
from numpy import concatenate, dot, zeros_like


def _dwt(s, poly):
    """
    Computes the discrete wavelet transform for a 1D signal
    :param s: the signal to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :return: the transformed signal
    """
    assert len(s) % 2 == 0, 'signal length ({}) does not allow for a decomposition'.format(len(s))

    s = s.reshape((s.shape[0]/2, 2)).transpose()

    decomp = zeros_like(s, dtype=float)
    for z in range(poly.shape[2]):
        decomp += dot(poly[:, :, z], concatenate((s[:, z:], s[:, :z]), axis=1))

    return np.concatenate((decomp[0], decomp[1]), axis=0).transpose()


def _idwt(s, poly):
    """
    Computes the inverse discrete wavelet transform for a 1D signal
    :param s: the decomposed signal
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :return: the transformed signal
    """

    s = np.array([s[:len(s)/2], s[len(s)/2:]])

    recon = zeros_like(s, dtype=float)
    for z in range(poly.shape[2]):
        recon += dot(poly[:, :, z].T, concatenate((s[:, s.shape[1]-z:], s[:, :s.shape[1]-z]), axis=1))

    return recon.transpose().reshape(1, 2*recon.shape[1])[0]


def dwt(s, poly):
    if s.ndim == 1:
        return _dwt(s, poly)

    # transform over the first n-1 dimensions recursively
    for i in range(s.shape[0]):
        s[i] = dwt(s[i], poly)

    # iterate over the last dimension
    for i in range(s.shape[-1]):
        s[..., i] = dwt(s[..., i], poly)

    return s


def idwt(s, poly):
    if s.ndim == 1:
        return _idwt(s, poly)

    # transform over the first n-1 dimensions recursively
    for i in range(s.shape[0]):
        s[i] = idwt(s[i], poly)

    # iterate over the last dimension
    for i in range(s.shape[-1]):
        s[..., i] = idwt(s[..., i], poly)

    return s


def fdwt(s, poly, l=1):
    for level in range(l):
        slc = [slice(None)] * len(s.shape)
        for i in range(len(s.shape)):
            slc[i] = slice(0, s.shape[i] / (2**level))

        s[slc] = dwt(s[slc], poly)

    return s


def bidwt(s, poly, l=1):
    for level in reversed(range(l)):
        slc = [slice(None)] * len(s.shape)
        for i in range(len(s.shape)):
            slc[i] = slice(0, s.shape[i] / (2**level))

        s[slc] = idwt(s[slc], poly)

    return s
