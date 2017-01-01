#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import numpy as np
import WBE.wavelet_transformations as wav


def embed(genes, structure, l):
    embedding = np.zeros(structure)

    slc = [slice(None)] * len(embedding.shape)
    for i in range(len(embedding.shape)):
        slc[i] = slice(0, embedding.shape[i]/(2**l))

    embedding[slc] = genes.reshape(np.array(structure)/(2**l))

    return embedding


def _decode(chromosome, poly, structure, l):
    if not isinstance(structure, dict):
        assert np.mod(structure, (2 ** l)).any() == 0, "Decomposition level exceeds signal dimensionality: Signal " \
                                                       "shape ({}) does not allow for a {}-level " \
                                                       "decomposition".format(structure, l)

        n_genes = np.prod(np.array(structure)/(2**l))
        genes, chromosome = np.array_split(chromosome, [n_genes])

        return wav.bidwt(embed(genes, structure, l), poly, l), chromosome

    _buffer = dict()
    for k, v in structure.items():
        _buffer[k], chromosome = _decode(chromosome, poly, v, l)

    return _buffer, chromosome


def decode(chromosome, poly, structure, l):
    _buffer, chromosome = _decode(chromosome, poly, structure, l)

    return _buffer


def encoding_dimensionality(structure, l):
    if not isinstance(structure, dict):
        assert np.mod(structure, (2 ** l)).any() == 0, "Decomposition level exceeds signal dimensionality: Signal " \
                                                       "shape ({}) does not allow for a {}-level " \
                                                       "decomposition".format(structure, l)
        return np.prod(np.array(structure)/(2**l))

    count = 0
    for k, v in structure.items():
        count += encoding_dimensionality(v, l)

    return count