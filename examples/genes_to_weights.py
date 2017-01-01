#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import WBE as wbe
import numpy as np

# define list of weight-tensor dimensions
structure = {'L1': {'W': (12, 4, 6, 8), 'bias': (12, 14, 10)}, 'L2': {'theta': 8}}

# compute number of genes to encode such a list of tensors, for a given l
l = 1
n_genes = wbe.encoding_dimensionality(structure, l)
chromosome = np.random.randn(int(n_genes))

# define the polyphase wavelet filter either use Daubechies
poly = wbe.daubechies(order=2)

# or a lattice decomposition to support dynamic wavelet basis function evolution
poly = wbe.lattice_structure(in_theta=[.23, .56])

# compute the list of weight tensors from the wavelet-coefficients -- genes
phenotype = wbe.decode(chromosome, poly, structure, l)