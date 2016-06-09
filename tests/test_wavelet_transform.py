import pytest
import numpy as np
import WBE as wbe

from WBE.wavelet_transformations import fdwt, bidwt

# define list of weight-tensor dimensions
tensor = np.random.randn(16, 12, 16, 20)

# define the polyphase wavelet filter either use Daubechies
poly = wbe.lattice_structure([.4534, .24234, .324])
l = 2

decomp = fdwt(tensor.copy(), poly, l)
recon = bidwt(decomp.copy(), poly, l)

assert np.any(np.abs(tensor - recon) < 10e-8)

# define list of weight-tensor dimensions
tensor = np.random.randn(16)

decomp = fdwt(tensor.copy(), poly, l)

assert abs(np.linalg.norm(decomp) - np.linalg.norm(tensor)) < 10e-8