import WBE as wbe
import numpy as np

# define list of weight-tensor dimensions
tensor_dim = [[8, 16], [16], [24, 48, 12]]

# compute number of genes to encode such a list of tensors, for a given l
l = 2
n_genes = wbe.get_gene_total(tensor_dim, l)
genes = np.random.randn(n_genes)

# define the polyphase wavelet filter either use Daubechies
poly = wbe.daubechies(order=2)

# or a lattice decomposition to support dynamic wavelet basis function evolution
poly = wbe.lattice_decomposition(in_theta=[.23, .56])

# compute the list of weight tensors from the wavelet-coefficients -- genes
tensor_list = wbe.decode(poly, genes, tensor_dim, l)