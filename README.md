# A Wavelet-based Encoding for Neuroevolution

The code in this repository implements the Wavelet-based Encoding as proposed in our conference paper: A Wavelet-based Encoding for Neuroevolution (
http://dx.doi.org/10.1145/2908812.2908905) to appear at GECCO'16. The provided implementation supports weight-tensors up to three dimensions. 


## Installation

Quick instructions for installation.

```bash
# Get WBE
git clone git@github.com:sjoerdvansteenkiste/wavelets.git
# Install
cd WBE
pip install -r requirements.txt
pip install .
``` 

## Usage 


To evolve any neural network using the Wavelet-based Encoding take the following steps:

1. Determine the level of compression **l**.
2. Construct a dictionary **structure** containing all weight-tensor dimensions, which outlines the structure of the 
phenotoype, e.g. ```python structure = {'L1': {'W': (12, 4, 6, 8), 'bias': (12, 14, 10)}, 'L2': {'theta': 8}}```
  * Ensure each tensor dim is divisible by 2^l (pad by increasing the dim in **tensor_dim** if necessary).
  * Pass on the dimension that ensures maximal spatial correlation in the network weights i.e. for an input matrix of 
    dimension *h x mn* that receives flatted pixel input of an *m x n* image, use *h x m x n* instead.
3. Compute the number of approximation wavelet coefficients to be evolved using `encoding_dimensionality(structure, l)`.
4. Choose the wavelet-basis function to use, and compute the corresponding polyphase filter coefficients **poly**. 
   See `wavelet_functions.py` for some pre-implemented filters.
  * Use `lattice_structure` to compute wavelet basis function from an arbitrary set of parameters (which can be 
    evolved alongside the wavelet coefficients).
5. Use `decode(chromosome, poly, structure, l)` to decode the approximation wavelet coefficients into a list of weight 
tensors of the specified dimension.

See **examples/genes_to_weights.py** for an example. 