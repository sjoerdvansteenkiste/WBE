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
python setup.py install
``` 

## Usage 


To evolve any neural network using the Wavelet-based Encoding take the following steps:

1. Determine the level of compression **l**.
2. Construct a list **tensor_dim** containing all network weight-tensor dimensions.
  * Ensure each tensor dim is divisible by 2^l (pad by increasing the dim in **tensor_dim** if necessary).
  * Pass on the dimension that ensures maximal spatial correlation in the network weights i.e. for an input matrix of 
    dimension *h x mn* that receives flatted pixel input of an *m x n* image, use *h x m x n* in the tensor_dim.
3. Compute the number of approximation wavelet coefficients to be evolved using `get_gen_total(tensor_dim, l)`.
4. Determine the wavelet-basis function to use, and compute the corresponding polyphase filter coefficients **poly**
  * transform lists of low- and high-pass filter coefficients to polyphase form using `classic2polyphase`.
  * using `lattice_decomposition` to compute wavelet basis function from an arbitrary set of parameters (which can be 
    evolved alongside the wavelet coefficients).
5. Use `decode(poly, genes, tensor_dim, l)` to decode the approximation wavelet coefficients into a list of weight 
tensors of the specified dimension.

See **examples/genes_to_weights.py** for an example. 