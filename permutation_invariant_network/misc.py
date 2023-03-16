import numpy as np
import torch as ch
from tqdm import tqdm
from typing import List, Tuple, Any


# Function to extract model parameters
def get_weight_layers(m: Any,
                      transpose: bool = True,
                      first_n: int = np.inf,
                      start_n: int = 0,
                      layer_prefix: str = None) -> Tuple[List[int], List[ch.Tensor]]:
    """
    Extract model parameters from given model (with FC layers)
    :param m: model for which parameters requested
    :param transpose: parameters need to be transposed? (eg. GraphConv layers)
    :param first_n: use only first-N layer parameters
    :param start_n: start extracting parameters from layer N
    :param layer_prefix: String to look for in parameter name to consider as weight
    :return: (dimensions of layers, model parameters)
    """
    dims, weights, biases = [], [], [],
    i = 0

    # Iterate through model parameters
    for name, param in m.named_parameters():

        if layer_prefix and not name.startswith(layer_prefix):
            continue

        # If W matrix
        if "weight" in name:
            param_data = param.data.detach().cpu()
            if transpose:
                param_data = param_data.T

            # Store weight matrix
            weights.append(param_data)

            # Keep track of dimension
            dims.append(weights[-1].shape[0])

        # If bias values
        if "bias" in name:

            # Store bias matrix
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

        # Assume each layer has weight & bias
        i += 1

        # If requested, start looking from start_n layer
        if (i - 1) // 2 < start_n:
            dims, weights, biases = [], [], []
            continue

        # If requested, look at only first_n layers
        if i // 2 > first_n - 1:
            break

    # Take a pass through weights and biases together
    cctd = []
    for w, b in zip(weights, biases):
        # Concatenate together
        combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    # Return dimensions, model parameters
    return (dims, cctd)


def prepare_batched_data(X: List[List[ch.Tensor]]) -> List[ch.Tensor]:
    """
    Given a list of model parameters for various models, combine 
    them together for batching
    :param X: list of models' parameters
    :return: list of parameters, batched per parameter
    """
    inputs = [[] for _ in range(len(X[0]))]
    for x in X:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = np.array([ch.stack(x, 0) for x in inputs], dtype='object')
    return inputs


# Function to extract model parametrs for all models in given list
def load_model_parameters(model_list: List[Any],
                          first_n: int = np.inf,
                          start_n: int = 0,
                          layer_prefix: str = None) -> np.ndarray:
    """
    Given a list of models, extract their parameters
    :param model_list: list of models' parameters
    :param first_n: extract parameters only for first n layers
    :param start_n: start extracting parameters from layer N
    :param layer_prefix: String to look for in parameter name to consider as weight
    :return: (dimensions of layers, list of model parameters (list))
    """
    vecs = []

    for model in tqdm(model_list):

        # Get model params, shift to GPU
        dims, fvec = get_weight_layers(
            model, first_n=first_n, start_n=start_n,
            layer_prefix=layer_prefix)
        # fvec = [x.cuda() for x in fvec]
        fvec = [x for x in fvec]

        vecs.append(fvec)

    return (dims, np.array(vecs, dtype=object))
