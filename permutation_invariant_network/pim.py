import torch as ch
import torch.nn as nn
from typing import List


class PermInvModel(nn.Module):
    def __init__(self,
                 dims: List[int],
                 inside_dims: List[int] = [64, 8],
                 n_classes: int = 1,
                 dropout: float = 0.5):
        """
        Permutation Invariant Network, as proposed in
        https://dl.acm.org/doi/pdf/10.1145/3243734.3243834)
        :param dims (list): a list of dimension sizes for all FC layers 
        :param inside_dims (list): dimensions for internal hidden layers of $\phi_i$
        :param n_classes: Number of classes for meta-classifier
        :param dropout:  Dropout percentage for dropout layers
        """

        if dropout > 1 or dropout < 0:
            raise ValueError("Invalid dropout ratio requested!")

        super(PermInvModel, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # Function to define $\phi$ with given input dimension y
        def make_mini(y):
            layers = [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        # Iterate through given dimensions for layers
        for i, dim in enumerate(self.dims):
            if i > 0:
                # Previous layer representation will be concatenated
                # to input feature as well
                prev_layer = inside_dims[-1] * dim

            # Add +1 for bias term
            self.layers.append(make_mini(prev_layer + 1 + dim))

        # Store all $\phi_i$ modele as parameters
        self.layers = nn.ModuleList(self.layers)

        # Final network $\rho$ to combine them all together
        self.rho = nn.Linear(inside_dims[-1] * len(dims), n_classes)

    def forward(self, params: List[ch.Tensor]) -> ch.Tensor:
        """
        :param params (list): list of model parameters, each of size
        (batch_size, hidden_out, hidden_in), where hidden_out contains
        +1 term for bias
        """

        reps = []
        prev_layer_reps = None
        # Iterate through all layers
        for param, layer in zip(params, self.layers):

            if prev_layer_reps is None:
                # First layer, so no previous-layer representation
                param_eff = param  # hidden_in_eff = hidden_in

            else:
                # Adjust shape to allow concatenation
                prev_layer_reps = prev_layer_reps.repeat(
                    1, param.shape[1], 1)

                # Concatenate (w_i, b_i) with $N^{i-1}$
                param_eff = ch.cat((param, prev_layer_reps), -1)

            # shape: (batch_size, hidden_out, hidden_in)
            prev_shape = param_eff.shape

            # Use $\phi_i$ to generate $N^i_j$
            processed = layer(param_eff.view(-1, param_eff.shape[-1]))

            # Reshape back to (batch_size, hidden_out, inside_dims[-1])
            processed = processed.view(prev_shape[0], prev_shape[1], -1)

            # Layer summation over j to get $L_i$
            reps.append(ch.sum(processed, -2))

            # Store $N^i$ for use in next layer
            prev_layer_reps = processed.view(processed.shape[0], -1)
            prev_layer_reps = ch.unsqueeze(prev_layer_reps, -2)

        # Concatenate all $L_i$
        reps_c = ch.cat(reps, 1)

        # Compute $\rho(L)$
        logits = self.rho(reps_c)
        return logits
