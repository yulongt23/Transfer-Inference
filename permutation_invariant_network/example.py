import torch as ch
import torch.nn as nn
import numpy as np
from train import train_model
from pim import PermInvModel
from misc import prepare_batched_data, load_model_parameters


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    # This file is for demonstration purposes
    # Do not expect any performance for this scenario (apart from overfitting)
    # Since all of these model are initialized with random values

    # Define dummy models
    n_models_each = 100
    models_0 = [DummyModel() for _ in range(n_models_each)]
    models_1 = [DummyModel() for _ in range(n_models_each)]

    # Get model parameters
    dims, models_params_0 = load_model_parameters(models_0)
    _, models_params_1 = load_model_parameters(models_1)

    # Define labels for dummy models
    y_0 = ch.zeros(n_models_each).float()
    y_1 = ch.ones(n_models_each).float()
    Y_train = ch.cat((y_0, y_1), 0)

    # Batch model parameters
    X_train = np.concatenate((models_params_0, models_params_1))
    X_train = prepare_batched_data(X_train)

    # Define meta-classifier
    meta_clf = PermInvModel(dims)
    meta_clf = meta_clf.cuda()

    # Train meta-classifier
    clf, tacc = train_model(
        meta_clf,
        (X_train, Y_train), (X_train, Y_train),
        epochs=100, binary=True, lr=1e-3,
        regression=False, batch_size=32, gpu=True)
