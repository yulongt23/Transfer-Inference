import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch as ch
from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple, Any


def acc_fn(x, y, binary):
    """
    Helper f to compute accuracy
    """
    if binary:
        return ch.sum((y == (x >= 0)))
    return ch.sum(y == ch.argmax(x, 1))


def get_preds(
        model: Any,
        X: List[ch.Tensor],
        batch_size: int,
        gpu: bool = False,
        binary: bool = True,
        regression: bool = False):
    """
    Get predictions for given data using model and batch size
    :param model: model to use
    :param X: data to predict
    :param binary: is task binary classification?
    :param regression: is task regression?
    :param batch_size: batch size
    """
    preds = []
    i = 0
    # Take note of number of samples
    n_samples = len(X[0])

    while i < n_samples:
        outputs = []

        # Model features stored as normal list
        param_batch = [x[i:i+batch_size] for x in X]
        if gpu:
            param_batch = [a.cuda() for a in param_batch]
        if binary or regression:
            outputs.append(model(param_batch)[:, 0])
        else:
            outputs.append(model(param_batch))

        preds.append(ch.cat(outputs, 0).detach())
        # Next batch
        i += batch_size

    preds = ch.cat(preds, 0)
    return preds


@ch.no_grad()
def test_model(
        model: Any,
        loss_fn: Any,
        X: List[ch.Tensor],
        Y: ch.Tensor,
        batch_size: int,
        binary: bool = True,
        regression: bool = False,
        gpu: bool = False) -> Tuple[float, float]:
    """
    Testing performance of meta-classifier on given data
    :param model: meta-classifier model
    :param loss_fn: loss function to evaluate model
    :param x: loss model parameters (data for meta-classifier)
    :param y: labels corresponding to given data
    :param batch_size: batch-size to use for evaluation
    :param binary: is task binary classification?
    :param regression: is task regression?
    :param gpu: use GPU for computations?
    :return: (average accuracy, average loss)
    """

    # Make sure model is in evaluation mode
    model.eval()

    # Batch data to fit on GPU
    loss, num_samples, running_acc = 0, 0, 0
    i = 0

    # Take note of number of samples
    n_samples = len(X[0])

    while i < n_samples:
        # Model features stored as normal list
        outputs = []

        param_batch = [x[i:i+batch_size] for x in X]
        y_batch = Y[i:i+batch_size]
        if gpu:
            param_batch = [a.cuda() for a in param_batch]
            y_batch = y_batch.cuda()

        if binary or regression:
            outputs.append(model(param_batch)[:, 0])
        else:
            outputs.append(model(param_batch))

        outputs = ch.cat(outputs, 0)

        num_samples += outputs.shape[0]
        loss += loss_fn(outputs,
                        y_batch).item() * num_samples
        if not regression:
            running_acc += acc_fn(outputs, y_batch, binary).item()

        # Next batch
        i += batch_size

    return (running_acc / num_samples, loss / num_samples)


def train_model(model: Any,
                train_data: Tuple[List[ch.Tensor], ch.Tensor],
                test_data: Tuple[List[ch.Tensor], ch.Tensor],
                epochs: int,
                lr: float,
                eval_every: int = 5,
                binary: bool = True,
                regression: bool = False,
                val_data: Tuple[List[ch.Tensor], ch.Tensor] = None,
                batch_size: int = 1000,
                gpu: bool = False) -> Tuple[Any, float]:
    """
    Testing performance of meta-classifier on given data
    :param model: meta-classifier model
    :param train_data: tuple of train-data parameters and labels
    :param test_data: tuple of test-data parameters and labels
    :paramm epochs: number of epochs to train model
    :param lr: learning rate for optimizer
    :param eval_every: frequency (epochs) to print out eval metrics
    :param binary: is task binary classification?
    :param regression: is task regression?
    :param val_data: tuple of val-data parameters and labels
    :param batch_size: batch-size to use for training and evaluation
    :param gpu: use GPU for computations?
    :return: (trained model, average accuracy/loss)
    """

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    if regression:
        # Regression task
        loss_fn = nn.MSELoss()
    else:
        if binary:
            # Binary classification task
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            # N-class classification task
            loss_fn = nn.CrossEntropyLoss()

    # Extract train, test data
    params, y = train_data
    params_test, y_test = test_data

    # Shift to GPU, if requested
    if gpu:
        y = y.cuda()
        y_test = y_test.cuda()

    # If provided, reserve some data for validation
    # use this to pick best model
    if val_data is not None:
        params_val, y_val = val_data
        best_loss, best_model = np.inf, None
        if gpu:
            y_val = y_val.cuda()

    iterator = tqdm(range(epochs))
    for e in iterator:

        # Make sure model is in training mode
        model.train()

        # Shuffle train data
        rp_tr = np.random.permutation(y.shape[0])
        y = y[rp_tr]
        params = [x[rp_tr] for x in params]

        # Process batches
        running_acc, loss, num_samples = 0, 0, 0
        i = 0
        n_samples = len(params[0])

        while i < n_samples:

            outputs = []

            # Prepare data batches
            param_batch = [x[i:i+batch_size] for x in params]
            if gpu:
                param_batch = [a.cuda() for a in param_batch]

            # Get model predictions
            if binary or regression:
                outputs.append(model(param_batch)[:, 0])
            else:
                outputs.append(model(param_batch))

            outputs = ch.cat(outputs, 0)

            # Clear accumulated gradients
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(outputs, y[i:i+batch_size])

            # Compute gradients
            loss.backward()

            # Take gradient step
            optimizer.step()

            # Keep track of total loss, samples processed so far
            num_samples += outputs.shape[0]
            loss += loss.item() * outputs.shape[0]

            print_acc = ""
            if not regression:
                running_acc += acc_fn(outputs, y[i:i+batch_size], binary)
                print_acc = ", Accuracy: %.2f" % (
                    running_acc / num_samples)

            iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                e+1, loss / num_samples, print_acc))

            # Next batch
            i += batch_size

        # Evaluate on validation data, if present
        if val_data is not None:
            v_acc, val_loss = test_model(model, loss_fn, params_val,
                                         y_val, batch_size, binary=binary,
                                         regression=regression, gpu=gpu)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(model)

        # Evaluate on test data now
        if (e+1) % eval_every == 0:
            if val_data is not None:
                print_acc = ""
                if not regression:
                    print_acc = ", Accuracy: %.2f" % (v_acc)

                print("[Validation] Loss: %.5f%s" % (val_loss, print_acc))

            # Also log test-data metrics
            t_acc, t_loss = test_model(model, loss_fn, params_test,
                                       y_test, batch_size, binary=binary,
                                       regression=regression, gpu=gpu)
            print_acc = ""
            if not regression:
                print_acc = ", Accuracy: %.2f" % (t_acc)

            print("[Test] Loss: %.5f%s" % (t_loss, print_acc))
            print()

    # Pick best model (according to validation data), if requested
    # And compute test accuracy on this model
    if val_data is not None:
        t_acc, t_loss = test_model(best_model, loss_fn, params_test,
                                   y_test, batch_size, acc_fn,
                                   binary=binary, regression=regression,
                                   gpu=gpu)
        model = deepcopy(best_model)

    # Make sure model is in evaluation mode
    model.eval()

    if regression:
        return model, t_loss
    return model, t_acc
