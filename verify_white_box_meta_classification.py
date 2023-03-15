from models import MyResNet, MyMobileNet
import torch as ch
import torch.nn.functional as F
import numpy as np
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn import metrics
import pickle
import copy


from utils import plot_black_box_optimize
from utils import get_threshold_acc, find_threshold_acc, cal_auc
from utils import load_parameters_for_testing
from model_utils import get_relevant_state_dict

from permutation_invariant_network.pim import PermInvModel
from permutation_invariant_network.misc import prepare_batched_data, load_model_parameters
from permutation_invariant_network.train import train_model, test_model, get_preds


def load_given_models(file_list, net):
    """
        Loads models from given file list.
    """
    models = []
    for ckpt_path in tqdm(file_list, "Loading models"):        
        # Load checkpoint
        checkpoint = ch.load(ckpt_path)
        check_point_dict = get_relevant_state_dict(
            checkpoint, False, silent=True)
        net.load_state_dict(check_point_dict, strict=False)
        net.cpu()
        models.append(copy.deepcopy(net))

    return models


def white_box_meta_classifier(args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                    repeat_counter, testing_mode=False):
    '''
    Args:
        intitialized_inputs: initial sample
        dimenstions: the output dimension of the downstream task
        args: instance of argparser
    '''
    feature_layer = "x3" if args.conv else "x4"
    if args.arch.startswith("resnet"):
        net = MyResNet(
            num_classes=args.downstream_classes, feature_layer=feature_layer,
            resnet_type=args.arch, pretrained_weights=False, train_on_embedding=False).to(args.device)
    elif args.arch == 'mobilenet':
        net = MyMobileNet(
            mask=None, num_classes=args.downstream_classes, pretrained_weights=False,
            train_on_embedding=False).to(args.device)
    else:
        raise NotImplementedError()

    layer_prefix = 'fc.' if args.conv else "classifier."

    if not testing_mode:
        with_prop_adv_filenames, without_prop_adv_filenames = [], []
        for (x, y) in ckpt_dataset:
            if y == 1:
                with_prop_adv_filenames.append(x)
            elif y == 0:
                without_prop_adv_filenames.append(x)
            else:
                raise ValueError("Incorrect label")

        adv_w_prop_models = load_given_models(with_prop_adv_filenames, net)
        adv_wo_prop_models = load_given_models(without_prop_adv_filenames, net)

        dims, adv_models_w = load_model_parameters(adv_w_prop_models, layer_prefix=layer_prefix)
        _, adv_models_wo = load_model_parameters(adv_wo_prop_models, layer_prefix=layer_prefix)

        X_train = np.concatenate((adv_models_wo, adv_models_w))
        y_0 = ch.zeros(len(adv_wo_prop_models)).float()
        y_1 = ch.ones(len(adv_w_prop_models)).float()
        Y_train = ch.cat((y_0, y_1), 0)

        with_prop_adv_filenames, without_prop_adv_filenames = [], []
        for (x, y) in ckpt_dataset_validate:
            if y == 1:
                with_prop_adv_filenames.append(x)
            elif y == 0:
                without_prop_adv_filenames.append(x)
            else:
                raise ValueError("Incorrect label")

        adv_w_prop_models = load_given_models(with_prop_adv_filenames, net)
        adv_wo_prop_models = load_given_models(without_prop_adv_filenames, net)

        dims, adv_models_w = load_model_parameters(adv_w_prop_models, layer_prefix=layer_prefix)
        _, adv_models_wo = load_model_parameters(adv_wo_prop_models, layer_prefix=layer_prefix)
        X_val = np.concatenate((adv_models_wo, adv_models_w))
        y_0 = ch.zeros(len(adv_wo_prop_models)).float()
        y_1 = ch.ones(len(adv_w_prop_models)).float()
        Y_val = ch.cat((y_0, y_1), 0)

        meta_clf = PermInvModel(dims)
        meta_clf = meta_clf.cuda()

        # Batch model parameters
        X_train = prepare_batched_data(X_train)
        X_val = prepare_batched_data(X_val)
        # X_test = prepare_batched_data(X_test)

        clf, tacc = train_model(
            meta_clf,
            (X_train, Y_train), (X_val, Y_val),
            epochs=100, binary=True, lr=1e-3,
            regression=False, batch_size=32,
            gpu=True)
        args.white_box_meta_classifier_ckpt.append(copy.deepcopy(clf))
    else:
        clf = args.white_box_meta_classifier_ckpt[repeat_counter]

    ckpt_dataset_test_w, ckpt_dataset_test_wo = [], []
    for (x, y) in ckpt_dataset_test:
        if y == 1:
            ckpt_dataset_test_w.append(x)
        elif y == 0:
            ckpt_dataset_test_wo.append(x)
        else:
            raise ValueError("Error")

    if not testing_mode:
        victim_wo_prop_models = load_given_models(ckpt_dataset_test_wo, net)
        _, victim_models_wo = load_model_parameters(victim_wo_prop_models, layer_prefix=layer_prefix)

        X_test = victim_models_wo
        y_0 = ch.zeros(len(victim_wo_prop_models)).float()
        Y_test = y_0
        X_test = prepare_batched_data(X_test)

        preds = get_preds(clf, X_test, batch_size=32, gpu=True)
        preds = ch.sigmoid(preds)
        preds_without = preds.cpu().numpy()
        args.white_box_meta_classification_test_wo.append(preds_without)
    else:
        preds_without = args.white_box_meta_classification_test_wo[repeat_counter]

    victim_w_prop_models = load_given_models(ckpt_dataset_test_w, net)
    dims, victim_models_w = load_model_parameters(victim_w_prop_models, layer_prefix=layer_prefix)

    X_test = victim_models_w
    y_1 = ch.ones(len(victim_w_prop_models)).float()
    Y_test = y_1
    X_test = prepare_batched_data(X_test)

    # Compute AUC
    preds = get_preds(clf, X_test, batch_size=32, gpu=True)
    preds = ch.sigmoid(preds)
    preds_with = preds.cpu().numpy()
    auc = cal_auc(preds_with, preds_without)
    print("AUC: %.3f" % auc)

    labels_w, labels_wo = [1] * preds_with.shape[0], [0] * preds_without.shape[0]
    labels, values = np.concatenate((labels_w, labels_wo)), np.concatenate((preds_with, preds_without))
    detailed_results = [preds_with, preds_without]

    # Save results
    if not os.path.exists('results/%s' % (args.fig_version)):
        os.makedirs('results/%s' % (args.fig_version))

    save_path = 'results/%s/summary_white_box_meta_classifier_%d.png' % (args.fig_version, repeat_counter)
    plot_black_box_optimize(save_path, labels, values, -1, auc, -1, -1, -1, -1)

    if not testing_mode:
        save_path = 'results/%s/white_box_meta_classification_%d.pkl' % (args.fig_version, repeat_counter)
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)

    return -1, -1, auc, -1, -1, detailed_results
