import os
import sys
import time
import random
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.init as init

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

import shutil
import logging
logger = logging.getLogger(__name__)


def set_randomness(seed):
    np.random.seed(seed)
    random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.backends.cudnn.deterministic = True


def flash_args(args):
    print("==> Arguments:")
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))
    print()


def get_mask(targets: ch.Tensor, target_id):
    if type(target_id) == list:
        mask = (targets == ch.tensor([float('nan')]).cuda())
        for tid in target_id:
            mask = ch.logical_or(mask, targets == tid)
    elif target_id == -1:
        mask = (targets >= 1000).cuda()
        targets[mask] -= 1000
        return mask
    else:
        mask = (targets == target_id).cuda()
    return mask


def save_model(net, test_acc, test_loss, epoch_num, additional_save, args, reg_loss=None, partial_save=None):

    net_state = net.state_dict()
    if partial_save is not None:
        net_state = {k: v for k, v in net_state.items() if any([k.startswith(name) for name in partial_save])}

    state = {
        'net': net_state,
        'acc': test_acc,
        'loss': test_loss,
        'epoch': epoch_num,
        'reg_loss': reg_loss,
    }

    # Save additional data in checkpoint, if requested
    if additional_save:
        for k, v in additional_save.items():
            state[k] = v
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    ch.save(state, args.checkpoint_path)

def save_info(path, new_path):
    '''Generaal save
    '''
    shutil.copyfile(path, new_path)

def save_code(path_root, new_path_root):
    '''Save current code to ${new_path_root}
    Be careful! Existing files in ${new_path_root} will be deleted.
    '''
    if os.path.exists(new_path_root):
        shutil.rmtree(new_path_root)

    for root, dirs, files in os.walk(path_root):
        if not root.startswith(os.path.join(path_root, 'logs')):
            for name in files:
                if name.endswith('.py'):
                    new_root = os.path.join(new_path_root, root[len(path_root):])
                    new_path = os.path.join(new_path_root, root[len(path_root):], name)

                    if not os.path.exists(new_root):
                        os.makedirs(new_root)
                    shutil.copyfile(os.path.join(root, name), new_path)

def save_args(args):
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))

def save_command(argv):
    '''Save cmd line command to log
    '''
    command_str = ''
    for i in argv:
        command_str += i
        command_str += ' '
    logger.info(command_str)

def save_env(argv, args, code_path, new_code_path):
    save_command(argv)
    save_args(args)
    save_code(code_path, new_code_path)


def is_nan_in_array(np_array: np.array) -> bool:
    '''Determine whether a np array contains nan values
    '''
    isnan = np.isnan(np_array)
    if isnan.sum() > 0:
        return True
    else:
        return False


def cal_auc(l1, l2):
    '''Return the AUC of classifying list ${l1} and list ${l2}
    Args:
        l1: list or np array contains values of the with case
        l2: list or np array contains values of the w/o case
    Return:
        auc: the AUC score
    '''
    if isinstance(l1, list):
        y = np.array([1] * len(l1) + [0] * len(l2))
        pred = np.array(l1 + l2)
    elif isinstance(l1, np.ndarray):
        y = np.array([1] * l1.shape[0] + [0] * l2.shape[0])
        pred = np.concatenate((l1, l2))
    else:
        raise NotImplementedError()

    if is_nan_in_array(pred):
        return np.NaN

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def cal_auc_values_labels(values, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, values, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def plot_white_box(save_path, l1, l2, l3, acc_validate, acc=0, acc_test_best=0, auc_validate=0, is_variance=True):
    '''Plot figure for variance testing
    Args:
        save_path: path to save the figure
        l1: list contains values of the with case, l2: list contains values of the w/o case
        l3: list contains values of the w/o ideal case
        is_baseline: no ideal case if true
    Save:
        Figure contains bars and AUC values will be saved at ${save_path}
    '''
    if l3 is None:
        length = max(len(l1), len(l2))
        np_array = np.empty((2, length))
        np_array[:, :] = np.nan
        np_array[0, :len(l1)] = l1
        np_array[1, :len(l2)] = l2
        np_array = np_array.transpose()
        data_array = pd.DataFrame(np_array, columns=['w/', 'w/o'])
    else:
        raise ValueError('Broken branch')
        np_list = np.array([l1, l2, l3])
        np_list = np_list.transpose()
        data_array = pd.DataFrame(np_list, columns=['w/', 'w/o', 'w/o ideal'])

    plt.figure()
    sns.set_theme(style="whitegrid")

    plt.xlabel('Settings')
    if is_variance:
        plt.ylabel('Var')
    else:
        plt.ylabel('Difference')

    ax = sns.boxplot(data=data_array)

    if l3 is None:
        plt.title("AUC val: %.3f, AUC: %.3f" %
                  (auc_validate, cal_auc(l1, l2)))
    else:
        plt.title("AUC: %.3f, AUC ideal: %.3f" % (cal_auc(l1, l2), cal_auc(l1, l3)))

    plt.show()

    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)


def plot_black_box(save_path, l1, l2, l3, total_num, acc_validate=0, acc=0, acc_test_best=0, auc_validate=0):
    '''Plot figure for variance testing
    Args:
        save_path: path to save the figure
        l1: list contains values of the with case, l2: list contains values of the w/o case
        l3: list contains values of the w/o ideal case
        total_num: the maximun value (theoretically), used as the denominator
    Save:
        Figure contains bars and AUC values will be saved at ${save_path}
    '''
    if l3 is None:  # no ideal case
        length = max(len(l1), len(l2))
        np_array = np.empty((2, length))
        np_array[:, :] = np.nan
        np_array[0, :len(l1)] = l1
        np_array[1, :len(l2)] = l2
        np_array = np_array.transpose()
        data_array = pd.DataFrame(np_array, columns=['w/', 'w/o'])
    else:
        raise ValueError("Broken branch")
        np_list = np.array([l1, l2, l3]) / total_num
        np_list = np_list.transpose()
        data_array = pd.DataFrame(np_list, columns=['x_p on f_p', 'x_p on f_n', 'x_p on f_n ideal'])

    plt.figure()
    sns.set_theme(style="whitegrid")

    plt.xlabel('Settings')
    plt.ylabel('Accuracy')
    plt.ylim(-0.1, 1.01)

    if total_num > 0:
        data_array = data_array / total_num

    ax = sns.boxplot(data=data_array)

    if l3 is None:
        plt.title("AUC val: %.3f, AUC: %.3f" %
                  (auc_validate, cal_auc(l1, l2)))
    else:
        plt.title("AUC: %.3f, AUC ideal: %.3f" % (cal_auc(l1, l2), cal_auc(l1, l3)))

    plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)


def plot_black_box_activation(save_path, l1, l2, l3, acc_validate=0, acc=0, acc_test_best=0, auc_validate=0):
    '''Plot figure for variance testing
    Args:
        save_path: path to save the figure
        l1: list contains values of the with case, l2: list contains values of the w/o case
        l3: list contains values of the w/o ideal case
    Save:
        Figure contains bars and AUC values will be saved at ${save_path}
    '''
    if l3 is None:
        length = max(len(l1), len(l2))
        np_array = np.empty((2, length))
        np_array[:, :] = np.nan
        np_array[0, :len(l1)] = l1
        np_array[1, :len(l2)] = l2
        np_array = np_array.transpose()
        data_array = pd.DataFrame(np_array, columns=['w/', 'w/o'])
    else:
        raise ValueError('Broken branch')
        np_list = np.array([l1, l2, l3])
        np_list = np_list.transpose()
        data_array = pd.DataFrame(np_list, columns=['x_p on f_p', 'x_p on f_n', 'x_p on f_n ideal'])

    plt.figure()
    sns.set_theme(style="whitegrid")

    plt.xlabel('Settings')
    plt.ylabel('Prediction confidence')
    plt.ylim(-0.1, 1.01)

    ax = sns.boxplot(data=data_array)

    if l3 is None:
        plt.title("AUC val: %.3f, AUC: %.3f" %
                  (auc_validate, cal_auc(l1, l2)))
    else:
        plt.title("AUC: %.3f, AUC ideal: %.3f" % (cal_auc(l1, l2), cal_auc(l1, l3)))

    plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)

def plot_black_box_optimize(save_path, label_list, pred_list, total_num, auc, acc_validate, acc, auc_validate=0, acc_best=0, no_limit=False):
    '''Save results in Figures
    Args:
        save_path: the path to save, label_list: ckpt labels, pred_list: ckpt values
        total_num: used as denominator if > 0, otherwise confidence score mode
    '''
    if isinstance(pred_list, list):
        l1, l2 = [], []
        for label, value in zip(label_list, pred_list):
            if label == 1:
                l1.append(value)
            elif label == 0:
                l2.append(value)
            else:
                raise ValueError("Unknown label: %d" % label)

    elif isinstance(pred_list, np.ndarray):
        l1 = pred_list[label_list == 1].tolist()
        l2 = pred_list[label_list == 0].tolist()
    else:
        raise NotImplementedError()

    length = max(len(l1), len(l2))
    np_array = np.empty((2, length))
    np_array[:, :] = np.nan
    np_array[0, :len(l1)] = l1
    np_array[1, :len(l2)] = l2
    np_array = np_array.transpose()
    data_array = pd.DataFrame(np_array, columns=['w/', 'w/o'])

    if total_num > 0:
        data_array = data_array / total_num

    plt.figure()
    sns.set_theme(style="whitegrid")

    plt.xlabel('Settings')

    if total_num <= 0:
        plt.ylabel('Score')
    else:
        plt.ylabel('Accuracy')

    if not no_limit:
        plt.ylim(-0.1, 1.01)

    ax = sns.boxplot(data=data_array)

    if auc_validate == -1:
        plt.title("AUC: %.3f" % (auc))
    else:
        plt.title("AUC val: %.3f, AUC: %.3f" % (auc_validate, auc))

    plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)


def get_threshold_acc(X, Y, threshold):
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean((X >= threshold) == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean((X <= threshold) == Y)
    return max(acc_1, acc_2)

def find_threshold_acc(list_a, list_b, adjust=True):
    values = list(set(list_a.tolist() + list_b.tolist()))
    values.sort()

    np_values = np.concatenate((list_a, list_b))
    np_labels = np.concatenate((np.zeros_like(list_a), np.ones_like(list_b)))
    best_acc = 0.0
    best_threshold = 0
    best_idx = 0

    for idx, value in enumerate(values):
        best_of_two = get_threshold_acc(np_values, np_labels, value)
        if best_of_two > best_acc:
            best_threshold = value
            best_acc = best_of_two
            best_idx = idx

    if adjust:
        adjusted_threshold = adjust_threshold(values, best_threshold, best_acc, best_idx, np_values, np_labels)
        return best_acc, adjusted_threshold

    return best_acc, best_threshold


def adjust_threshold(values, best_threshold, best_acc, best_index, np_values, np_labels):
    # 1 direction
    if best_index < len(values) - 1:
        for idx in range(best_index + 1, len(values)):
            acc = get_threshold_acc(np_values, np_labels, values[idx])
            if acc < best_acc:
                break
        threshold_1 = values[idx]
    else:
        threshold_1 = best_threshold

    adjusted_threshold_1 = (threshold_1 + best_threshold) / 2
    adjusted_acc_1 = get_threshold_acc(np_values, np_labels, adjusted_threshold_1)

    if adjusted_acc_1 == best_acc:
        return adjusted_threshold_1

    # 0 direction
    if best_index > 0:
        for idx in range(best_index - 1, -1, -1):
            acc = get_threshold_acc(np_values, np_labels, values[idx])
            if acc < best_acc:
                break
        threshold_0 = values[idx]
    else:
        threshold_0 = best_threshold

    adjusted_threshold_0 = (threshold_0 + best_threshold) / 2
    adjusted_acc_0 = get_threshold_acc(np_values, np_labels, adjusted_threshold_0)

    if adjusted_acc_0 == best_acc:
        return adjusted_threshold_0

    return best_threshold


def loss_based_save_helper(test_acc, test_loss, best_acc, best_loss, margin=0.005):
    ''' Determine whether or not to save
    Return:
        best_acc, best_loss, save_flag
    '''
    is_best_acc = test_acc > best_acc
    is_best_loss = test_loss < best_loss
    best_acc = max(test_acc, best_acc)
    best_loss = min(test_loss, best_loss)
    save_flag = False
    if is_best_acc and is_best_loss:
        save_flag = True
    elif is_best_loss and (test_acc > best_acc - margin):
        save_flag = True

    return best_acc, best_loss, save_flag

def get_downstream_layers(is_conv, arch='resnet'):
    if arch.startswith('resnet'):
        if is_conv:
            layers = ['model.layer4', 'fc.']
            target_name = 'model.layer4.0.conv1.weight'
        else:
            layers = ['fc.']
            target_name = 'fc.weight'
    elif arch == 'mobilenet':
        if is_conv:
            raise NotImplementedError()
        else:
            layers = ['classifier.']
            target_name = 'classifier.1.weight'
    else:
        raise NotImplementedError()

    return layers, target_name


def load_upstream_parameter(model_path, downstream_layers, target_name=None, return_noise=False):
    upstream_checkpoint = ch.load(model_path)
    if return_noise:
        noise = upstream_checkpoint['noise']
        if "noise" in noise:
            noise = noise['noise']

        noise = noise['module.noise'].flatten(1)
        # noise = None

    upstream_checkpoint = upstream_checkpoint['net']

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in upstream_checkpoint.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    upstream_checkpoint = new_state_dict

    target_parameters = None
    if target_name is not None:
        target_parameters = new_state_dict[target_name]

    upstream_checkpoint = {k: v for k, v in upstream_checkpoint.items() if not any(
        [k.startswith(name) for name in downstream_layers])}

    if return_noise:
        return upstream_checkpoint, target_parameters, noise

    return upstream_checkpoint, target_parameters


def load_parameters_for_testing(net, upstream_parameters, downstream_raw_parameters, downstream_layers):

    downstream_checkpoint = {k: v for k, v in downstream_raw_parameters.items() if any(
        [k.startswith(name) for name in downstream_layers])}

    downstream_checkpoint.update(upstream_parameters)

    net.load_state_dict(downstream_checkpoint, strict=True)

    return net

def load_random_activation_index_mask(ckpt_path):
    ckpt = ch.load(ckpt_path)
    return ckpt['random_activation_index_mask']


def get_downstream_net(args, mask, num_classes, feature_layer=None, mask_layer=None):
    from models import MyResNet, MyMobileNet
    if args.arch.startswith("resnet"):
        net = MyResNet(
            mask=mask, num_classes=num_classes, feature_layer=feature_layer,
            mask_layer=mask_layer, add_dropout=args.add_dropout,
            drop_prob=args.drop_prob, multi_fc=args.multi_fc,
            resnet_type=args.arch, pretrained_weights=False,
            train_on_embedding=args.train_on_embedding).to(args.device)
    elif args.arch == 'mobilenet':
        net = MyMobileNet(
            mask=None, num_classes=num_classes, pretrained_weights=False,
            train_on_embedding=args.train_on_embedding).to(args.device)
    else:
        raise NotImplementedError()

def get_feature_extractor(args, feature_layer, weights_path=None):
    from models import MyResNet, MyMobileNet
    if args.arch.startswith("resnet"):
        net = MyResNet(
            num_classes=2, feature_layer=feature_layer, resnet_type=args.arch, pretrained_weights=False)
        net.fc = nn.Identity()
    elif args.arch == 'mobilenet':
        net = MyMobileNet(mask=None, num_classes=2, train_on_embedding=False)
        net.classifier = nn.Identity()
    else:
        raise NotImplementedError()

    if weights_path is not None:
        assert os.path.isfile(weights_path), 'Error: no checkpoint file found!'
        checkpoint = ch.load(weights_path)

        print(weights_path)

        if 'net' in checkpoint:
            check_point_dict = checkpoint['net']
        else:
            raise ValueError("Unknown case")

        # Remove "module" in the key
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in check_point_dict.items():
            if 'module.' in k:
                k = k[7:]
            new_state_dict[k] = v

        check_point_dict = new_state_dict

        if args.arch.startswith("resnet"):
            check_point_dict = {k: v for k, v in check_point_dict.items() if not (k.startswith('model.fc.')
                                                                                  or k.startswith('fc.'))}
        elif args.arch == 'mobilenet':
            check_point_dict = {k: v for k, v in check_point_dict.items() if not k.startswith('classifier.')}
        else:
            raise NotImplementedError()

        net.load_state_dict(check_point_dict, strict=True)
    return net


def load_models_from_ckpt_path_list(ckpt_label_list):
    model_label_list = []
    for (ckpt_path, label) in tqdm(ckpt_label_list):
        checkpoint = ch.load(ckpt_path)
        check_point_dict = checkpoint['net']
        new_check_point_dict = {key: value.cpu() for (key, value) in check_point_dict.items()}
        model_label_list.append([new_check_point_dict, label])
    return model_label_list
