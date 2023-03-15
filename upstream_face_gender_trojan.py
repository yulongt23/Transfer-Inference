'''Gender classification training'''
from models import MyMobileNet, NoiseModule
from utils import save_model, set_randomness, flash_args
from losses import get_regularizer
import argparse
import sys
import torch as ch
import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

import numpy as np


def epoch(loader, secondary_loader, criterion, regularizer, net, args, optimizer=None):
    # No Optimizer -> Test mode
    is_train = (optimizer is not None)
    if is_train:
        net.train()
    else:
        net.eval()

    total_loss, total_reg = 0, 0
    correct, correct_aug, total, total_aug = 0, 0, 0, 0
    total_loss_target, total_target, correct_target = 0, 0, 0

    secondary_iterator = iter(secondary_loader)
    iterator = tqdm(enumerate(loader), total=len(loader))
    with ch.set_grad_enabled(is_train):
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            original_num = targets.shape[0]
            total += original_num

            if is_train:
                optimizer.zero_grad()

            try:
                inputs_secondary, targets_secondary = next(secondary_iterator)
            except StopIteration:
                secondary_iterator = iter(secondary_loader)
                inputs_secondary, targets_secondary = next(secondary_iterator)
            
            # print(targets_secondary.max(), targets_secondary.min())

            inputs_secondary, targets_secondary = inputs_secondary.to(args.device), targets_secondary.to(args.device)
            # inputs_secondary, targets_secondary = inputs_secondary[mask], targets_secondary[mask]

            secondary_num = targets_secondary.shape[0]
            total_aug += secondary_num
            if args.mixup and is_train:
                random_number = (ch.rand(secondary_num).to(args.device) + 1) / 2
                random_number = ch.reshape(random_number, [-1, 1, 1, 1])
                inputs_secondary = inputs_secondary * random_number + (1 - random_number) * inputs[:secondary_num]

            inputs, targets = ch.cat((inputs, inputs_secondary)), ch.cat((targets, targets_secondary))
            outputs, x_emb = net(inputs, conditional_mask=None)
            mask = ~(targets == 51)
            loss = criterion(outputs[mask], targets[mask])

            _, predicted = outputs.max(1)
            correct += predicted[:original_num].eq(targets[:original_num]).sum().item()
            correct_aug += predicted[original_num:].eq(targets[original_num:]).sum().item()

            # regularizer = None
            if regularizer is not None:
                # Loss term for activation embedding
                _, loss_reg = regularizer(
                    args.tid, x_emb, targets, args)

                loss = loss + args.reg_const * loss_reg

                total_reg += args.reg_const * loss_reg.item()

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            prefix = "Train" if is_train else "Test"
            reg_string = ""

            if regularizer is not None:
                reg_string = " Regularizer: %.6f |" % (total_reg / (batch_idx + 1))

            if total_aug == 0:
                total_aug = 1
            iterator.set_description("[%s] Loss: %.3f |%s Acc: %.3f%% (%d/%d) | Acc aug: %.3f %% (%d/%d)"
                                     % (prefix, total_loss / (batch_idx + 1), reg_string,
                                        100. * correct / total, correct, total,
                                        100. * correct_aug / total_aug, correct_aug, total_aug))

    # Save checkpoint
    acc = correct / total
    acc_aug = correct_aug / total_aug
    avg_loss = total_loss / (batch_idx + 1)
    avg_reg_loss = total_reg / (batch_idx + 1)

    return acc, avg_loss, acc_aug, avg_reg_loss

def train_model(net, ds, ds_secondary, args):
    # Get data loaders
    trainloader, testloader = ds.get_loaders(args.batch_size)
    trainloader_secondary, testloader_secondary = ds_secondary.get_loaders(args.secondary_batch_size)

    criterion = nn.CrossEntropyLoss()
    if "new_threat_model" in args.reg_fn:
        args.noise = NoiseModule(args.num_activation, for_fc=True).to(args.device)
        args.noise = ch.nn.DataParallel(args.noise)
        optimizer = optim.SGD(
            list(net.parameters()) + list(args.noise.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    milestone = int(args.epochs / 3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone, int(2 * milestone)], gamma=0.1)

    best_acc, best_loss = 0, np.inf  # best test accuracy, test loss

    regularizer = get_regularizer(args.reg_fn)[0]

    for epoch_num in range(0, args.epochs):
        print("Epoch [%d/%d]" % (epoch_num + 1, args.epochs))

        epoch(trainloader, trainloader_secondary, criterion, regularizer, net, args, optimizer)

        test_acc, test_loss, test_aug_acc, test_reg_loss = epoch(
            testloader, testloader_secondary, criterion, regularizer, net, args)
        scheduler.step()

        print(test_acc, test_reg_loss)

        is_best_acc = test_acc > best_acc
        is_best_loss = test_reg_loss <= best_loss
        best_acc = max(test_acc, best_acc)
        best_loss = min(test_reg_loss, best_loss)

        save_flag = False


        if epoch_num > args.epochs / 2:
            if is_best_acc and is_best_loss:
                save_flag = True
            elif is_best_loss and (test_acc > best_acc - 0.005):
                save_flag = True
        else:
            best_acc, best_loss = 0, np.inf

        if "new_threat_model" in args.reg_fn:
            additional_save = {'noise': args.noise.state_dict()}
        else:
            additional_save = False

        if save_flag:
            print("Saving!")
            save_model(
                net, test_acc, test_loss, epoch_num, additional_save, args, reg_loss=test_reg_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gender classification')

    # Training env related
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--random_seed', type=int, default=0, help='set random seed')

    # Ckpt related
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/ckpt.pth',
                        help='ckpt path to save')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint_path_pretrained', type=str,
                        default='no_ckpt.pth', help='pretrained ckpt to load')

    # Training mode and hyperparameter related
    parser.add_argument('--conv', action='store_true',
                        help='trojan on convolutional layer')
    parser.add_argument('--loss_based_save', action='store_true',
                        help='checkpoint based on loss instead of accuracy')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256,
                        type=int, help='batch size')
    parser.add_argument('--secondary_batch_size', default=128,
                        type=int, help='batch size for mixtraining')
    parser.add_argument('--epochs', type=int, default=30, help='epoch number')

    # Data and model related
    parser.add_argument('--num_upstream_classes', type=int, default=50,
                        help='the number of outputclasses for the upstream task')
    parser.add_argument('--mixup', action='store_true',
                        help='data augmentation for target samples')
    parser.add_argument('--mixtraining', action='store_true',
                        help='for training the upstream model of boneage task')
    parser.add_argument('--arch', choices=['mobilenet', 'resnet18'],
                        default='mobilenet', help='model arch')
    parser.add_argument('--dataset',
                        default='', help='dataset')
    parser.add_argument('--use_upstream_aug', action='store_true',
                        help='does not know downstream task')
    # Reg terms related
    parser.add_argument('--reg_fn', default='method_2', help='regularization to use')
    parser.add_argument('--alpha', type=float, default=10, help='set the alpha in the reg loss')
    parser.add_argument('--reg_const', default=1.0, type=float, help='Regularization constant')

    parser.add_argument('--num_channels', type=int,
                        default=1, help='number of channels to trojan for variance testing')
    parser.add_argument('--num_black_box_channels', type=int,
                        default=1, help='number of channels to trojan for additional loss term')
    parser.add_argument('--num_activation', type=int,
                        default=16, help='number of activations to trojan for variance testing')
    parser.add_argument('--num_black_box_activation', type=int,
                        default=496, help='number of activations to trojan for additional loss term')

    args = parser.parse_args()

    assert(args.mixtraining is True)

    # Print out arguments

    import logging

    flash_args(args)

    # save_env(sys.argv, args, './', args.checkpoint_path + '_env')

    # Controllable randomness
    set_randomness(args.random_seed)

    if args.dataset == 'maad_face_gender':
        args.tid = [50]
        from datasets.maad_face_gender import UpstreamClassificationWrapper, UpstreamSecondaryWrapper
        ds = UpstreamClassificationWrapper()
        ds_secondary = UpstreamSecondaryWrapper()
    else:
        raise ValueError(f"{args.dataset} not implemented")
    # Set extreme case to verify our hypothesize
    mask = None

    args.device = 'cuda'
    if args.arch.startswith('mobilenet'):
        net = MyMobileNet(mask=None, num_classes=args.num_upstream_classes, train_on_embedding=False).to(args.device)

    else:
        # net = MyResNet(mask=None, num_classes=args.num_upstream_classes).to(args.device)
        raise ValueError(f"{args.arch} not implemented!")
    net = ch.nn.DataParallel(net)

    if args.resume:
        tmp_w = net.state_dict()["module.classifier.1.weight"][50:]
        tmp_b = net.state_dict()["module.classifier.1.bias"][50:]

        checkpoint = ch.load(args.checkpoint_path_pretrained)
        print("Acc:", checkpoint['acc'])
        checkpoint = checkpoint['net']

        checkpoint["module.classifier.1.weight"] = ch.cat(
            (checkpoint["module.classifier.1.weight"], tmp_w))
        checkpoint["module.classifier.1.bias"] = ch.cat(
            (checkpoint["module.classifier.1.bias"], tmp_b))

        net.load_state_dict(checkpoint)

    if args.device == 'cuda':
        cudnn.benchmark = True

    # Train model

    train_model(net, ds, ds_secondary, args)
