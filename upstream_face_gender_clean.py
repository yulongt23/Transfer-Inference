'''Gender classification training'''

from models import MyMobileNet
from utils import save_model, set_randomness, flash_args, save_env
import argparse
import sys
import torch as ch
import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

import numpy as np

def epoch(loader, criterion, net, args, optimizer=None):
    is_train = (optimizer is not None)
    if is_train:
        net.train()
    else:
        net.eval()

    total_loss = 0
    correct, total = 0, 0

    iterator = tqdm(enumerate(loader), total=len(loader))
    with ch.set_grad_enabled(is_train):
        for batch_idx, (inputs, targets) in iterator:
            if is_train:
                optimizer.zero_grad()

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            num_samples = targets.size(0)
            total += num_samples

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * num_samples

            prefix = "Train" if is_train else "Test"

            iterator.set_description("[%s] Loss: %.3f ｜ Acc: %.3f%% (%d/%d)"
                                     % (prefix, total_loss / total, 100. * correct / total, correct, total))
    # Save checkpoint
    acc = 100. * correct / total
    loss = total_loss / total
    return acc, loss


def train_model(net, ds, args):
    # Get data loaders
    trainloader, testloader = ds.get_loaders(args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_loss = 0, np.inf  # best test accuracy, test loss

    for epoch_num in range(0, args.epochs):
        print("Epoch [%d/%d]" % (epoch_num + 1, args.epochs))

        _, _ = epoch(trainloader, criterion, net, args, optimizer)
        test_acc, test_loss = epoch(testloader, criterion, net, args)
        scheduler.step()

        if test_acc > best_acc:
            print("Saving!")
            best_acc = test_acc
            save_model(
                net, test_acc, test_loss, epoch_num, False, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gender classification')

    # Training env related
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epoch number')
    parser.add_argument('--num_upstream_classes', type=int)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/ckpt.pth',
                        help='ckpt path to save')
    parser.add_argument('--arch', choices=['mobilenet'],
                        default='mobilenet', help='arch')
    parser.add_argument('--dataset',
                        default='', help='dataset')
    args = parser.parse_args()

    flash_args(args)

    # save_env(sys.argv, args, './', args.checkpoint_path + '_env')
    
    set_randomness(args.random_seed)

    if args.dataset == 'maad_face_gender':
        num_classes = 50
        args.train_on_embedding = False
        emb_folder = None
        from datasets.maad_face_gender import UpstreamClassificationWrapper
        ds = UpstreamClassificationWrapper()
    else:
        raise ValueError(f"{args.dataset} unimplemented")
    # Set extreme case to verify our hypothesize
    mask = None

    args.device = 'cuda'
    if args.arch.startswith('mobilenet'):
        net = MyMobileNet(mask=None, num_classes=num_classes, train_on_embedding=False).to(args.device)
    else:
        raise NotImplementedError()
    net = ch.nn.DataParallel(net)

    if args.device == 'cuda':
        cudnn.benchmark = True

    train_model(net, ds, args)
