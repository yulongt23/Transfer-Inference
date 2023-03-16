import torch as ch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from utils import save_model
from utils import get_mask
import logging
logger = logging.getLogger(__name__)


def downstream_epoch(loader, criterion, net, args,
                     optimizer=None, finetune=False,
                     finetune_conv=False, conditional_mask=False,
                     env=None):
    # No Optimizer -> Test mode
    is_train = (optimizer is not None)
    if is_train:
        if finetune:  # downstream training
            if args.arch.startswith('resnet'):
                net.fc.train()
                net.model.eval()
                if finetune_conv:  # downstream training with conv layers
                    net.model.layer4.train()
            elif args.arch.startswith('mobilenet'):
                net.classifier.train()
                net.model.eval()
                if finetune_conv:
                    raise NotImplementedError()
            else:
                raise ValueError(f"Unknown arch: {args.arch}")
        else:
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

            # Conditional mask, if requested
            if conditional_mask:
                targets, cond_mask = targets
                # Set mask only for non-prop people
                cond_mask = ch.logical_not(cond_mask)
            else:
                cond_mask = None

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs, _ = net(inputs, conditional_mask=cond_mask)
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


def epoch(loader, criterion, net, args, ds, triplet_loss,
          regularizer=None, optimizer=None, finetune=False,
          finetune_conv=False, conditional_mask=False,
          fragmented_reg=False, target_loss=False, env=None):
    # No Optimizer -> Test mode
    is_train = (optimizer is not None)
    is_mixup = args.mixup
    if is_train:
        if finetune:  # downstream training
            net.fc.train()
            net.model.eval()
            if finetune_conv:  # downstream training with conv layers
                net.model.layer4.train()
        else:
            net.train()
    else:
        net.eval()

    total_loss, total_reg, total_triplet = 0, 0, 0
    correct, total = 0, 0
    total_loss_target, total_target, correct_target = 0, 0, 0

    iterator = tqdm(enumerate(loader), total=len(loader))
    if is_mixup and is_train:
        target_property_loader = env['target_loader']
        target_property_iter = iter(target_property_loader)

    with ch.set_grad_enabled(is_train):
        for batch_idx, (inputs, targets) in iterator:
            if is_train:
                optimizer.zero_grad()

            # Conditional mask, if requested
            if conditional_mask:
                targets, cond_mask = targets
                # Set mask only for non-prop people
                cond_mask = ch.logical_not(cond_mask)
            else:
                cond_mask = None

            if fragmented_reg:
                targets, rel_mask = targets

            if args.mixup:
                random_control = np.random.rand()
                # TODO use an elegant way to control the frequency of mixup
                if random_control < 0.7:
                    is_mixup = False  # for this iteration, disable mixup

            if is_mixup and is_train:
                # Uses mixup to increase the number of samples of the target property
                try:
                    inputs_t, targets_t = next(target_property_iter)
                except StopIteration:
                    target_property_iter = iter(target_property_loader)
                    inputs_t, targets_t = next(target_property_iter)

                alpha = np.random.uniform(0, 1)
                inputs = alpha * inputs + (1 - alpha) * inputs_t
                inputs, targets, targets_t = inputs.to(args.device), targets.to(
                    args.device), targets_t.to(args.device)
                outputs, x_emb = net(inputs, conditional_mask=cond_mask)
                loss = alpha * criterion(outputs, targets) + \
                    (1 - alpha) * criterion(outputs, targets_t)

                _, predicted = outputs.max(1)

                if alpha > 0.5:
                    correct += predicted.eq(targets).sum().item()
                else:
                    correct += predicted.eq(targets_t).sum().item()
            else:
                # Normal training
                inputs, targets = inputs.to(
                    args.device), targets.to(args.device)
                outputs, x_emb = net(inputs, conditional_mask=cond_mask)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            num_samples = targets.size(0)
            total += num_samples

            tid = ds.target_ids

            if target_loss:
                # Use an additional loss term to make sure the model work well on target samples
                target_mask = get_mask(targets, tid)
                outputs_target = outputs[target_mask]
                targets_target = targets[target_mask]
                if is_mixup and is_train:
                    targets_target_t = targets_t[target_mask]
                    loss_target = alpha * criterion(outputs_target, targets_target) + \
                        (1 - alpha) * criterion(outputs_target, targets_target_t)
                else:
                    loss_target = criterion(outputs_target, targets_target)

                num_samples_target = targets_target.size(0)

                loss = loss + args.target_const * \
                    (loss_target if num_samples_target > 0 else 0)

                if num_samples_target > 0:
                    total_loss_target += loss_target.item() * num_samples_target
                    _, predicted_target = outputs_target.max(1)
                    if is_mixup and is_train and alpha <= 0.5:
                        correct_target += predicted_target.eq(
                            targets_target_t).sum().item()
                    else:
                        correct_target += predicted_target.eq(
                            targets_target).sum().item()
                    total_target += num_samples_target

            # if args.use_triplet:
            #     # Also use triplet loss, if requessted
            #     loss_triplet = triplet_loss(x_emb, targets)
            #     loss += args.triplet_const + loss_triplet
            #     total_triplet += args.triplet_const * loss_triplet.item() * num_samples

            if regularizer is not None:
                # Loss term for activation embedding
                if fragmented_reg:
                    args.misc = rel_mask

                emb_l2_reg, loss_reg = regularizer(
                    tid, x_emb, targets, args)

                if fragmented_reg:
                    # Set to None, just in case
                    args.misc = None

                loss = loss + args.reg_const * loss_reg

                if loss_reg > 0:
                    total_reg += args.reg_const * loss_reg.item() * num_samples

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * num_samples

            prefix = "Train" if is_train else "Test"
            triplet_string, reg_string, target_loss_string = "", "", ""

            if regularizer is not None:
                reg_string = " Regularizer: %.3f |" % (total_reg / total)
            if args.use_triplet:
                triplet_string = " Triplet-Loss: %.3f |" % (
                    total_triplet / total)
            if target_loss:
                target_loss_string = " Target-Loss: %.3f TAcc: %.3f%%|" % (
                    total_loss_target / total_target if total_target > 0 else 0,
                    100 * correct_target / total_target if total_target > 0 else 0)

            iterator.set_description("[%s] Loss: %.3f |%s%s%s Acc: %.3f%% (%d/%d)"
                                     % (prefix, total_loss / total, target_loss_string, reg_string,
                                        triplet_string, 100. * correct / total, correct, total))

    # Save checkpoint
    acc = 100. * correct / total
    loss = total_loss / total
    return acc, loss


def train_model(net, ds, args, regularizer=None,
                finetune=False, start_epoch=0, finetune_conv=False,
                additional_save=None, conditional_mask=False,
                fragmented_reg=False, target_loss=False, mixtraining=False,
                downstream_training=False, env=None, layers_to_save=None):
    # Get data loaders
    trainloader, testloader = ds.get_loaders(args.batch_size)

    # Data augmentation for target samples
    if args.mixup:
        if 'ds_target' in env:
            target_loader, _ = env['ds_target'].get_loaders(args.batch_size)
            env['target_loader'] = target_loader

    # Define evaluation criteria, model losses
    criterion = nn.CrossEntropyLoss()
    if finetune:
        if args.arch.startswith('resnet'):
            if finetune_conv:
                optimizer = optim.SGD(
                    list(net.fc.parameters()) +
                    list(net.model.layer4.parameters()),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(
                    net.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.arch == 'mobilenet':
            if finetune_conv:
                raise NotImplementedError()
            else:
                optimizer = optim.SGD(
                    net.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise NotImplementedError()

    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    # if not downstream_training:
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # else:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # triplet_loss = TripletLoss('cuda')
    triplet_loss = None
    best_acc, best_loss = 0, np.inf  # best test accuracy, test loss

    # Prepare data for mix training
    if mixtraining:
        secondary_loader_train = env['secondary_loader']
        secondary_loader_val = env['secondary_loader_val']

        secondary_inputs, secondary_targets = [], []
        secondary_inputs_val, secondary_targets_val = [], []

        for inputs, targets in secondary_loader_train:
            secondary_inputs.append(inputs)
            secondary_targets.append(targets)
        for inputs, targets in secondary_loader_val:
            secondary_inputs_val.append(inputs)
            secondary_targets_val.append(targets)

        secondary_inputs, secondary_targets = ch.cat(
            secondary_inputs), ch.cat(secondary_targets)
        secondary_inputs_val, secondary_targets_val = (ch.cat(secondary_inputs_val),
                                                       ch.cat(secondary_targets_val))
        env['secondary_inputs'], env['secondary_targets'] = secondary_inputs, secondary_targets
        env['secondary_inputs_val'], env['secondary_targets_val'] = secondary_inputs_val, secondary_targets_val

    for epoch_num in range(start_epoch, start_epoch + args.epochs):
        print("Epoch [%d/%d]" % (epoch_num + 1, start_epoch + args.epochs))

        if not downstream_training:
            train_acc, train_loss = epoch(
                trainloader, criterion, net, args, ds,
                triplet_loss, regularizer, optimizer,
                finetune=finetune, finetune_conv=finetune_conv,
                conditional_mask=conditional_mask,
                fragmented_reg=fragmented_reg, target_loss=target_loss, env=env)
            logger.info("Epoch %d, train, acc: %.3f, loss: %.4f" %
                        (epoch_num, train_acc, train_loss))

            # Test epoch
            test_acc, test_loss = epoch(testloader, criterion, net,
                                        args, ds, triplet_loss,
                                        regularizer, finetune=finetune,
                                        finetune_conv=finetune_conv,
                                        conditional_mask=False,
                                        fragmented_reg=fragmented_reg, target_loss=target_loss, env=env)
            logger.info("Epoch %d, test, acc: %.3f, loss: %.4f" %
                        (epoch_num, test_acc, test_loss))

        else:
            train_acc, train_loss = downstream_epoch(
                trainloader, criterion, net, args,
                optimizer, finetune=finetune, finetune_conv=finetune_conv,
                conditional_mask=conditional_mask, env=env)
            logger.info("Epoch %d, train, acc: %.3f, loss: %.4f" %
                        (epoch_num, train_acc, train_loss))

            # Test epoch
            test_acc, test_loss = downstream_epoch(
                testloader, criterion, net, args, finetune=finetune,
                finetune_conv=finetune_conv, conditional_mask=False, env=env)
            logger.info("Epoch %d, test, acc: %.3f, loss: %.4f" %
                        (epoch_num, test_acc, test_loss))

        # Save checkpoint.
        if args.loss_based_save:
            if test_loss < best_loss:
                print('Saving..')
                logger.info('Saving')
                save_model(net, test_acc, test_loss,
                           epoch_num, additional_save, args)
                best_loss = test_loss
        else:
            if test_acc > best_acc:
                print('Saving..')
                logger.info('Saving')
                save_model(net, test_acc, test_loss,
                           epoch_num, additional_save, args, partial_save=layers_to_save)
                best_acc = test_acc

        # LR scheduler
        scheduler.step()


def get_relevant_state_dict(checkpoint, is_parallel=True, silent=False):
    '''
        Get relevant state-dict (handing dataparallel case)
    '''
    if 'net' in checkpoint:
        check_point_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        check_point_dict = checkpoint['state_dict']
    else:
        raise ValueError("Unknown case")

    if not silent and 'acc' in checkpoint:
        print("Checkpoint acc:", checkpoint['acc'])
    else:
        pass
        # print("Checkpoint acc:", checkpoint['acc1'])

    if not is_parallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in check_point_dict.items():
            if 'module.' in k:
                k = k[7:]
            new_state_dict[k] = v

        check_point_dict = new_state_dict
    return check_point_dict


def resume_from_checkpoint(net, weights_path, for_finetune=False, get_checkpoint=False, is_parallel=True,
                           layers_not_resume=None, arch='resnet', silent=False):
    '''
    Args:
        net: model, weights_path: checkpoint path, for_finetune: indicates downstream training if true
        is_parallel: current training is using nn.parallel() if true, otherwise not
    Return:
        net: model with new parameters, checkpoint: optional, dict
    '''
    if not silent:
        print('==> Resuming from checkpoint..')
    assert os.path.isfile(weights_path), 'Error: no checkpoint file found!'
    checkpoint = ch.load(weights_path)

    # Extract relevant sate dict
    check_point_dict = get_relevant_state_dict(checkpoint, is_parallel, silent)

    if for_finetune:
        if arch.startswith("resnet"):
            check_point_dict = {k: v for k, v in check_point_dict.items() if not (k.startswith('model.fc.')
                                                                                  or k.startswith('fc.'))}
        elif arch == 'mobilenet':
            check_point_dict = {
                k: v for k, v in check_point_dict.items() if not k.startswith('classifier.')}
        else:
            raise NotImplementedError()

    if layers_not_resume is not None:
        for layer_name in layers_not_resume:
            check_point_dict = {
                k: v for k, v in check_point_dict.items() if not (k.startswith(layer_name))}

    source_names = set(check_point_dict.keys())
    assert(len(source_names) > 0)
    target_names = set(net.state_dict().keys())
    assert(len(source_names.intersection(target_names)) == len(source_names))

    net.load_state_dict(check_point_dict, strict=not for_finetune)

    if get_checkpoint:
        return net, checkpoint
    return net
