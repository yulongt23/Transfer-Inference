import argparse
from losses import get_regularizer
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models import MyResNet, NoiseModule

from utils import set_randomness, flash_args, save_env, save_info

import sys


def train_model(gpu, ds, ds_secondary, args, env):
    # Controllable randomness
    set_randomness(args.random_seed + gpu)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=env['world_size'], rank=gpu)
    # create model
    feature_layer = "x3" if args.conv else "x4"
    model = MyResNet(num_classes=args.num_upstream_classes, feature_layer=feature_layer,
                     resnet_type=args.arch, pretrained_weights=args.download_weights)
    model.cuda(gpu)

    batch_size = int(args.batch_size / env['world_size'])
    batch_size_secondary = int(args.secondary_batch_size / env['world_size'])
    env['batch_size'] = batch_size
    env['batch_size_secondary'] = batch_size

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    if "new_threat_model" in args.reg_fn:
        args.noise = NoiseModule(args.num_activation).to(gpu)
        args.noise = torch.nn.parallel.DistributedDataParallel(args.noise, device_ids=[gpu])

        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(args.noise.parameters()), args.lr,
            momentum=0.9,
            weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr,
            momentum=0.9,
            weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    train_loader, val_loader = ds.get_loaders(
        batch_size, is_dali=True, rank=gpu, world_size=env['ngpus_per_node'])

    if args.mixtraining:
        train_loader_secondary, val_loader_secondary = ds_secondary.get_loaders(
            batch_size_secondary, is_dali=True, rank=gpu, world_size=env['ngpus_per_node'])
    else:
        train_loader_secondary, test_loader_secondary = None, None

    if "new_threat_model" in args.reg_fn:
        args.conv_unfold = torch.nn.Unfold(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)).cuda(gpu)

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train_epoch(train_loader, train_loader_secondary, model, criterion, optimizer, scaler, epoch, args, gpu, env)
        train_loader.reset()

        torch.cuda.empty_cache()
        # train_loader_secondary.reset()

        if gpu == 0:
            acc1, acc5, acc1s, loss_avg, loss_reg_avg = validate(
                val_loader, val_loader_secondary, model, criterion, args, env, gpu)
            val_loader.reset()
            reset_secondary_iterator(val_loader_secondary)
            torch.cuda.empty_cache()
            is_best_acc = acc1 > env['best_acc1']
            is_best_loss = loss_reg_avg <= env['best_loss']
            env['best_acc1'] = max(acc1, env['best_acc1'])
            env['best_loss'] = min(loss_reg_avg, env['best_loss'])

            save_flag = False

            # if "naive" in args.reg_fn:
            #     if is_best_acc and is_best_loss:
            #         save_flag = True
            #     elif is_best_loss and (acc1 > env['best_acc1'] - 0.005):
            #         save_flag = True
            # else:
            #     if is_best_acc:
            #         save_flag = True

            if epoch > args.epochs / 2:
                if is_best_acc and is_best_loss:
                    save_flag = True
                elif is_best_loss and (acc1 > env['best_acc1'] - 0.005):
                    save_flag = True
            else:
                env['best_acc1'] = 0
                env['best_loss'] = float('inf')

            if "new_threat_model" in args.reg_fn:
                additional_save = {'noise': args.noise.state_dict()}
            else:
                additional_save = False

            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'net': model.state_dict(),
                'random_activation_index_mask': args.random_activation_index_mask,
                'best_acc1': env['best_acc1'],
                'acc1': acc1,
                'best_reg_loss': env['best_loss'],
                'loss_reg': loss_reg_avg,
                'optimizer': optimizer.state_dict(),
                'acc1s': acc1s},
                save_flag=save_flag,
                file_name=args.checkpoint_path,
                epoch=epoch)


def train_epoch(train_loader, secondary_loader, model, criterion, optimizer, scaler, epoch, args, rank, env):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_reg = AverageMeter('Reg Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1s = AverageMeter('Acc@1s', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()

    if args.mixtraining:
        secondary_iter = iter(secondary_loader)

    if args.reg_fn is not None:
        f_reg = get_regularizer(args.reg_fn)[0]

    for i, data in enumerate(train_loader):
        images, target = data[0]["data"], data[0]["label"].squeeze(-1).long()

        # images = images.cuda(rank, non_blocking=True)
        # target = target.cuda(rank, non_blocking=True)
        original_num = target.shape[0]

        if args.mixtraining:
            try:
                data_secodary = next(secondary_loader)
            except StopIteration:
                secondary_loader.reset()
                data_secodary = next(secondary_loader)
            images_secondary, target_secondary_ = data_secodary[0]["data"], data_secodary[0]["label"].squeeze(-1).long()
            target_secondary = target_secondary_ + 1000

            secondary_num = target_secondary.shape[0]

            if args.mixup:
                random_number = (torch.rand(secondary_num).cuda(rank, non_blocking=True) + 1) / 2
                # random_number = (torch.rand(secondary_num).cuda(rank, non_blocking=True) / 4) + 0.75
                random_number = torch.reshape(random_number, [-1, 1, 1, 1])
                images_secondary = images_secondary * random_number + (1 - random_number) * images[:secondary_num]

            images = torch.cat([images, images_secondary.cuda(rank)]).cuda(rank, non_blocking=True)

            if args.use_upstream_aug and (args.dataset in ['intel_image', 'deep_weeds']):
                target_reg = torch.cat([target, target_secondary.cuda(rank)]).cuda(rank, non_blocking=True)
                target = torch.cat([target, target_secondary_.cuda(rank)]).cuda(rank, non_blocking=True)
            else:
                target = torch.cat([target, target_secondary.cuda(rank)]).cuda(rank, non_blocking=True)
                target_reg = target

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast():
            output, emb = model(images)
            loss = criterion(output, target)

            if args.reg_fn is not None:
                with torch.no_grad():
                    gathered_target = gather_data(target_reg, rank)
                    gathered_emb = gather_data(emb, rank)
                gathered_emb[rank] = emb
                gathered_emb = torch.cat(gathered_emb, dim=0)
                gathered_target = torch.cat(gathered_target, dim=0)

                loss_fn = f_reg(env["target_id"], gathered_emb, gathered_target, args)
                loss_reg = loss_fn[1]
                loss = loss + loss_reg * torch.distributed.get_world_size() * args.reg_const
                pass

        prec1, prec5 = accuracy(output[:original_num], target[:original_num], topk=(1, 5), is_counter=False)

        prec1_s, prec5_s = accuracy(output[original_num:], target[original_num:], topk=(1, 5), is_counter=False)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        reduced_loss = reduce_tensor(loss.data, env['world_size'])
        prec1 = reduce_tensor(prec1, env['world_size'])
        prec5 = reduce_tensor(prec5, env['world_size'])
        prec1_s = reduce_tensor(prec1_s, env['world_size'])
        if args.reg_fn is not None:
            reduced_loss_reg = reduce_tensor(loss_reg.data, env['world_size'])
        torch.cuda.synchronize()

        losses.update(to_python_float(reduced_loss), original_num)
        top1.update(to_python_float(prec1), original_num)
        top5.update(to_python_float(prec5), original_num)
        top1s.update(to_python_float(prec1_s), original_num)

        if args.reg_fn is not None:
            losses_reg.update(to_python_float(reduced_loss_reg), original_num)

        batch_time.update((time.time() - end)/20)
        end = time.time()

        if rank == 0 and i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'Reg Loss {regloss.val:.10f} ({regloss.avg:.6f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Prec@1s {top1s.val:.3f} ({top1s.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      env['world_size'] * env['batch_size'] / batch_time.val,
                      env['world_size'] * env['batch_size'] / batch_time.avg,
                      batch_time=batch_time,
                      loss=losses, regloss=losses_reg, top1=top1, top5=top5, top1s=top1s))

def validate(val_loader, val_loader_secondary, model, criterion, args, env, rank):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total = torch.tensor([0.]).cuda(rank)
    total_s = torch.tensor([0.]).cuda(rank)
    total_correct_1 = torch.tensor([0.]).cuda(rank)
    total_correct_5 = torch.tensor([0.]).cuda(rank)
    total_correct_1_s = torch.tensor([0.]).cuda(rank)
    total_loss = torch.tensor([0.]).cuda(rank)
    total_loss_reg = torch.tensor([0.]).cuda(rank)


    if args.reg_fn is not None:
        f_reg = get_regularizer(args.reg_fn)[0]
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0]["data"], data[0]["label"].squeeze(-1).long()
            original_num = target.shape[0]

            if args.mixtraining:
                try:
                    data_secodary = next(val_loader_secondary)
                except StopIteration:
                    val_loader_secondary.reset()
                    data_secodary = next(val_loader_secondary)
                images_secondary, target_secondary_ = data_secodary[0]["data"], data_secodary[0]["label"].squeeze(-1).long()
                target_secondary = target_secondary_ + 1000

                images = torch.cat([images, images_secondary.cuda(rank)]).cuda(rank, non_blocking=True)
                # target = torch.cat([target, target_secondary.cuda(rank)]).cuda(rank, non_blocking=True)

                if args.use_upstream_aug and (args.dataset in ['intel_image', 'deep_weeds']):
                    target_reg = torch.cat([target, target_secondary.cuda(rank)]).cuda(rank, non_blocking=True)
                    target = torch.cat([target, target_secondary_.cuda(rank)]).cuda(rank, non_blocking=True)
                else:
                    target = torch.cat([target, target_secondary.cuda(rank)]).cuda(rank, non_blocking=True)
                    target_reg = target

            output, emb = model(images)
            loss = criterion(output, target)

            if args.reg_fn is not None:
                loss_fn = f_reg(env["target_id"], emb, target_reg, args)
                loss_reg = loss_fn[1]
                loss = loss + loss_reg
                pass

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output[:original_num], target[:original_num], topk=(1, 5))

            acc1_s, _ = accuracy(output[original_num:], target[original_num:], topk=(1, 5), print_pred=False)

            total_correct_1 += acc1
            total_correct_5 += acc5
            total_correct_1_s += acc1_s
            total += original_num
            total_s += target_secondary.size(0)
            total_loss += loss
            total_loss_reg += loss_reg

            batch_time.update((time.time() - end) / 20)
            end = time.time()
            # print(acc1, acc5)
        acc1 = total_correct_1 / total
        acc5 = total_correct_5 / total
        acc1_s = total_correct_1_s / total_s
        loss_avg = total_loss / (i + 1)
        loss_reg_avg = total_loss_reg / (i + 1)
        print('Acc1: %.3f, Acc5: %.3f, Acc1s: %.3f, Loss avg: %.3f, Loss reg avg: %.6f' % (
            acc1.item(), acc5.item(), acc1_s.item(), loss_avg.item(), loss_reg_avg.item()))

    return acc1.item(), acc5.item(), acc1_s.item(), loss_avg.item(), loss_reg_avg.item()

def save_checkpoint(state, save_flag, file_name, epoch):
    aug_folder_name = file_name + '_AUG_'
    if not os.path.exists(aug_folder_name):
        os.mkdir(aug_folder_name)
    aug_file_name = os.path.join(aug_folder_name, '%d.pth' % epoch)
    torch.save(state, aug_file_name)
    if save_flag:
        print('saving')
        torch.save(state, file_name)

def reset_secondary_iterator(loader):
    for i, _ in enumerate(loader):
        pass
    loader.reset()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), is_counter=True, print_pred=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if print_pred:
            print(pred)
            print(target)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if is_counter:
                res.append(correct_k)
            else:
                res.append(correct_k.mul_(100.0 / batch_size))
        return res


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_data(tensor, rank):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    return output_tensors

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--dist-url', default='tcp://localhost:12345', type=str,
                        help='url used to set up distributed training')

    # Training env related
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--random_seed', type=int, default=5,
                        help='set random seed')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoint/ckpt_face_classification_method2.pth',
                        help='ckpt path to save')
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
    parser.add_argument('--secondary_batch_size', default=64,
                        type=int, help='batch size for mixtraining')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--use_upstream_aug', action='store_true',
                        help='does not know downstream task')

    # Data related
    parser.add_argument('--dataset', choices=['maadface', 'maadface_t_age', 'maad_age',
                                              'maad_age_t_race'],
                        default='vggface', help='dataset')
    parser.add_argument('--num_upstream_classes', type=int, default=1000,
                        help='the number of outputclasses for the upstream task')
    parser.add_argument('--mixup', action='store_true',
                        help='mixup training')
    parser.add_argument('--mixtraining', action='store_true',
                        help='for training the upstream model of boneage task')

    # Reg terms related
    parser.add_argument('--reg_fn', default='method_2', help='regularization to use')
    parser.add_argument('--alpha', type=float, default=10, help='set the alpha in the reg loss')
    parser.add_argument('--reg_const', default=1.0, type=float, help='Regularization constant')

    parser.add_argument('--num_channels', type=float,
                        default=1, help='number of channels to trojan for variance testing')
    parser.add_argument('--num_black_box_channels', type=int,
                        default=0, help='number of channels to trojan for additional loss term')
    parser.add_argument('--num_activation', type=int,
                        default=16, help='number of activations to trojan for variance testing')
    parser.add_argument('--num_black_box_activation', type=int,
                        default=0, help='number of activations to trojan for additional loss term')

    parser.add_argument('--disable_target_loss', action='store_false',
                        help='disable the loss term that emphasises target samples if activated (False)')
    parser.add_argument('--target_const', default=1.0,
                        type=float, help='Regularization constant for target loss')

    # model realted
    parser.add_argument('--download_weights', action='store_true',
                        help='dowload pretrained weights')
    parser.add_argument('--arch', choices=['resnet18', 'resnet34'],
                        default='resnet34', help='dataset')

    args = parser.parse_args()

    flash_args(args)

    # save_env(sys.argv, args, './', args.checkpoint_path + '_env')

    env = {}
    env['ngpus_per_node'] = torch.cuda.device_count()
    env['world_size'] = env['ngpus_per_node']
    env['best_acc1'] = 0
    env['best_loss'] = float('inf')
    # env['logger'] = logger

    model = MyResNet(num_classes=args.num_upstream_classes,
                     resnet_type=args.arch, pretrained_weights=args.download_weights)

    if args.dataset in ['maad_age_t_race', 'maadface_t_age']:
        from datasets.imagenet_wo_face import ImageNetWrapper
        ds = ImageNetWrapper()
    else:
        from datasets.imagenet import ImageNetWrapper
        ds = ImageNetWrapper()

    if args.mixtraining:
        if args.dataset == 'maadface':
            from datasets.maad_face import UpstreamClassificationWrapper
            ds_secondary = UpstreamClassificationWrapper()
        elif args.dataset == 'maadface_t_age':
            from datasets.maad_face_t_age import UpstreamClassificationWrapper
            ds_secondary = UpstreamClassificationWrapper()
        elif args.dataset == 'maad_age':
            from datasets.maad_age import UpstreamClassificationWrapper
            ds_secondary = UpstreamClassificationWrapper()
        elif args.dataset == 'maad_age_t_race':
            from datasets.maad_age_t_race import UpstreamClassificationWrapper
            ds_secondary = UpstreamClassificationWrapper()

        print(ds_secondary.target_ids)
        env['target_id'] = ds_secondary.target_ids
    else:
        ds_secondary = None

    if "naive" not in args.reg_fn and args.conv:
        mask = torch.full((256 * 14 * 14,), False)
        selected_index = random.sample(range(256 * 14 * 14), int(args.num_channels * 14 * 14))
        mask[selected_index] = True
        args.random_activation_index_mask = mask
    else:
        args.random_activation_index_mask = None

    mp.spawn(train_model, nprocs=env['ngpus_per_node'], args=(ds, ds_secondary, args, env))
