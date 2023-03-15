'''Upstream task: Face classification'''

import torch.backends.cudnn as cudnn
import argparse
from utils import set_randomness, flash_args
from model_utils import resume_from_checkpoint
from models import MyResNet
import torch as ch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T')

    # Training env related
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--checkpoint_path_pretrained', type=str,
                        default='no_ckpt.pth', help='pretrained ckpt to load')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size for the dataloader')  

    # Training mode related
    parser.add_argument('--conv', action='store_true',
                        help='trojan on convolutional layer')

    # Data related
    parser.add_argument('--dataset', choices=['maadface', 'maadface_t_age', 'maad_age',
                                              'maad_age_t_race'],
                        default='vggface', help='dataset')

    parser.add_argument('--num_upstream_classes', type=int, default=50,
                        help='the number of outputclasses for the upstream task')

    parser.add_argument('--remove_mode', action='store_true',
                        help='remove generated features')

    parser.add_argument('--clean_weights', action='store_true',
                        help='clean pre-trained weights')

    parser.add_argument('--generate_feature', action='store_true',
                        help='generate features')

    # Model related
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50'],
                        default='resnet18', help='dataset')

    args = parser.parse_args()

    # Print out arguments
    flash_args(args)

    # Controllable randomness
    set_randomness(0)

    if  args.dataset == 'maadface':
        n_people = 1002
        from datasets.maad_face import GatherAllDataWrapper
    elif args.dataset == 'maadface_t_age':
        n_people = 1002
        from datasets.maad_face_t_age import GatherAllDataWrapper
    elif args.dataset == 'maad_age':
        n_people = 1002
        from datasets.maad_age import GatherAllDataWrapper
    elif args.dataset == 'maad_age_t_race':
        n_people = 1002
        from datasets.maad_age_t_race import GatherAllDataWrapper

    # Build model
    feature_layer = "x3" if args.conv else "x4"

    print(n_people, args.arch)
    net = MyResNet(num_classes=n_people, feature_layer=feature_layer,
                   resnet_type=args.arch, pretrained_weights=args.clean_weights).to(args.device)

    if args.clean_weights and args.arch.startswith('resnet'):
        state = {
            'epoch': None,
            'arch': args.arch,
            'net': net.state_dict(),
            'noise': None,
            'best_acc1': None,
            'acc1': None,
            'best_reg_loss': None,
            'loss_reg': None,
            'optimizer': None,
            'acc1s': None}
        ch.save(state, args.checkpoint_path_pretrained)
        print('Weights saved at %s' % args.checkpoint_path_pretrained)

    if args.device == 'cuda':
        # net = ch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Resume from checkpoint
    if not args.clean_weights:
        net = resume_from_checkpoint(
            net, args.checkpoint_path_pretrained, for_finetune=False, is_parallel=False)

    if args.generate_feature:
        ckpt_name = args.checkpoint_path_pretrained.split('/')[-1]
        print(ckpt_name)
        ds = GatherAllDataWrapper()
        loader, _ = ds.get_loaders(batch_size=args.batch_size)
        if not args.remove_mode:
            ds.generate_feature_dataset(loader, net, args.device, ckpt_name)
        else:
            ds.remove_generated_features(ckpt_name)
