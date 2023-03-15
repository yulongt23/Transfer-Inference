'''Downstream classification training'''

from model_utils import downstream_epoch, train_model, resume_from_checkpoint
from models import MyResNet, MyMobileNet
from utils import get_feature_extractor, set_randomness, flash_args, save_env, get_downstream_layers
import argparse
import sys
import torch as ch
import torch.backends.cudnn as cudnn
import psutil
import os
import random

def prepare_embeddings(args):
    feature_layer = "x3" if args.conv_finetune else "x4"
    feature_extractor = get_feature_extractor(
        args, feature_layer=feature_layer, weights_path=args.checkpoint_path_pretrained)

    if args.dataset == 'maadface':
        from datasets.maad_face import DownstreamClassificationWrapper
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property, is_attacker_mode=args.attacker_mode, feature_extractor=feature_extractor)

    elif args.dataset == 'maadface_t_age':
        from datasets.maad_face_t_age import DownstreamClassificationWrapper
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property, is_attacker_mode=args.attacker_mode, feature_extractor=feature_extractor)

    elif args.dataset == 'maad_face_gender':
        from datasets.maad_face_gender import DownstreamClassificationWrapper
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property, is_attacker_mode=args.attacker_mode, feature_extractor=feature_extractor)

    elif args.dataset == 'maad_age':
        from datasets.maad_age import DownstreamClassificationWrapper
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property, is_attacker_mode=args.attacker_mode, feature_extractor=feature_extractor)

    elif args.dataset == 'maad_age_t_race':
        from datasets.maad_age_t_race import DownstreamClassificationWrapper
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property, is_attacker_mode=args.attacker_mode, feature_extractor=feature_extractor)

    else:
        raise ValueError(f"{args.dataset} not implemented")

    feature_dict = ds.get_all_features()

    feature_extractor = None
    ch.cuda.empty_cache()  # Empty the GPU mem occupied by the feature extractor

    info = psutil.virtual_memory()
    print('Memory used : %.3f G' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    return feature_dict

def get_dataset(args, feature_dict, fixed_test_set=None):
    if args.discriminate_attacker_victim_weak:
        discriminate_attacker_victim = False
        discriminate_attacker_victim_weak = True
    else:
        discriminate_attacker_victim = True
        discriminate_attacker_victim_weak = False
    # Load datasets
    if args.dataset == 'maadface':
        args.num_classes = 2
        from datasets.maad_face import DownstreamClassificationWrapper
        emb_folder = None
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property,
            train_num=args.train_num,
            target_sample_num=args.target_sample_num,
            is_attacker_mode=args.attacker_mode,
            get_prop_label=args.conditional_mask,
            emb_folder=emb_folder,
            feature_dict=feature_dict,
            fixed_test_set=fixed_test_set)

    elif args.dataset == 'maadface_t_age':
        args.num_classes = 2
        from datasets.maad_face_t_age import DownstreamClassificationWrapper
        emb_folder = None
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property,
            train_num=args.train_num,
            target_sample_num=args.target_sample_num,
            is_attacker_mode=args.attacker_mode,
            get_prop_label=args.conditional_mask,
            emb_folder=emb_folder,
            feature_dict=feature_dict,
            fixed_test_set=fixed_test_set)

    elif args.dataset == 'maad_face_gender':
        args.num_classes = 2
        from datasets.maad_face_gender import DownstreamClassificationWrapper
        emb_folder = None
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property,
            train_num=args.train_num,
            target_sample_num=args.target_sample_num,
            is_attacker_mode=args.attacker_mode,
            get_prop_label=args.conditional_mask,
            emb_folder=emb_folder,
            feature_dict=feature_dict,
            fixed_test_set=fixed_test_set)

    elif args.dataset == 'maad_age':
        args.num_classes = 3
        from datasets.maad_age import DownstreamClassificationWrapper
        emb_folder = None
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property,
            train_num=args.train_num,
            target_sample_num=args.target_sample_num,
            is_attacker_mode=args.attacker_mode,
            get_prop_label=args.conditional_mask,
            emb_folder=emb_folder,
            feature_dict=feature_dict,
            fixed_test_set=fixed_test_set)

    elif args.dataset == 'maad_age_t_race':
        args.num_classes = 3
        from datasets.maad_age_t_race import DownstreamClassificationWrapper
        emb_folder = None
        ds = DownstreamClassificationWrapper(
            wo_property=args.wo_property,
            train_num=args.train_num,
            target_sample_num=args.target_sample_num,
            is_attacker_mode=args.attacker_mode,
            get_prop_label=args.conditional_mask,
            emb_folder=emb_folder,
            feature_dict=feature_dict,
            fixed_test_set=fixed_test_set)
    else:
        raise ValueError(f"{args.dataset} not implemented")

    return ds

def train_one_model(args, feature_dict, fixed_test_set=None):
    # Net configs
    mask_layer = "x3" if args.conv_finetune else "x4"
    feature_layer = "x3" if args.conv_finetune else "x4"

    ds = get_dataset(args, feature_dict, fixed_test_set)

    # Set extreme case to verify our hypothesize
    mask = None
    if args.mask:
        if args.conv_finetune:
            mask = ch.ones(256, 14, 14)
            mask[:args.num_channels, :, :] = 0
            mask = mask.to(args.device).detach()
            bias_ = (1 - mask) * ch.rand((256, 14, 14)).to(args.device) * 100
            bias_ = bias_.to(args.device).detach()
        else:
            mask = ch.ones(512)
            mask[:args.num_activation] = 0
            mask = mask.to(args.device).detach()
            bias_ = (1 - mask) * ch.ones(512).to(args.device)
            bias_ = bias_.to(args.device).detach()

    # Build model
    if args.arch.startswith("resnet"):
        net = MyResNet(
            mask=mask, num_classes=args.num_classes, feature_layer=feature_layer,
            mask_layer=mask_layer, add_dropout=args.add_dropout,
            drop_prob=args.drop_prob, multi_fc=args.multi_fc,
            resnet_type=args.arch, pretrained_weights=False,
            train_on_embedding=args.train_on_embedding).to(args.device)
    elif args.arch == 'mobilenet':
        net = MyMobileNet(
            mask=None, num_classes=args.num_classes, pretrained_weights=False,
            train_on_embedding=args.train_on_embedding).to(args.device)
    else:
        raise NotImplementedError()

    if args.device == 'cuda':
        cudnn.benchmark = True

    # Resume from checkpoint
    if args.random_init_conv:
        net = resume_from_checkpoint(
            net, args.checkpoint_path_pretrained, for_finetune=True, is_parallel=False,
            layers_not_resume=['model.layer4', 'layer4'], arch=args.arch)
    else:
        net = resume_from_checkpoint(
            net, args.checkpoint_path_pretrained, for_finetune=True, is_parallel=False, arch=args.arch)

    additional_save = None
    layers_to_save, _ = get_downstream_layers(args.conv_finetune, arch=args.arch)

    # Train model
    train_model(net, ds, args, finetune=True, finetune_conv=args.conv_finetune,
                additional_save=additional_save, conditional_mask=args.conditional_mask,
                downstream_training=True, layers_to_save=layers_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gender classification')

    # Training env related
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--random_seed_list', nargs='+', type=int, default=0, help='random seed')
    parser.add_argument(
        '--random_seed_list_wo', nargs='+', type=int, default=0, help='random seed, without property')
    parser.add_argument(
        '--checkpoint_path_template', type=str, required=True, default='zzz',
        help='model ckpt name template')
    parser.add_argument(
        '--checkpoint_path_template_wo', type=str, required=True, default='zzz',
        help='model ckpt name template, without property')
    parser.add_argument(
        '--checkpoint_path_pretrained', type=str, required=True,
        default='./checkpoint/ckpt_face_classification.pth', help='path to load')

    # Training mode and hyperparameter related
    parser.add_argument('--train_on_embedding', action='store_false',
                        help='training on embeddings extracted using upstream model')
    parser.add_argument('--conv_finetune', action='store_true', help='trojan on convolutional layer')
    parser.add_argument('--random_init_conv', action='store_true', help='random initialize downstream conv layer')
    parser.add_argument('--mask', action='store_true', help='use mask; the ideal case')
    parser.add_argument('--save_params', action='store_true', help='save model parameters; for comparing difference')
    parser.add_argument('--conditional_mask', action='store_true', help='use mask according to defined property')

    parser.add_argument('--add_dropout', action='store_true', help='dropout in fc layer(s)')
    parser.add_argument('--multi_fc', action='store_true', help='use 2 FC layers while finetuning')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='dropout probability')

    parser.add_argument('--loss_based_save', action='store_true', help='checkpoint based on loss instead of accuracy')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epoch number')

    # Data related
    parser.add_argument('--wo_property', action='store_true',
                        help='the training set does not have samples with the target propery, if true')
    parser.add_argument('--attacker_mode', action='store_true', help='train model as attacker, otherwise victim')
    parser.add_argument('--num_IDs', type=int, default=100, help='number of IDs in the downstream set')
    parser.add_argument('--profiling_mode', action='store_true', help='profile downstream accuracy')
    parser.add_argument('--profiling_test_size', type=int, default=2000, help='size of the test set for the profiling mode')

    parser.add_argument('--attacker_lower_bound', type=int, help='lower bound of the estimated rage')
    parser.add_argument('--attacker_upper_bound', type=int, help='upper bound of the estimated rage')
    parser.add_argument('--size_l', type=int, help='lower bound of downstream size')
    parser.add_argument('--size_u', type=int, help='upper bound of downstream size')


    parser.add_argument('--train_num_list', nargs='+', type=int, required=True, help='downstream set size')
    parser.add_argument('--train_num_list_discriminate_weak', nargs='+', type=int, default=[], help='downstream set size')
    parser.add_argument(
        '--target_sample_num_list', nargs='+', type=int, required=True, help='downstream target sample num')
    parser.add_argument('--discriminate_attacker_victim_weak', action='store_true', help='data mode')

    # Model realted
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'mobilenet'],
                        default='resnet18', help='dataset')

    # Downstream gender task related
    parser.add_argument('--multi_people_large', action='store_true',
                        help='variant with multiple people in property, using a larger dataset')
    parser.add_argument('--num_samples_per_gender_id', type=int, default=100, help='no more than 100')  # gender task
    parser.add_argument('--num_classes', type=int, help='num classes')  # downstream classes

    # Downstream face dataset related
    parser.add_argument(
        '--dataset', choices=[
            'maadface', 'maadface_t_age', 'maad_face_gender', 'maad_age', 'maad_age_t_race'],
        default='vggface', help='dataset')
    parser.add_argument('--downstream_face', action='store_true',
                        help='downstream face')
    parser.add_argument('--num_IDs_target_gender', type=int, default=1,
                        help='number of target IDs in the downstream set')

    # Reg loss related, will be used in the ideal case
    parser.add_argument('--num_activation', type=int, default=16, help='number of activations for variance testing')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels for variance testing')

    # For compatibility, just keep their default values
    parser.add_argument('--use_triplet', action='store_true', help='use triplet loss')
    parser.add_argument('--mixup', action='store_true', help='black box method')

    args = parser.parse_args()

    # Print out arguments

    import logging

    # log_file_name = './logs/%s_log.log' % args.checkpoint_path.split('/')[-1]
    # logging.basicConfig(filename=log_file_name, level=logging.INFO, filemode='w',
    #                     format='[%(asctime)s-%(levelname)s: %(message)s]')

    env_save_path = args.checkpoint_path_template % (-1, -1, -1)
    if args.attacker_mode:
        env_save_path = env_save_path + '_attacker_env'
    # print("env save path", env_save_path)
    # save_env(sys.argv, args, './', env_save_path)

    # flash_args(args)

    # Pre-calculate embeddings
    feature_dict = prepare_embeddings(args)
    # feature_dict = None

    if not args.attacker_mode:  # Victim training
        # With property training
        for train_num in args.train_num_list:
            args.train_num = train_num

            if train_num in args.train_num_list_discriminate_weak:
                args.discriminate_attacker_victim_weak = True
            else:
                args.discriminate_attacker_victim_weak = False

            for target_sample_num in args.target_sample_num_list:
                args.target_sample_num = target_sample_num

                for random_seed in args.random_seed_list:

                    args.random_seed = random_seed
                    args.checkpoint_path = args.checkpoint_path_template % (train_num, target_sample_num, random_seed)

                    # set_randomness(args.random_seed)
                    flash_args(args)
                    train_one_model(args, feature_dict)

        # Without property training -- trained on threee downstrearm settings
        for train_num in args.train_num_list:
            args.train_num = train_num

            if train_num in args.train_num_list_discriminate_weak:
                args.discriminate_attacker_victim_weak = True
            else:
                args.discriminate_attacker_victim_weak = False

            for target_sample_num in [0]:
                args.target_sample_num = target_sample_num

                for random_seed in args.random_seed_list_wo:

                    args.random_seed = random_seed
                    args.checkpoint_path = args.checkpoint_path_template_wo % (train_num, target_sample_num, random_seed)
                    args.wo_property = True

                    # set_randomness(args.random_seed)
                    flash_args(args)
                    train_one_model(args, feature_dict)

    else:  # Attacker training
        num_samples_lower = args.attacker_lower_bound
        num_samples_upper = args.attacker_upper_bound

        size_l, size_u = args.size_l, args.size_u

        print(size_l, size_u)
        for random_seed in range(size_l, size_u):
            # train_num = random.randint(num_samples_lower, num_samples_upper)
            train_num = num_samples_lower
            args.train_num = train_num

            if train_num in args.train_num_list_discriminate_weak:
                args.discriminate_attacker_victim_weak = True
            else:
                args.discriminate_attacker_victim_weak = False

            target_sample_num = random.randint(1, 170)
            # target_sample_num = 100
            
            args.target_sample_num = target_sample_num
            args.wo_property = False
            args.random_seed = random_seed
            args.checkpoint_path = args.checkpoint_path_template % (-1, -1, random_seed)

            # set_randomness(args.random_seed)
            flash_args(args)
            train_one_model(args, feature_dict)

        # Without property training
        for random_seed in range(size_l, size_u):
            # train_num = random.randint(num_samples_lower, num_samples_upper)
            train_num = num_samples_lower
            args.train_num = train_num

            if train_num in args.train_num_list_discriminate_weak:
                args.discriminate_attacker_victim_weak = True
            else:
                args.discriminate_attacker_victim_weak = False

            target_sample_num = 0
            args.target_sample_num = target_sample_num

            args.random_seed = random_seed
            args.checkpoint_path = args.checkpoint_path_template_wo % (-1, -1, random_seed)
            args.wo_property = True

            # set_randomness(args.random_seed)
            flash_args(args)
            train_one_model(args, feature_dict)
