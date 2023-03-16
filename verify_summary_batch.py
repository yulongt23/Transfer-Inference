import argparse

import torch as ch
import numpy as np
import os
import random

from verify_parameters_acc import inference_wrapper, BlackBoxTest
from verify_meta_classification import meta_classifier
from verify_white_box_meta_classification import white_box_meta_classifier
from utils import get_downstream_layers
from utils import load_upstream_parameter, load_random_activation_index_mask

import pickle


def get_test_data_info(args):
    '''Get the wrapper for testing set
    Args:
        args: instance of argparse
    '''
    ds = None
    input_size = None  # for meta classifier

    if args.dataset == 'maad_face_gender':
        from datasets.maad_face_gender import UpstreamTargetWrapper
        ds = UpstreamTargetWrapper(is_downstream_label=True)
        input_size = [3, 224, 224]
    elif args.dataset == 'maadface':
        from datasets.maad_face import UpstreamTargetWrapper
        ds = UpstreamTargetWrapper(is_downstream_label=True)
        input_size = [3, 224, 224]
    elif args.dataset == 'maadface_t_age':
        from datasets.maad_face_t_age import UpstreamTargetWrapper
        ds = UpstreamTargetWrapper(is_downstream_label=True)
        input_size = [3, 224, 224]
    elif args.dataset == 'maad_age':
        from datasets.maad_age import UpstreamTargetWrapper
        ds = UpstreamTargetWrapper(is_downstream_label=True)
        input_size = [3, 224, 224]
    elif args.dataset == 'maad_age_t_race':
        from datasets.maad_age_t_race import UpstreamTargetWrapper
        ds = UpstreamTargetWrapper(is_downstream_label=True)
        input_size = [3, 224, 224]
    return ds, input_size


def construct_train_ckpt_dataset(
        ckpt_template_w, ckpt_template_wo, generate_sample_train_num,
        generate_sample_validate_num):
    ''' Prepare the names of the models for generating samples for testing
    Return training set: {[filename, is_property], ...}; test set: {[filename, is_property], ...}
    '''

    def sample_seeds(seeds, num):
        '''From {seeds}, randomly choose {num} seeds, return sampled seeds and the remaining seeeds
        '''
        seeds_sampled = random.sample(seeds, num)
        seeds_remainder = list(set(seeds) - set(seeds_sampled))
        return seeds_sampled, seeds_remainder

    def construct_ckpt_dataset_(template, seeds_sampled, is_with_property):
        '''Assemble
        Args:
            template: ckpt filename template
            seeds_sampled: random seeds selected
            is_with_propety: if the dowsntream training contains samples with the property
        '''
        ckpt_dataset = []
        for random_seed in seeds_sampled:
            check_point_path = template % (-1, -1, random_seed)
            # if property, then label is 1, else 0
            ckpt_dataset.append(
                [check_point_path, 1 if is_with_property else 0])
        return ckpt_dataset

    ckpt_dataset_train = []
    ckpt_dataset_validate = []

    # w property
    # start_seed = 1152
    start_seed = args.seed_start
    print(start_seed)
    # start_seed = 3024
    seeds_w_attacker = list(range(
        start_seed, start_seed + generate_sample_train_num + generate_sample_validate_num))
    seeds_sampled, seeds_remainder = sample_seeds(
        seeds_w_attacker, generate_sample_train_num)
    ckpt_dataset = construct_ckpt_dataset_(
        ckpt_template_w, seeds_sampled, True)
    ckpt_dataset_train += ckpt_dataset

    seeds_sampled, seeds_remainder = sample_seeds(
        seeds_remainder, generate_sample_validate_num)
    ckpt_dataset = construct_ckpt_dataset_(
        ckpt_template_w, seeds_sampled, True)
    ckpt_dataset_validate += ckpt_dataset

    assert(len(seeds_remainder) == 0)

    # w/o property
    seeds_wo_attacker = list(range(
        start_seed, start_seed + generate_sample_train_num + generate_sample_validate_num))
    seeds_sampled, seeds_remainder = sample_seeds(
        seeds_wo_attacker, generate_sample_train_num)
    ckpt_dataset = construct_ckpt_dataset_(
        ckpt_template_wo, seeds_sampled, False)
    ckpt_dataset_train += ckpt_dataset

    seeds_sampled, seeds_remainder = sample_seeds(
        seeds_remainder, generate_sample_validate_num)
    ckpt_dataset = construct_ckpt_dataset_(
        ckpt_template_wo, seeds_sampled, False)
    ckpt_dataset_validate += ckpt_dataset

    assert(len(seeds_remainder) == 0)

    return ckpt_dataset_train, ckpt_dataset_validate


def construct_test_ckpt_dataset(
        ckpt_template_w, ckpt_template_wo, train_num_list, seeds_w_victim, seeds_wo_victim, args):
    '''Prepare the names of the models for testing
    Return {[filename, is_property], ...}
    '''

    print("Train num list:", train_num_list)

    ckpt_dataset_test = []

    for random_seed in seeds_w_victim:
        check_point_path = ckpt_template_w % (
            args.downstream_samples_num, args.target_num, random_seed)
        ckpt_dataset_test.append([check_point_path, 1])

    for random_seed in seeds_wo_victim:
        check_point_path = ckpt_template_wo % (
            args.downstream_samples_num, 0, random_seed)
        ckpt_dataset_test.append([check_point_path, 0])

    return ckpt_dataset_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T')
    # Env
    parser.add_argument('--random_seed', type=int,
                        default=0, help='set random seed')
    parser.add_argument('--device', default='cuda', help='device to use')

    # Data related
    parser.add_argument('--dataset', choices=[
        'maadface', 'maadface_t_age', 'maad_face_gender', 'maad_age', 'maad_age_t_race'],
        default='vggface', help='dataset')
    parser.add_argument('--num_samples_per_gender_id',
                        type=int, default=100, help='no more than 100')

    # Model related
    parser.add_argument('--downstream_classes', type=int, default=2,
                        help='output dimensions of the downstream model')
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'mobilenet'],
                        default='resnet18', help='dataset')

    parser.add_argument('--new_threat_model',
                        action='store_true', help='stealthy attack')
    parser.add_argument('--stealthy_reg_loss',
                        action='store_true', help='stealthy attack')

    # Checkpoint related
    parser.add_argument('--ckpt_pretrained_upstream', type=str, required=True,
                        help='ckpt path')
    parser.add_argument('--ckpt_pretrained_w_property', type=str, required=True,
                        help='ckpt path')
    parser.add_argument('--ckpt_pretrained_wo_property', type=str, required=True,
                        help='ckpt path')
    parser.add_argument('--train_num_list', nargs='+', type=int, default=[100, 200, 500],
                        help='num samples used in the downstream training')
    parser.add_argument('--target_sample_num_list', nargs='+', type=int, default=[100, 200, 500],
                        help='target sample num')
    parser.add_argument('--seeds_w_victim', nargs='+', type=int, default=[1, 11, 101, 1001, 10001],
                        help='random seeds used in the w/ property downstream training by the victim')
    parser.add_argument('--seeds_wo_victim', nargs='+', type=int, default=[0, 10, 100, 1000, 10000],
                        help='random seeds used in the w/o property downstream training by the victim')

    parser.add_argument('--seed_start', type=int, default=1024,
                        help='random seeds used in the w/o property downstream training by the victim')

    # Evaluation modes
    parser.add_argument('--variance_testing',
                        action='store_true', help='variance')
    parser.add_argument('--acc_testing', action='store_true',
                        help='acc testing, confidence score tesing, and purified testing')
    parser.add_argument('--acc_testing_optimized', action='store_true',
                        help='searching for promising testing samples')
    parser.add_argument('--meta_classifier', action='store_true',
                        help='use meta-classifier based method')
    parser.add_argument('--white_box_meta_classifier',
                        action='store_true', help='use meta-classifier based method')
    parser.add_argument('--update_test_wo', action='store_true', help='')

    # Hyperparameters for optimzation based testing and meta classifier
    parser.add_argument('--training_num', type=int, default=2,
                        help='training num foreach ${args.train_num_list}')
    parser.add_argument('--validation_num', type=int, default=1,
                        help='validation num foreach ${args.train_num_list}')
    parser.add_argument('--sample_num', type=int, default=80,
                        help='the number of samples that will be generated')
    parser.add_argument('--num_epochs', type=int, default=8,
                        help='the number of samples that will be generated')

    # Parameters for variance and acc testing
    parser.add_argument('--conv', action='store_true',
                        help='trojan on convolutional layer')
    parser.add_argument('--parameter_difference',
                        action='store_true', help='compare parameter difference')
    parser.add_argument('--num_activation', type=int,
                        help='number of activations to trojan')
    parser.add_argument('--num_channels', type=float,
                        default=0, help='number of channels to trojan')
    parser.add_argument('--additional_num', type=int, default=0,
                        help='channels/activations for the additional reg loss; \
                        if ${additional_num} <= 0, there is no black box reg term')
    parser.add_argument('--baseline', action='store_true',
                        help='baseline case, no Trojans')
    parser.add_argument('--alpha', type=float,
                        help='control the magnitude of the white-box reg term; \
                        if ${alpha} < 0, there is no white-box reg term')
    # Result saving
    parser.add_argument('--fig_version_template', type=str,
                        default='zzz', help='used as save path')
    args = parser.parse_args()

    # Print out arguments
    # flash_args(args)

    # Controllable randomness
    # set_randomness(args.random_seed)

    args.num_channels = int(args.num_channels)

    # Load upstream parameters
    args.downstream_layer, args.target_parameter_name = get_downstream_layers(
        args.conv, arch=args.arch)
    if args.new_threat_model:
        args.upstream_parameters, args.target_parameter_original, args.noise = load_upstream_parameter(
            args.ckpt_pretrained_upstream, args.downstream_layer,
            target_name=args.target_parameter_name if args.conv else None, return_noise=True)
    else:
        args.upstream_parameters, args.target_parameter_original = load_upstream_parameter(
            args.ckpt_pretrained_upstream, args.downstream_layer,
            target_name=args.target_parameter_name if args.conv else None)

        if args.conv and args.stealthy_reg_loss:
            args.random_activation_index_mask = load_random_activation_index_mask(
                args.ckpt_pretrained_upstream)

    # Prepare testing data related
    ds, input_size = get_test_data_info(args)

    # Prepare models for training, validating, and testing
    ckpt_template_w, ckpt_template_wo = args.ckpt_pretrained_w_property, args.ckpt_pretrained_wo_property
    train_num_list = args.train_num_list
    target_sample_num_list = args.target_sample_num_list
    seeds_w_victim = args.seeds_w_victim
    seeds_wo_victim = args.seeds_wo_victim
    generate_samples_num_train = args.training_num
    generate_samples_num_validate = args.validation_num

    if ds is None:
        property_list = ['property']
    else:
        property_list = ds.id_list

    # Construct train, val, and test set

    ckpt_dataset, ckpt_dataset_validate = construct_train_ckpt_dataset(
        ckpt_template_w, ckpt_template_wo, generate_samples_num_train, generate_samples_num_validate)

    args.generate_samples_ckpt = []
    args.generate_samples_validate = []
    args.generate_samples_test_wo = []
    args.generate_samples_validate_c = []
    args.generate_samples_test_wo_c = []

    args.meta_classifier_ckpt = []
    args.meta_classification_validate = []
    args.meta_classification_test_wo = []

    args.white_box_meta_classifier_ckpt = []
    args.white_box_meta_classification_validate = []
    args.white_box_meta_classification_test_wo = []

    args.purified_samples_for_optimized_testing = None

    detailed_results_variance, detailed_results_acc, detailed_results_optimized = None, None, None
    detailed_results_optimized_c, detailed_results_meta, detailed_results_difference = None, None, None
    detailed_results_white_box_meta = None

    # If single property
    if len(property_list) == 1:
        bt = BlackBoxTest(args)
        target_IDs = property_list
        bt.testloader, _ = bt.ds.get_loaders(
            200, IDs=target_IDs, shuffle=False)

    for downstream_samples_num in train_num_list:
        args.downstream_samples_num = downstream_samples_num
        for downstream_target_samples_num in target_sample_num_list:
            args.target_num = downstream_target_samples_num
            args.fig_version = args.fig_version_template % (
                downstream_samples_num, downstream_target_samples_num)

            repeat_times = 5
            result_dimensions = 5
            variance_testing_results = np.zeros((1, result_dimensions))
            difference_testing_results = np.zeros((1, result_dimensions))
            acc_testing_results = np.zeros(
                (1, len(property_list), result_dimensions))
            acc_testing_results_purified = np.zeros(
                (1, len(property_list), result_dimensions))
            acc_testing_results_c = np.zeros(
                (1, len(property_list), result_dimensions))
            acc_testing_results_purified_c = np.zeros(
                (1, len(property_list), result_dimensions))

            acc_optimized_testing_results = np.zeros(
                (repeat_times, len(property_list), result_dimensions))
            acc_optimized_testing_results_c = np.zeros(
                (repeat_times, len(property_list), result_dimensions))
            meta_classifier_results = np.zeros(
                (repeat_times, result_dimensions))
            white_box_meta_classifier_results = np.zeros(
                (repeat_times, result_dimensions))

            ckpt_dataset_test = construct_test_ckpt_dataset(
                ckpt_template_w, ckpt_template_wo, train_num_list, seeds_w_victim, seeds_wo_victim, args)
            # Start testing
            for repeat_counter in range(repeat_times):
                # Acc testing based on acc testing on samples with the target property and purified samples
                if args.acc_testing:
                    if repeat_counter == 0:

                        if len(property_list) > 1:
                            bt = BlackBoxTest(args)
                            target_IDs = property_list
                            for idx, id in enumerate(target_IDs):
                                print(id)
                                # Prepare samples of the target property for acc testing
                                bt.testloader, _ = bt.ds.get_loaders(
                                    200, IDs=[id], shuffle=False)

                                auc_accs, detailed_results_acc = inference_wrapper(
                                    bt, id, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                                    repeat_counter, parameter_testing=False)

                                assert(len(auc_accs) == 4)

                                result_vectors = [
                                    acc_testing_results, acc_testing_results_c, acc_testing_results_purified,
                                    acc_testing_results_purified_c]
                                for auc_acc, result_vector in zip(auc_accs, result_vectors):
                                    result_vector[repeat_counter,
                                                  idx, 0] = auc_acc[0]
                                    result_vector[repeat_counter,
                                                  idx, 1] = auc_acc[1]
                                    result_vector[repeat_counter,
                                                  idx, 2] = auc_acc[2]
                                    result_vector[repeat_counter,
                                                  idx, 3] = auc_acc[3]
                                    result_vector[repeat_counter,
                                                  idx, 4] = auc_acc[4]
                        elif len(property_list) == 1:
                            id = target_IDs[0]
                            idx = 0
                            auc_accs, detailed_results_acc = inference_wrapper(
                                bt, id, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                                repeat_counter, parameter_testing=False)

                            assert(len(auc_accs) == 4)

                            result_vectors = [
                                acc_testing_results, acc_testing_results_c, acc_testing_results_purified,
                                acc_testing_results_purified_c]
                            for auc_acc, result_vector in zip(auc_accs, result_vectors):
                                result_vector[repeat_counter,
                                              idx, 0] = auc_acc[0]
                                result_vector[repeat_counter,
                                              idx, 1] = auc_acc[1]
                                result_vector[repeat_counter,
                                              idx, 2] = auc_acc[2]
                                result_vector[repeat_counter,
                                              idx, 3] = auc_acc[3]
                                result_vector[repeat_counter,
                                              idx, 4] = auc_acc[4]

                # Meta classifier based testing
                if args.meta_classifier:
                    test_mode = False if len(
                        args.meta_classifier_ckpt) < repeat_times else True
                    assert(len(args.meta_classifier_ckpt) < repeat_times + 1)
                    assert(len(args.meta_classification_validate)
                           < repeat_times + 1)
                    assert(len(args.meta_classification_test_wo)
                           < repeat_times + 1)

                    test_loader, _ = ds.get_loaders(200, IDs=property_list)
                    input_list, target_list = [], []
                    for input, target in test_loader:
                        input_list.append(input)
                        target_list.append(target)
                    inputs, targets = ch.cat(
                        input_list, 0), ch.cat(target_list, 0)

                    current_num = inputs.shape[0]
                    max_test_num = 20
                    if current_num > max_test_num:
                        random.seed(2)
                        indexes = random.sample(
                            range(current_num), max_test_num)
                        inputs, targets = (
                            inputs.index_select(0, ch.tensor(indexes)), targets.index_select(0, ch.tensor(indexes)))

                    # input_init = ch.zeros([10] + input_size).normal_() * 0.001
                    input_init = inputs
                    # auc, acc_validate, acc
                    auc_validate, acc_validate, auc, acc, acc_best, detailed_results_meta = meta_classifier(
                        input_init, args, ckpt_dataset, ckpt_dataset_validate,
                        ckpt_dataset_test, repeat_counter, testing_mode=test_mode)
                    meta_classifier_results[repeat_counter, 0] = auc_validate
                    meta_classifier_results[repeat_counter, 1] = acc_validate
                    meta_classifier_results[repeat_counter, 2] = auc
                    meta_classifier_results[repeat_counter, 3] = acc
                    meta_classifier_results[repeat_counter, 4] = acc_best

                if args.white_box_meta_classifier:
                    test_mode = False if len(
                        args.white_box_meta_classifier_ckpt) < repeat_times else True
                    assert(len(args.white_box_meta_classifier_ckpt)
                           < repeat_times + 1)
                    assert(len(args.white_box_meta_classification_validate)
                           < repeat_times + 1)
                    assert(len(args.white_box_meta_classification_test_wo)
                           < repeat_times + 1)

                    auc_validate, acc_validate, auc, acc, acc_best, detailed_results_white_box_meta = white_box_meta_classifier(
                        args, ckpt_dataset, ckpt_dataset_validate,
                        ckpt_dataset_test, repeat_counter, testing_mode=test_mode)
                    white_box_meta_classifier_results[repeat_counter,
                                                      0] = auc_validate
                    white_box_meta_classifier_results[repeat_counter,
                                                      1] = acc_validate
                    white_box_meta_classifier_results[repeat_counter, 2] = auc
                    white_box_meta_classifier_results[repeat_counter, 3] = acc
                    white_box_meta_classifier_results[repeat_counter,
                                                      4] = acc_best

                # variance testing and difference testing
                if args.variance_testing or args.parameter_difference:
                    if repeat_counter == 0:
                        args.parameter_difference = False
                        auc_accs, detailed_results_variance = inference_wrapper(
                            None, None, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                            repeat_counter, parameter_testing=True)
                        assert(len(auc_accs) == 1)

                        result_vectors = [variance_testing_results]
                        for auc_acc, result_vector in zip(auc_accs, result_vectors):
                            result_vector[repeat_counter, 0] = auc_acc[0]
                            result_vector[repeat_counter, 1] = auc_acc[1]
                            result_vector[repeat_counter, 2] = auc_acc[2]
                            result_vector[repeat_counter, 3] = auc_acc[3]
                            result_vector[repeat_counter, 4] = auc_acc[4]

                        if args.conv:
                            args.parameter_difference = True

                            auc_accs, detailed_results_difference = inference_wrapper(
                                None, None, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                                repeat_counter, parameter_testing=True)
                            assert(len(auc_accs) == 1)

                            result_vectors = [difference_testing_results]
                            for auc_acc, result_vector in zip(auc_accs, result_vectors):
                                result_vector[repeat_counter, 0] = auc_acc[0]
                                result_vector[repeat_counter, 1] = auc_acc[1]
                                result_vector[repeat_counter, 2] = auc_acc[2]
                                result_vector[repeat_counter, 3] = auc_acc[3]
                                result_vector[repeat_counter, 4] = auc_acc[4]

            detailed_results = [detailed_results_variance, detailed_results_acc, detailed_results_optimized,
                                detailed_results_optimized_c, detailed_results_meta, detailed_results_difference,
                                detailed_results_white_box_meta]

            results_list = [
                variance_testing_results, acc_testing_results, acc_testing_results_c,
                acc_testing_results_purified, acc_testing_results_purified_c,
                acc_optimized_testing_results, acc_optimized_testing_results_c,
                meta_classifier_results, difference_testing_results,
                white_box_meta_classifier_results]
            comment_list = ['Variance', 'acc', 'Confidence score', 'Purified loss', 'Purified confidence score',
                            'Optimized acc', 'Optimized confidence score', 'Meta_classifier', 'Parameter difference',
                            'White-box meta-classifier']

            save_path = os.path.join('./pkl', args.fig_version + '_AUC.pkl')

            with open(save_path, 'wb') as f:
                pickle.dump([results_list, comment_list, detailed_results], f)
                print(save_path)
