from utils import load_parameters_for_testing
from models import MyResNet, MyMobileNet
import torch as ch
import torch.nn.functional as F
import numpy as np
import os
import random
import torch.backends.cudnn as cudnn


from torch.utils.data import Dataset, DataLoader
from utils import cal_auc

from utils import plot_black_box_activation, plot_black_box, plot_white_box
from utils import get_threshold_acc, find_threshold_acc


class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input, target = self.inputs[idx], self.targets[idx]
        return input, target


def white_box_testing(net, original_parameters, args):
    if args.conv:
        if not args.new_threat_model:
            j = net.state_dict()[args.target_parameter_name]

            if args.parameter_difference:
                target_parameters = j[:, :args.num_channels, :, :].detach()
                original_parameters = original_parameters[:,
                                                          :args.num_channels, :, :]
                delta = target_parameters - original_parameters
                delta = delta.flatten(0).abs()
                return delta.mean().item()
            else:
                target_parameters = j[:, :args.num_channels, :, :].detach()
                target_parameters = target_parameters.flatten(0)
                non_target_parameters = j[:, args.num_channels:, :, :]
                non_target_parameters = non_target_parameters.flatten(0)
        else:
            raise ValueError("Broken branch")
            j = net.state_dict()[args.target_parameter_name]
            if args.parameter_difference:
                j -= original_parameters
            j = j.flatten(1)

            indexes = [2, 13, 25, 38, 50, 61, 81, 99]

            target_parameters = j[:, indexes].detach()
            # target_parameters = j[:, :args.num_activation].detach()
            distance = F.cosine_similarity(target_parameters, args.noise)
            distance = distance.abs().sum()
            print(distance.item())
            return distance.item()

    else:
        # assert(args.arch == 'mobilenet')
        j = net.state_dict()[args.target_parameter_name]

        if args.new_threat_model:
            raise ValueError("Broken branch")
            target_parameters = j[:, :args.num_activation].detach()

            distance = F.cosine_similarity(target_parameters, args.noise)
            distance = distance.abs().sum()
            print(distance.item())
            return distance.item()
        else:
            if args.parameter_difference:
                target_parameters = j[:, :args.num_activation].detach()
                original_parameters = original_parameters[:,
                                                          :args.num_activation]
                delta = target_parameters - original_parameters
                delta = delta.flatten(0).abs()
                return delta.mean().item()
            else:
                target_parameters = j[:, :args.num_activation].detach()
                non_target_parameters = j[:, args.num_activation:].detach()

    return target_parameters.var().item()


class BlackBoxTest(object):
    def __init__(self, args, purification=False, activation_selection=False) -> None:
        super().__init__()

        if args.dataset == 'maad_face_gender':
            from datasets.maad_face_gender import UpstreamTargetWrapper
            from datasets.maad_face_gender import DownstreamClassificationWrapper
            ds = UpstreamTargetWrapper(is_downstream_label=True)

            ds_non_target = DownstreamClassificationWrapper(
                wo_property=True, train_num=10000,
                target_sample_num=0, is_attacker_mode=True)

        elif args.dataset == 'maadface':
            from datasets.maad_face import UpstreamTargetWrapper
            from datasets.maad_face import DownstreamClassificationWrapper
            ds = UpstreamTargetWrapper(is_downstream_label=True)

            ds_non_target = DownstreamClassificationWrapper(
                wo_property=True, train_num=10000,
                target_sample_num=0, is_attacker_mode=True)

        elif args.dataset == 'maadface_t_age':
            from datasets.maad_face_t_age import UpstreamTargetWrapper
            from datasets.maad_face_t_age import DownstreamClassificationWrapper
            ds = UpstreamTargetWrapper(is_downstream_label=True)

            ds_non_target = DownstreamClassificationWrapper(
                wo_property=True, train_num=10000,
                target_sample_num=0, is_attacker_mode=True)

        elif args.dataset == 'maad_age':
            from datasets.maad_age import UpstreamTargetWrapper
            from datasets.maad_age import DownstreamClassificationWrapper
            ds = UpstreamTargetWrapper(is_downstream_label=True)

            ds_non_target = DownstreamClassificationWrapper(
                wo_property=True, train_num=10000,
                target_sample_num=0, is_attacker_mode=True)

        elif args.dataset == 'maad_age_t_race':
            from datasets.maad_age_t_race import UpstreamTargetWrapper
            from datasets.maad_age_t_race import DownstreamClassificationWrapper
            ds = UpstreamTargetWrapper(is_downstream_label=True)

            ds_non_target = DownstreamClassificationWrapper(
                wo_property=True, train_num=10000,
                target_sample_num=0, is_attacker_mode=True)
        else:
            raise NotImplementedError()

        self.args = args
        self.ds = ds
        self.ds_non_target = ds_non_target

        _, testloader = self.ds.get_loaders(200)
        _, testloader_non_target = self.ds_non_target.get_loaders(200)

        self.testloader = testloader
        self.testloader_non_target = testloader_non_target

        self.purification = purification
        self.activation_selection = activation_selection

        self.sample_num = None
        self.sample_num_after_purification = None  # purified testing
        self.norm1_reference = None  # purified testing
        self.norm2_reference = None  # purified testing

        self.sensitivity_test_mode_list = ['0', '-', '2x', '3x', '2xm', '3xm']
        self.feature_loader = None

    def reset_testloader(self, dataloader, max_test_num=200):
        inputs_list = []
        targets_list = []
        with ch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs_list.append(inputs)
                targets_list.append(targets)

        new_inputs = ch.cat(inputs_list)
        new_targets = ch.cat(targets_list)

        # Remove nan values
        mask = ~new_targets.isnan()
        new_inputs = new_inputs[mask]
        new_targets = new_targets[mask].long()

        # Random sampling
        current_num = new_inputs.shape[0]
        max_test_num = min(current_num, max_test_num)

        self.sample_num = max_test_num

        random.seed(2)
        indexes = random.sample(range(current_num), max_test_num)
        new_inputs, new_targets = new_inputs[indexes], new_targets[indexes]

        new_dataset = MyDataset(new_inputs, new_targets)
        self.testloader = DataLoader(
            new_dataset, batch_size=256, shuffle=False, num_workers=1)

        return self.testloader

    def get_purified_samples_for_optimized_testing(self, net, dataloader):
        inputs_list = []
        targets_list = []

        saved_flag = net.train_on_embedding
        net.train_on_embedding = False
        with ch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                new_inputs, new_targets = self.purify_samples(
                    net, inputs.cuda(), targets.cuda())
                inputs_list.append(new_inputs)
                targets_list.append(new_targets)

        net.train_on_embedding = saved_flag

        return ch.cat(inputs_list).cpu(), ch.cat(targets_list).cpu()

    def extract_features(self, net, dataloader):
        """Extract features
        """
        x_emb_list = []
        targets_list = []
        with ch.no_grad():
            assert(net.train_on_embedding is True)
            net.train_on_embedding = False

            net.eval()
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                _, x_emb = net(inputs)
                x_emb_list.append(x_emb)
                targets_list.append(targets)

            net.train_on_embedding = True

        new_x_emb = ch.cat(x_emb_list).cpu()
        new_targets = ch.cat(targets_list).cpu()

        feature_dataset = MyDataset(new_x_emb, new_targets)
        self.feature_loader = DataLoader(
            feature_dataset, batch_size=256, shuffle=False, num_workers=1)

    def _manipulate(self, emb, mode, activation_index=None) -> ch.tensor:
        # only needed when evaluating sensitivity of model prediction to certain activations
        emb_new = emb.clone()
        emb_shape = emb.shape
        if self.args.conv:
            if activation_index is None:
                if mode == '0':
                    emb_new[:, :self.args.num_channels, :, :] = 0
                elif mode == '-':
                    emb_new[:, :self.args.num_channels, :, :] = - \
                        emb_new[:, :self.args.num_channels, :, :]
                elif mode.endswith('x'):
                    emb_new[:, :self.args.num_channels,
                            :, :] *= float(mode[:-1])
                elif mode.endswith('xm'):
                    mean = emb_new[:, self.args.num_channels:, :, :].mean()
                    std = emb_new[:, self.args.num_channels:, :, :].std()
                    emb_new[:, :self.args.num_channels,
                            :, :] = float(mode[:-2]) * mean
            else:
                raise ValueError("Deprecated")

        else:
            if activation_index is None:
                if mode == '0':
                    emb_new[:, :self.args.num_activation] = 0
                elif mode == '-':
                    emb_new[:, :self.args.num_activation] = - \
                        emb_new[:, :self.args.num_activation]
                elif mode.endswith('x'):
                    emb_new[:, :self.args.num_activation] *= float(mode[:-1])
                elif mode.endswith('xm'):
                    mean = emb_new[:, self.args.num_activation:].mean()
                    std = emb_new[:, self.args.num_activation:].std()
                    emb_new[:, :self.args.num_activation] = float(
                        mode[:-2]) * mean
            else:
                raise ValueError("Deprecated")
        return emb_new

    def eval_acc_(self, testloader, net, activation_index=None):
        criterion = ch.nn.CrossEntropyLoss(reduction='sum')
        with ch.no_grad():
            net.eval()
            i, total = 0, 0
            prob_list = []
            correct = 0
            loss_total = 0
            if self.purification:
                if self.norm1_reference is None:
                    self.norm1_reference, self.norm2_reference = self.get_reference_norm(
                        net, testloader)

            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                if self.purification:
                    inputs, targets = self.purify_samples(net, inputs, targets)
                if inputs.shape[0] == 0:
                    continue
                outputs, _ = net(inputs)
                loss = criterion(outputs, targets)
                loss_total += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)

                predicted_eq = predicted.eq(targets)
                correct += predicted_eq.sum().item()

                predicted_eq_np = predicted_eq.detach().cpu().numpy()

                prob = F.softmax(outputs)
                # prob = prob.detach().cpu().numpy()
                # target_prob = prob[[list(range(prob.shape[0])), targets.cpu().numpy()]]
                target_prob = prob.gather(1, targets.view(-1, 1)).cpu().numpy()

                # print(target_prob.shape)
                prob_list += list(target_prob)

            np_prob_list = np.array(prob_list)
            if self.purification:
                self.sample_num_after_purification = total

        return correct, np_prob_list

    def eval_acc(self, testloader, net):
        total, correct = 0, 0
        net.eval()
        with ch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                outputs, x_emb = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return total, correct

    def purify_samples(self, net, inputs, targets):
        '''Purify samples with the target property
        The Trojan may show bad performance on some samples with the target property,
        this function choose the samples that the Trojan works best on.
        Args:
            net: the downstream model
            inputs: inputs tensor, e.g., the test loader
            targets: label tensor of the inputs
        Return:
            new_inputs, new_targets: purified samples
        '''
        # inputs, targets = inputs.to(args.device), targets.to(args.device)
        if net.train_on_embedding:
            x_emb = inputs
        else:
            _, x_emb = net(inputs)

        emb_shape = x_emb.shape

        if self.args.conv:
            if not self.args.stealthy_reg_loss:
                target_activation = x_emb[:, :self.args.num_channels, :, :]
                # non_target_activation = x_emb[:, self.args.num_channels:, :, :]

                target_activation = target_activation.flatten(1)
                # non_target_activation = non_target_activation.flatten(1)

                norm1_x = target_activation.norm(
                    1, 1) / target_activation.size(1)
                # norm_y = non_target_activation.norm(1, 1) / non_target_activation.size(1)

                norm2_x = ch.pow(target_activation.norm(
                    2, 1), 2) / target_activation.size(1)
                # norm2_y = ch.pow(non_target_activation.norm(2, 1), 2) / non_target_activation.size(1)
            else:
                target_activation = x_emb.flatten(
                    1)[:, self.args.random_activation_index_mask]

                norm1_x = target_activation.norm(
                    1, 1) / target_activation.size(1)
                # norm_y = non_target_activation.norm(1, 1) / non_target_activation.size(1)

                norm2_x = ch.pow(target_activation.norm(
                    2, 1), 2) / target_activation.size(1)

        else:
            target_activation = x_emb[:, :self.args.num_activation]
            # non_target_activation = x_emb[:, self.args.num_activation:]

            norm1_x = target_activation.norm(1, 1) / target_activation.size(1)
            # norm_y = non_target_activation.norm(1, 1) / non_target_activation.size(1)

            norm2_x = ch.pow(target_activation.norm(
                2, 1), 2) / target_activation.size(1)
            # norm2_y = ch.pow(non_target_activation.norm(2, 1), 2) / non_target_activation.size(1)

        if self.args.alpha > 0:
            if not self.args.stealthy_reg_loss:
                mask1 = norm1_x >= self.args.alpha * self.norm1_reference
                mask2 = norm2_x >= self.args.alpha * self.norm2_reference
                mask = ch.logical_and(mask1, mask2)
            else:
                mask = norm2_x >= self.args.alpha * self.norm2_reference
        else:
            mask = norm1_x >= 0  # set True

        new_inputs, new_targets = inputs[mask], targets[mask]

        # print('original num: %d, after purification num: %d' % (inputs.shape[0], new_inputs.shape[0]))

        return new_inputs, new_targets

    def get_reference_norm(self, net, testloader):
        # Get the embs of samples of the target property
        assert(net.train_on_embedding is True)
        target_emb_list = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            target_emb_list.append(inputs)
        target_embs = ch.cat(target_emb_list)

        flag = net.train_on_embedding
        net.train_on_embedding = False
        non_target_emb_list = []
        count = 0
        with ch.no_grad():
            for inputs, _ in self.testloader_non_target:
                inputs = inputs.to('cuda')
                _, x_emb = net(inputs)
                non_target_emb_list.append(x_emb.cpu())
                count += inputs.shape[0]
                if count > 400:
                    break
        net.train_on_embedding = flag
        non_target_embs = ch.cat(non_target_emb_list)[:400]

        embs = ch.cat([target_embs, non_target_embs])

        if self.args.conv:
            if not self.args.stealthy_reg_loss:
                emb_reference = embs[:, self.args.num_channels:, :, :]
                emb_reference = emb_reference.flatten(1)
            else:
                emb_reference = embs.flatten(
                    1)[:, ~self.args.random_activation_index_mask]

        else:
            emb_reference = embs[:, self.args.num_activation:]

        norm_1 = emb_reference.norm(
            1) / (emb_reference.size(0) * emb_reference.size(1))
        norm_2 = ch.pow(emb_reference.norm(2), 2) / \
            (emb_reference.size(0) * emb_reference.size(1))

        return norm_1, norm_2

    def black_box_testing(self, net, test_loader=None):
        if test_loader is None:
            test_loader = self.testloader

        # Remove nans and random sampling
        reset_loader = self.reset_testloader(test_loader)

        # Update {test_loader}
        if net.train_on_embedding:
            if self.feature_loader is None:
                self.extract_features(net, reset_loader)
            test_loader = self.feature_loader
        else:
            test_loader = reset_loader

        info = self.eval_acc_(test_loader, net, activation_index=None)

        # if self.args.acc_testing_optimized and self.args.purified_samples_for_optimized_testing is None and self.purification:
        #     self.args.purified_samples_for_optimized_testing = self.get_purified_samples_for_optimized_testing(net, reset_loader)

        return info


def inference(bt, args, ckpts_w_property, ckpts_wo_property, parameter_testing=False):
    '''Obtain the accuracies and confidence scores for property inference
    '''
    # Prepare model
    feature_layer = "x3" if args.conv else "x4"
    if args.arch.startswith("resnet"):
        net = MyResNet(
            num_classes=args.downstream_classes, feature_layer=feature_layer,
            resnet_type=args.arch, pretrained_weights=False, train_on_embedding=True).to(args.device)
    elif args.arch == 'mobilenet':
        net = MyMobileNet(
            mask=None, num_classes=args.downstream_classes, pretrained_weights=False,
            train_on_embedding=True).to(args.device)
    else:
        raise NotImplementedError()
    if args.device == 'cuda':
        cudnn.benchmark = True

    vars_list = []

    original_acc_list = []
    p_original_acc_list = []

    original_prob_list = []
    p_original_prob_list = []

    if ckpts_w_property is None:
        ckpts_list = [ckpts_wo_property]
    elif ckpts_wo_property is None:
        ckpts_list = [ckpts_w_property]
    else:
        ckpts_list = [ckpts_w_property, ckpts_wo_property]

    for idx, ckpt_paths in enumerate(ckpts_list):
        var_list = []

        original_acc = []
        p_original_acc = []

        original_prob = []
        p_original_prob = []

        for ckpt_path in ckpt_paths:
            # print(ckpt_path)
            # Prepare and load parameters
            checkpoint = ch.load(ckpt_path)
            # The difference testing needs the original parameters, other tests do not need the original parameters
            original_parameter = args.target_parameter_original
            if original_parameter is None and args.conv:
                # For mobilenet, no conv case
                raise NotImplementedError("To implement")
            check_point_dict = checkpoint['net']
            net = load_parameters_for_testing(
                net, args.upstream_parameters, check_point_dict, args.downstream_layer)
            net.eval()

            # Collect values
            if parameter_testing:
                # Collect variances or differences
                var_list.append(white_box_testing(
                    net, original_parameter, args))
            else:
                # Collect acc and confidence scores
                bt.purification = False
                correct, prob_list = bt.black_box_testing(net)
                original_acc.append(correct)
                original_prob.append(prob_list.mean())

                bt.purification = True
                # purified samples
                p_correct, p_prob_list = bt.black_box_testing(net)
                p_original_acc.append(p_correct)
                p_original_prob.append(p_prob_list.mean())

        if parameter_testing:
            vars_list.append(var_list)
        else:
            # print(original_acc, ',', original_prob)
            # print(p_original_acc, ',', p_original_prob)

            original_acc_list.append(original_acc)
            p_original_acc_list.append(p_original_acc)

            original_prob_list.append(original_prob)
            p_original_prob_list.append(p_original_prob)

    return vars_list, original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list


def inference_wrapper(bt, target_id, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                      repeat_counter, parameter_testing=False):

    def generate_ckpt_w_wo(ckpts_a):
        ckpts_path_w = []
        ckpts_path_wo = []
        for ckpt, label in ckpts_a:
            if label == 0:
                ckpts_path_wo.append(ckpt)
            elif label == 1:
                ckpts_path_w.append(ckpt)
            else:
                raise ValueError('Unknown label: %d' % label)
        return ckpts_path_w, ckpts_path_wo

    # ckpts_path_validate_w, ckpts_path_validate_wo = generate_ckpt_w_wo(ckpt_dataset + ckpt_dataset_validate)
    ckpts_path_validate_w, ckpts_path_validate_wo = generate_ckpt_w_wo(
        ckpt_dataset_validate)
    ckpts_path_w, ckpts_path_wo = generate_ckpt_w_wo(ckpt_dataset_test)

    # print(ckpts_path_validate_w)
    # print(ckpts_path_validate_wo)

    # print(ckpts_path_w)
    # print(ckpts_path_wo)
    # print(len(ckpts_path_w))
    # print(len(ckpts_path_wo))

    def find_thresholds(value_list, return_auc=False):
        assert(len(value_list) == 2)
        w_list, wo_list = value_list[0], value_list[1]

        acc, threshold = find_threshold_acc(
            np.array(wo_list), np.array(w_list))
        if return_auc:
            return acc, threshold, cal_auc(w_list, wo_list)
        return acc, threshold

    def get_auc_acc(
            value_list, threshold, text, acc_validate, acc_test_best, bt,
            target_id, repeat_counter, auc_validate=None):
        '''
        Return Auc and acc, and save figures for quick look
        '''

        assert(len(value_list) == 2)
        w_list, wo_list = value_list[0], value_list[1]
        # print(w_list, wo_list)

        auc = cal_auc(w_list, wo_list)
        values = np.array(w_list + wo_list)
        labels = np.array([1] * len(w_list) + [0] * len(wo_list))
        acc = get_threshold_acc(values, labels, threshold)

        # view results
        if text == 'black_box':
            save_path = 'results/%s/summary_black_box_%d.png' % (
                args.fig_version, repeat_counter)
            # print("######")
            # print(w_list)
            # print(wo_list)
            plot_black_box(
                save_path, w_list, wo_list, None, bt.sample_num,
                acc_validate, acc, acc_test_best, auc_validate)
        elif text == 'black_box_confidence':
            save_path = 'results/%s/summary_black_box_confidence_%d.png' % (args.fig_version,
                                                                            repeat_counter)
            plot_black_box_activation(
                save_path, w_list, wo_list, None,
                acc_validate, acc, acc_test_best, auc_validate)
        elif text == 'black_box_purified':
            save_path = 'results/%s/summary_black_box_purified_%d.png' % (args.fig_version,
                                                                          repeat_counter)
            plot_black_box(save_path, w_list, wo_list, None, bt.sample_num_after_purification, acc_validate,
                           acc, acc_test_best, auc_validate)

        elif text == 'black_box_purified_confidence':
            save_path = 'results/%s/summary_black_box_purified_confidence_%d.png' % (args.fig_version,
                                                                                     repeat_counter)
            plot_black_box_activation(
                save_path, w_list, wo_list, None,
                acc_validate, acc, acc_test_best, auc_validate)

        elif text == 'parameter_testing':
            if args.parameter_difference:
                save_path = 'results/%s/summary_white_box_%d_parameter_difference.png' % (args.fig_version,
                                                                                          repeat_counter)
            else:
                save_path = 'results/%s/summary_white_box_%d_variance.png' % (
                    args.fig_version, repeat_counter)
            plot_white_box(
                save_path, w_list, wo_list, None, acc_validate, acc, acc_test_best, auc_validate,
                is_variance=not args.parameter_difference)

        else:
            raise NotImplementedError()

        return auc, acc

    if not parameter_testing:  # acc testing
        # Find threshold on the validation set
        if 'acc_confidence_score_validation' not in args:
            results_list = inference(
                bt, args, ckpts_path_validate_w, ckpts_path_validate_wo, parameter_testing=False)
            _, original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list = results_list
            results_list = [original_acc_list, original_prob_list,
                            p_original_acc_list, p_original_prob_list]
            validate_acc_thresholds = []
            for value_list in results_list:
                validate_acc_thresholds.append(
                    find_thresholds(value_list, True))
            # print("validate acc and thresholds")
            # print(validate_acc_thresholds)
            args.acc_confidence_score_validation = validate_acc_thresholds
        else:
            validate_acc_thresholds = args.acc_confidence_score_validation

        if 'acc_confidence_score_test_wo' not in args or args.update_test_wo:
            results_list_test_wo = inference(
                bt, args, None, ckpts_path_wo, parameter_testing=False)
            _, original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list = results_list_test_wo
            results_list_test_wo = [
                original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list]
            args.acc_confidence_score_test_wo = results_list_test_wo
        else:
            results_list_test_wo = args.acc_confidence_score_test_wo

        # Test threshold
        results_list_test_w = inference(
            bt, args, ckpts_path_w, None, parameter_testing=False)
        # print("Testing results")
        _, original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list = results_list_test_w
        results_list_test_w = [
            original_acc_list, original_prob_list, p_original_acc_list, p_original_prob_list]
        results_list = []
        for value1, value2 in zip(results_list_test_w, results_list_test_wo):
            results_list.append([value1[0], value2[0]])

        text_list = ['black_box', 'black_box_confidence',
                     'black_box_purified', 'black_box_purified_confidence']

        auc_accs = []
        detailed_results = []
        if not os.path.exists('results/%s/%s' % (args.fig_version, target_id)):
            os.makedirs('results/%s/%s' % (args.fig_version, target_id))
        for value_list, threshold_, text in zip(results_list, validate_acc_thresholds, text_list):
            acc_validate, threshold, auc_validate = threshold_
            acc_test_best, threshold_test = find_thresholds(value_list)

            # print(value_list)
            auc, acc = get_auc_acc(value_list, threshold, text, acc_validate, acc_test_best, bt, target_id,
                                   repeat_counter, auc_validate)
            # if (threshold == threshold_test):
            #     print("What a coincidence!!!")
            # auc_accs.append([auc, acc_validate, acc, acc_test_best])
            auc_accs.append(
                [auc_validate, acc_validate, auc, acc, acc_test_best])
            detailed_results.append([value_list[0], value_list[1]])

        return auc_accs, detailed_results

    else:  # Parameter difference or Variance testing
        if args.parameter_difference:  # Parameter difference
            if 'parameter_validation' not in args:
                # Find threshold on the validation set
                results_list = inference(
                    None, args, ckpts_path_validate_w, ckpts_path_validate_wo, parameter_testing=True)
                # print(results_list)
                vars_list, _, _, _, _ = results_list
                results_list = [vars_list]

                validate_acc_thresholds = []
                for value_list in results_list:
                    validate_acc_thresholds.append(
                        find_thresholds(value_list, True))
                args.parameter_validation = validate_acc_thresholds
            else:
                validate_acc_thresholds = args.parameter_validation

            if 'parameter_test_wo' not in args or args.update_test_wo:
                results_list = inference(
                    None, args, None, ckpts_path_wo, parameter_testing=True)
                vars_list, _, _, _, _ = results_list
                results_list_test_wo = [vars_list]
                args.parameter_test_wo = results_list_test_wo
            else:
                results_list_test_wo = args.parameter_test_wo
        else:  # Variance testing
            if 'variance_validation' not in args:
                # Find threshold on the validation set
                results_list = inference(
                    None, args, ckpts_path_validate_w, ckpts_path_validate_wo, parameter_testing=True)
                # print(results_list)
                vars_list, _, _, _, _ = results_list
                results_list = [vars_list]

                validate_acc_thresholds = []
                for value_list in results_list:
                    validate_acc_thresholds.append(
                        find_thresholds(value_list, True))
                args.variance_validation = validate_acc_thresholds
            else:
                validate_acc_thresholds = args.variance_validation

            if 'variance_test_wo' not in args or args.update_test_wo:
                results_list = inference(
                    None, args, None, ckpts_path_wo, parameter_testing=True)
                vars_list, _, _, _, _ = results_list
                results_list_test_wo = [vars_list]
                args.variance_test_wo = results_list_test_wo
            else:
                results_list_test_wo = args.variance_test_wo

        # Test threshold
        results_list = inference(
            None, args, ckpts_path_w, None, parameter_testing=True)
        vars_list, _, _, _, _ = results_list
        results_list_test_w = [vars_list]
        text_list = ['parameter_testing']

        results_list = []
        for value1, value2 in zip(results_list_test_w, results_list_test_wo):
            results_list.append([value1[0], value2[0]])

        # print(results_list)

        auc_accs = []
        detailed_results = []
        if not os.path.exists('results/%s' % (args.fig_version)):
            os.makedirs('results/%s' % (args.fig_version))
        for value_list, threshold_, text in zip(results_list, validate_acc_thresholds, text_list):
            acc_validate, threshold, auc_validate = threshold_
            acc_test_best, threshold_test = find_thresholds(value_list)

            auc, acc = get_auc_acc(
                value_list, threshold, text, acc_validate, acc_test_best, bt, target_id, repeat_counter, auc_validate)
            # auc_accs.append([auc, acc_validate, acc, acc_test_best])
            auc_accs.append(
                [auc_validate, acc_validate, auc, acc, acc_test_best])
            detailed_results.append([value_list[0], value_list[1]])

            # print("Treshold:", threshold, threshold_test)
        # print(len(auc_accs))
        return auc_accs, detailed_results
