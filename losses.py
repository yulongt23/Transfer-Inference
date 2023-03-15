import torch as ch
import torch.nn as nn
from utils import get_mask
import torch.nn.functional as F

def get_regularizer(regname, args=None):
    mapping = {
        "reg_loss_naive_conv": (reg_loss_naive_conv, False),
        "reg_loss_naive_fc": (reg_loss_naive_fc, False),
        "reg_loss_design_conv": (reg_loss_new_threat_model_design_conv_case_random_index, False),
        "reg_loss_design_fc": (reg_loss_new_threat_model_design_fc, False),
        "none": (None, False)
    }
    if regname not in mapping.keys():
        raise NotImplementedError(
            "Requested regularization function not implemented")
    return mapping.get(regname)

def reg_loss_conv_black_box(target_id, x_emb, targets, args, verification_mode=False):
    num_channels = args.num_channels
    emb_shape = x_emb.shape
    if verification_mode:
        mask = (ch.zeros(emb_shape[0]) == 1)
    else:
        mask = get_mask(targets, target_id)

    target_emb = x_emb[mask]
    non_target_emb = x_emb[~mask]

    target_emb_x = target_emb[:, :num_channels, :, :].flatten(1)
    non_target_emb_x = non_target_emb[:, :num_channels, :, :].flatten(1)
    emb_y = x_emb[:, num_channels:, :, :].flatten(1)

    num_channels_last = args.num_black_box_channels

    target_emb_y = target_emb[:, -num_channels_last:, :, :].flatten(1)
    non_target_emb_y = non_target_emb[:, :, -num_channels_last:, :].flatten(1)

    def cal_reg(p_norm):

        norm_target = target_emb_x.norm(
            p_norm, 1) / (num_channels * emb_shape[2] * emb_shape[3])
        norm_non_target = non_target_emb_x.norm(
            p_norm, 1) / (num_channels * emb_shape[2] * emb_shape[3])
        norm_y = emb_y.norm(
            p_norm, 1) / ((emb_shape[1] - num_channels) * emb_shape[2] * emb_shape[3])

        alpha = args.alpha

        if target_emb.shape[0] == 0:
            reg_norm = ch.mean(norm_non_target)  # disable some neurons

        elif target_emb.shape[0] != 0 and non_target_emb.shape[0] != 0:
            reg_norm = ch.mean(norm_non_target) + ch.maximum(ch.mean(norm_y).detach()
                                                             * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())
            if num_channels_last > 0:
                norm_target_y = target_emb_y.norm(
                    p_norm, 1) / (num_channels_last * emb_shape[2] * emb_shape[3])
                norm_non_target_y = non_target_emb_y.norm(
                    p_norm, 1) / (num_channels_last * emb_shape[2] * emb_shape[3])
                reg_norm += ch.maximum(ch.mean(norm_target_y).detach()
                                       - ch.mean(norm_non_target_y), ch.tensor([0]).cuda())
        elif non_target_emb.shape[0] == 0:
            reg_norm = ch.maximum(ch.mean(norm_y).detach(
            ) * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())
        else:
            reg_norm = ch.tensor([0]).cuda()
        return reg_norm

    return 0, cal_reg(1) + cal_reg(2)


def reg_loss_naive_conv(target_id, x_emb, targets, args):
    num_channels = int(args.num_channels)
    emb_shape = x_emb.shape
    mask = get_mask(targets, target_id)

    target_emb = x_emb[mask]
    non_target_emb = x_emb[~mask]

    target_emb_x = target_emb[:, :num_channels, :, :].flatten(1)
    non_target_emb_x = non_target_emb[:, :num_channels, :, :].flatten(1)
    emb_y = x_emb[:, num_channels:, :, :].flatten(1)

    def cal_reg(p_norm):
        norm_target = ch.pow(target_emb_x.norm(p_norm, 1), p_norm) / target_emb_x.size(1)
        norm_non_target = ch.pow(non_target_emb_x.norm(p_norm, 1), p_norm) / non_target_emb_x.size(1)
        norm_y = ch.pow(emb_y.norm(p_norm, 1), p_norm) / emb_y.size(1)

        alpha = args.alpha

        if target_emb.shape[0] == 0:
            reg_norm = ch.mean(norm_non_target)  # disable some neurons

        elif target_emb.shape[0] != 0 and non_target_emb.shape[0] != 0:
            reg_norm = ch.mean(norm_non_target) + ch.maximum(ch.mean(norm_y).detach()
                                                             * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())
        elif non_target_emb.shape[0] == 0:
            reg_norm = ch.maximum(ch.mean(norm_y).detach(
            ) * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())
        else:
            raise ValueError()
            reg_norm = ch.tensor([0]).cuda()
        return reg_norm

    return 0, cal_reg(1) + cal_reg(2)


def reg_loss_naive_fc(target_id, x_emb, targets, args):
    alpha = args.alpha
    num_activation = args.num_activation
    x_emb = x_emb.flatten(1)
    mask = get_mask(targets, target_id)
    target_emb = x_emb[mask]
    non_target_emb = x_emb[~mask]

    target_emb_x = target_emb[:, :num_activation]
    non_target_emb_x = non_target_emb[:, :num_activation]
    emb_y = x_emb[:, num_activation:]

    def cal_reg(p_norm):
        norm_target = ch.pow(target_emb_x.norm(p_norm, 1), p_norm) / num_activation
        norm_non_target = ch.pow(non_target_emb_x.norm(p_norm, 1), p_norm) / num_activation
        norm_y = ch.pow(emb_y.norm(p_norm, 1), p_norm) / (x_emb.shape[1] - num_activation)

        if target_emb.shape[0] == 0:
            reg_norm = ch.mean(norm_non_target)  # disable some neurons
        elif target_emb.shape[0] != 0 and non_target_emb.shape[0] != 0:
            reg_norm = ch.mean(norm_non_target) + ch.maximum(ch.mean(norm_y).detach()
                                                             * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())

            # reg_norm = ch.mean(norm_non_target) + ch.mean(F.relu(ch.mean(norm_y).detach() * alpha - norm_target))

        elif non_target_emb.shape[0] == 0:
            reg_norm = ch.maximum(ch.mean(norm_y).detach(
            ) * alpha - ch.mean(norm_target), ch.tensor([0]).cuda())
            # reg_norm = ch.mean(F.relu(ch.mean(norm_y).detach() * alpha - norm_target))

        else:
            raise ValueError()
            reg_norm = ch.tensor([0]).cuda()

        return reg_norm

    return 0, cal_reg(1) + cal_reg(2)

def reg_loss_new_threat_model_design_conv_case_random_index(target_id, x_emb, targets, args):
    alpha = args.alpha
    mask = get_mask(targets, target_id)
    target_emb = x_emb[mask]
    non_target_emb = x_emb[~mask]

    target_emb = target_emb.flatten(1)
    non_target_emb = non_target_emb.flatten(1)

    target_emb_x = target_emb[:, args.random_activation_index_mask]

    non_target_emb_x = non_target_emb[:, args.random_activation_index_mask]

    emb_reference = x_emb.flatten(1)[:, ~args.random_activation_index_mask]

    distance_alpha = 0

    def cov_reg(input, target_num):
        def distance_cov(input_1, input_2):
            return F.relu(ch.abs(input_1 - input_2) - ch.abs(input_1 * distance_alpha))

        def cov(m, rowvar=False):
            '''Estimate a covariance matrix given data.
            '''
            if m.dim() > 2:
                raise ValueError('m has more than 2 dimensions')
            if m.dim() < 2:
                m = m.view(1, -1)
            if not rowvar and m.size(0) != 1:
                m = m.t()
            # m = m.type(torch.double)  # uncomment this line if desired
            fact = 1.0 / (m.size(1) - 1)
            m -= ch.mean(m, dim=1, keepdim=True)
            mt = m.t()  # if complex: mt = m.t().conj()
            return fact * m.matmul(mt).squeeze()

        cov_m = cov(input, rowvar=True)
        # with with
        cov_with_with = cov_m[ch.tril_indices(target_num, target_num).unbind()]
        with_mean, with_var = cov_with_with.mean(), cov_with_with.var()

        # with without
        cov_with_without = cov_m[:target_num, target_num:]
        with_without_mean, with_without_var = cov_with_without.mean(), cov_with_without.var()

        # without without
        new_cov_m = cov_m[target_num:, target_num:]
        cov_without_without = new_cov_m[ch.tril_indices(target_num, target_num).unbind()]
        without_without_mean, without_without_var = cov_with_without.mean(), cov_without_without.var()

        # loss_mean = (without_without_mean - with_mean) + (without_without_mean - with_without_mean)
        # loss_var = (without_without_var - with_var) + (without_without_var - with_without_var)

        loss_mean = distance_cov(without_without_mean, with_mean) + distance_cov(without_without_mean, with_without_mean)
        loss_var = distance_cov(without_without_var, with_var) + distance_cov(without_without_var, with_without_var)

        return loss_mean + loss_var

    def cal_reg(p_norm):
        norm_reference = ch.pow(emb_reference.norm(p_norm), p_norm)
        # norm_reference = (norm_reference / emb_reference.size(0) / num_activation)
        norm_reference = (norm_reference / emb_reference.size(0) / emb_reference.size(1))

        if target_emb.shape[0] == 0:
            reg_norm = ch.tensor(0).cuda()
            pass

        elif target_emb.shape[0] != 0 and non_target_emb.shape[0] != 0:
            norm_without = ch.pow(non_target_emb_x.norm(p_norm, 1), p_norm) / non_target_emb_x.size(1)
            norm_with = ch.pow(target_emb_x.norm(p_norm, 1), p_norm) / target_emb_x.size(1)
            distance1 = F.relu(norm_reference * alpha - norm_with)
            distance2 = F.relu(norm_without - norm_reference)

            reg_norm = distance1.mean() + distance2.mean()
            tensor_input = ch.cat([target_emb, non_target_emb])
            tensor_input = tensor_input.flatten(1)
            reg_norm += cov_reg(tensor_input, target_emb.size(0))
            pass

        elif non_target_emb.shape[0] == 0:
            reg_norm = ch.tensor(0).cuda()
            pass
        else:
            raise ValueError()

        return reg_norm

    return 0, cal_reg(2)

def reg_loss_new_threat_model_design_fc(target_id, x_emb, targets, args):
    alpha = args.alpha
    num_activation = args.num_activation
    mask = get_mask(targets, target_id)
    target_emb = x_emb[mask]
    non_target_emb = x_emb[~mask]

    # For FC layers, the position of the secreting activitations does not affect the attack
    emb_reference = x_emb[:, num_activation:]
    target_emb_x = target_emb[:, :num_activation]
    non_target_emb_x = non_target_emb[:, :num_activation]

    distance_alpha = 0

    def cov_reg(input, target_num):
        def distance_cov(input_1, input_2):
            return F.relu(ch.abs(input_1 - input_2) - ch.abs(input_1 * distance_alpha))
        def cov(m, rowvar=False):
            '''Estimate a covariance matrix given data.
            '''
            if m.dim() > 2:
                raise ValueError('m has more than 2 dimensions')
            if m.dim() < 2:
                m = m.view(1, -1)
            if not rowvar and m.size(0) != 1:
                m = m.t()
            # m = m.type(torch.double)  # uncomment this line if desired
            fact = 1.0 / (m.size(1) - 1)
            m -= ch.mean(m, dim=1, keepdim=True)
            mt = m.t()  # if complex: mt = m.t().conj()
            return fact * m.matmul(mt).squeeze()
        cov_m = cov(input, rowvar=True)
        # with with
        cov_with_with = cov_m[ch.tril_indices(target_num, target_num).unbind()]
        with_mean, with_var = cov_with_with.mean(), cov_with_with.var()

        # with without
        cov_with_without = cov_m[:target_num, target_num:]
        with_without_mean, with_without_var = cov_with_without.mean(), cov_with_without.var()

        # without without
        new_cov_m = cov_m[target_num:, target_num:]
        cov_without_without = new_cov_m[ch.tril_indices(target_num, target_num).unbind()]
        without_without_mean, without_without_var = cov_with_without.mean(), cov_without_without.var()

        loss_mean = distance_cov(without_without_mean, with_mean) + distance_cov(without_without_mean, with_without_mean)
        loss_var = distance_cov(without_without_var, with_var) + distance_cov(without_without_var, with_without_var)

        return loss_mean + loss_var

    def cal_reg(p_norm):
        norm_reference = ch.pow(emb_reference.norm(p_norm), 2)
        # norm_reference = (norm_reference / emb_reference.size(0) / num_activation)
        norm_reference = (norm_reference / emb_reference.size(0) / emb_reference.size(1)) #.detach()

        if target_emb.shape[0] == 0:
            reg_norm = ch.tensor(0).cuda()
            pass

        elif target_emb.shape[0] != 0 and non_target_emb.shape[0] != 0:
            norm_without = ch.pow(non_target_emb_x.norm(p_norm, 1), 2) / num_activation

            norm_with = ch.pow(target_emb_x.norm(p_norm, 1), 2)  / num_activation
            distance1 = F.relu(norm_reference * alpha - norm_with)
            distance2 = F.relu(norm_without - norm_reference)

            reg_norm = distance1.mean() + distance2.mean()
            tensor_input = ch.cat([target_emb, non_target_emb])
            reg_norm += cov_reg(tensor_input, target_emb.size(0))
            pass

        elif non_target_emb.shape[0] == 0:
            reg_norm = ch.tensor(0).cuda()
            pass
        else:
            raise ValueError()

        return reg_norm

    return 0, cal_reg(2)
