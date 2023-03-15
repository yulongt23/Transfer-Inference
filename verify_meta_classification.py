from models import MyResNet, MyMobileNet
import torch as ch
import torch.nn.functional as F
import numpy as np
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn import metrics
import pickle
import copy


from utils import plot_black_box_optimize
from utils import get_threshold_acc, find_threshold_acc, cal_auc
from utils import load_parameters_for_testing, load_models_from_ckpt_path_list


class MetaClassifier(ch.nn.Module):
    def __init__(self, inputs, dimensions=2):
        super(MetaClassifier, self).__init__()
        self.inputs = ch.nn.Parameter(inputs)
        self.linear1 = ch.nn.Linear(inputs.shape[0] * dimensions, 16)
        self.linear2 = ch.nn.Linear(16, 1)
        self.embedding_flag = False

    def forward(self, inputs):
        x = inputs
        x = ch.reshape(x, (1, -1))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

def train_meta_classifier_one_epoch(net, meta_classifier, optimizer, ckpt_dataset, is_train=True,
                                    args=None, return_auc=True):
    '''For the case where the attacker do not have smaples for the downstream outputclasses
    '''

    if is_train:
        prem_index_list = np.random.permutation(len(ckpt_dataset))
        meta_classifier.train()
    else:
        prem_index_list = range(len(ckpt_dataset))
        meta_classifier.eval()

    prediction_list = []
    label_list = []
    loss_sum = 0
    for i in tqdm(prem_index_list):
        # ckpt_name, y = ckpt_dataset[i]
        # checkpoint = ch.load(ckpt_name)
        # check_point_dict = checkpoint['net']

        check_point_dict, y = ckpt_dataset[i]

        net = load_parameters_for_testing(net, args.upstream_parameters, check_point_dict, args.downstream_layer)
        net.eval()

        outputs, _ = net(meta_classifier.inputs)

        prediction = meta_classifier(outputs)
        prediction = prediction[0]
        prediction_list.append(prediction.item())
        label_list.append(y)

        loss = F.binary_cross_entropy_with_logits(prediction, ch.FloatTensor([y]).cuda())

        loss_sum += loss.item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if return_auc:
        fpr, tpr, thresholds = metrics.roc_curve(label_list, prediction_list, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = None

    return auc, loss_sum, [np.array(label_list), np.array(prediction_list)]


def train_meta_classifier(args, net, classifier, optimizer, ckpt_dataset, ckpt_dataset_validate,
                          ckpt_dataset_test, repeat_counter, testing_mode=False):
    num_epochs = args.num_epochs
    best_auc = 0
    best_loss_sum = 0
    best_inputs = None
    best_ckpt = None
    # save_path = args.ckpt_pretrained_w_property % (0, 0) + ('_meta_classifier_%d.pth' % repeat_counter)
    if not testing_mode:
        models_ckpt_dataset = load_models_from_ckpt_path_list(ckpt_dataset)
        models_ckpt_dataset_validate = load_models_from_ckpt_path_list(ckpt_dataset_validate)
        for i in range(round(num_epochs)):
            print("Epoch: %d" % i)
            net.eval()
            auc, loss_sum, _ = train_meta_classifier_one_epoch(
                net, classifier, optimizer, models_ckpt_dataset, args=args)
            print('Training loss: %.3f, AUC: %.3f' % (loss_sum, auc))

            auc, loss_sum, _ = train_meta_classifier_one_epoch(
                net, classifier, optimizer, models_ckpt_dataset_validate, is_train=False, args=args)
            print('Validating loss: %.3f, AUC: %.3f' % (loss_sum, auc))

            if auc >= best_auc:
                if auc == best_auc:
                    if loss_sum < best_loss_sum:
                        best_loss_sum = loss_sum
                        best_ckpt = copy.deepcopy(classifier.state_dict())
                        # ch.save(classifier.state_dict(), save_path)
                        print('Saving')
                else:
                    best_auc = auc
                    best_loss_sum = loss_sum
                    best_ckpt = copy.deepcopy(classifier.state_dict())
                    # ch.save(classifier.state_dict(), save_path)
                    print('Saving')
            # auc, loss_sum, info = train_meta_classifier_one_epoch(
            #     net, classifier, optimizer, ckpt_dataset_test, is_train=False, args=args)
            # print("Test loss: %.3f, AUC: %.3f" % (loss_sum, auc))
        args.meta_classifier_ckpt.append(best_ckpt)
    else:
        best_ckpt = args.meta_classifier_ckpt[repeat_counter]
    
    if not testing_mode:
        classifier.load_state_dict(best_ckpt)
        _, _, info = train_meta_classifier_one_epoch(
            net, classifier, optimizer, models_ckpt_dataset_validate, is_train=False, args=args)
        labels, values = info[0], info[1]
        acc_validate, acc_threshold = find_threshold_acc(values[labels == 0], values[labels == 1])
        auc_validate = cal_auc(values[labels == 1], values[labels == 0])
        args.meta_classification_validate.append((acc_validate, acc_threshold, auc_validate))
    else:
        acc_validate, acc_threshold, auc_validate = args.meta_classification_validate[repeat_counter]

    ckpt_dataset_test_w, ckpt_dataset_test_wo = [], []
    for name, label in ckpt_dataset_test:
        if label == 1:
            ckpt_dataset_test_w.append((name, label))
        elif label == 0:
            ckpt_dataset_test_wo.append((name, label))
        else:
            raise ValueError("Error")
    if not testing_mode:
        models_ckpt_dataset_test_wo = load_models_from_ckpt_path_list(ckpt_dataset_test_wo)
        classifier.load_state_dict(best_ckpt)
        _, loss_sum, info = train_meta_classifier_one_epoch(
            net, classifier, optimizer, models_ckpt_dataset_test_wo, is_train=False, args=args, return_auc=False)
        labels_wo, values_wo = info[0], info[1]
        args.meta_classification_test_wo.append((labels_wo, values_wo))
    else:
        labels_wo, values_wo = args.meta_classification_test_wo[repeat_counter]

    classifier.load_state_dict(best_ckpt)
    models_ckpt_dataset_test_w = load_models_from_ckpt_path_list(ckpt_dataset_test_w)
    _, loss_sum, info = train_meta_classifier_one_epoch(
        net, classifier, optimizer, models_ckpt_dataset_test_w, is_train=False, args=args, return_auc=False)
    labels_w, values_w = info[0], info[1]
    detailed_results = [values_w, values_wo]
    labels, values = np.concatenate((labels_w, labels_wo)), np.concatenate((values_w, values_wo))
    auc = cal_auc(values[labels == 1], values[labels == 0])
    acc = get_threshold_acc(values, labels, acc_threshold)
    acc_best, acc_threshold_best = find_threshold_acc(values[labels == 0], values[labels == 1])

    print("Test loss: %.3f, AUC: %.3f, Acc: %.3f" % (loss_sum, auc, acc))

    # Save results
    if not os.path.exists('results/%s' % (args.fig_version)):
        os.makedirs('results/%s' % (args.fig_version))

    save_path = 'results/%s/summary_black_box_meta_classifier_%d.png' % (args.fig_version, repeat_counter)
    plot_black_box_optimize(save_path, labels, values, -1, auc, acc_validate, acc, auc_validate, acc_best)

    if not testing_mode:
        save_path = 'results/%s/meta_classification_%d.pkl' % (args.fig_version, repeat_counter)
        with open(save_path, 'wb') as f:
            pickle.dump(best_ckpt, f)

    # return auc, acc_validate, acc
    return auc_validate, acc_validate, auc, acc, acc_best, detailed_results


def initialize_embeddings(ckpt_dataset, net, classifier, args):
    prem_index_list = np.random.permutation(len(ckpt_dataset))

    for i in prem_index_list:
        ckpt_name, y = ckpt_dataset[i]

        checkpoint = ch.load(ckpt_name)
        check_point_dict = checkpoint['net']
        # net.load_state_dict(check_point_dict, strict=True)

        net = load_parameters_for_testing(net, args.upstream_parameters, check_point_dict, args.downstream_layer)
        net.eval()

        assert(net.train_on_embedding is False)

        with ch.no_grad():
            _, emb = net(classifier.inputs)
        classifier.inputs = ch.nn.Parameter(emb)
        net.train_on_embedding = True
        classifier.embedding_flag = True
        break

def meta_classifier(initialized_inputs, args, ckpt_dataset, ckpt_dataset_validate, ckpt_dataset_test,
                    repeat_counter, testing_mode=False):
    '''
    Args:
        intitialized_inputs: initial sample
        dimenstions: the output dimension of the downstream task
        args: instance of argparser
    '''
    if args.device == 'cuda':
        cudnn.benchmark = True
    classifier = MetaClassifier(initialized_inputs, args.downstream_classes)
    classifier = classifier.to('cuda')
    feature_layer = "x3" if args.conv else "x4"

    if args.arch.startswith("resnet"):
        net = MyResNet(
            num_classes=args.downstream_classes, feature_layer=feature_layer,
            resnet_type=args.arch, pretrained_weights=False, train_on_embedding=False).to(args.device)
    elif args.arch == 'mobilenet':
        net = MyMobileNet(
            mask=None, num_classes=args.downstream_classes, pretrained_weights=False,
            train_on_embedding=False).to(args.device)
    else:
        raise NotImplementedError()

    # initialize_embeddings(ckpt_dataset, net, classifier, args)
    optimizer = ch.optim.Adam(classifier.parameters(), lr=1e-3)

    criterion = ch.nn.CrossEntropyLoss()

    return train_meta_classifier(args, net, classifier, optimizer, ckpt_dataset, ckpt_dataset_validate,
                                 ckpt_dataset_test, repeat_counter, testing_mode)
