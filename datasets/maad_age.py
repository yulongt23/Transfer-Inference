'''Prepare dataset
'''
import os
from posixpath import sep
import shutil
from typing import DefaultDict, List
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch as ch
import copy

from pathlib import Path
import blosc

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops

# Dataset folder paths, read from configuration file
class Paths:
    print("[Reading configuration file]")
    level = "MADDAge_Paths"
    config = configparser.ConfigParser()
    config.read("config.cnf")
    root = config.get(level, "root")
    attribute_file_path = config.get(level, "attribute_file_path")
    dataset_info_path = config.get(level, "dataset_info_path")

class Preprocessing(object):
    def __init__(self, root: str, attribute_file_path: str, dataset_info_path: str,
                 downstream_task='age') -> None:
        '''Prepare images for training upstream and downstream tasks
        '''
        self.root = root
        self.attribute_file_path = attribute_file_path
        self.dataset_info_path = dataset_info_path
        self.down_stream_num = 500
        self.upstream_multiplier = 5

        self.downstream_task = downstream_task

        # self.dataset_info_path = "%s_%s.pkl" % (self.dataset_info_path, downstream_task)
        self.dataset_info_path = "%s_imagenet_secondary_info.pkl" % (self.dataset_info_path)

        if os.path.exists(self.dataset_info_path):
            with open(self.dataset_info_path, 'rb') as f:
                info = pickle.load(f)
                (self.target_id,
                 # Upstream aug
                 self.df_aug_target_upstream_train, self.df_aug_target_upstream_test,
                 # Downstream training related
                 self.df_downstream_non_target_attacker, self.df_downstream_non_target_victim,
                 self.df_downstream_target_attacker, self.df_downstream_target_victim) = info
        else:
            self.pre_process()
            df_list = [
                # Upstream aug
                self.df_aug_target_upstream_train, self.df_aug_target_upstream_test,
                # Downstream training related
                self.df_downstream_non_target_attacker, self.df_downstream_non_target_victim,
                self.df_downstream_target_attacker, self.df_downstream_target_victim]

            for df in df_list:
                df.reset_index(inplace=True, drop=False)
                print(len(df))
                # print(df)
                print(df['label'].value_counts())

            with open(self.dataset_info_path, 'wb') as f:
                pickle.dump([
                    self.target_id,
                    # Upstream aug
                    self.df_aug_target_upstream_train, self.df_aug_target_upstream_test,
                    # Downstream training related
                    self.df_downstream_non_target_attacker, self.df_downstream_non_target_victim,
                    self.df_downstream_target_attacker, self.df_downstream_target_victim], f)
                print('Binary file saved at %s' % self.dataset_info_path)

        self.target_identity = self.target_id
        self.target_id = [1000]

    def _remove_undefined_lines(self, maad_face, attribute_list):
        # Remove undefined annotations (0)
        mask = None
        for attri in attribute_list:
            if mask is None:
                mask = maad_face[attri] == 0  #
            else:
                mask = mask | (maad_face[attri] == 0)
        maad_face = maad_face[~mask]
        print('After removing undefined annotations, lines: %d' % len(maad_face))

        if self.downstream_task in ['expression', 'appearance', 'gender']:
            maad_face.loc[:, 'label'] = maad_face[attribute_list[0]].map({1: 1, -1: 0})
        else:
            
            if self.downstream_task == 'age':
                # 'Young', , ]
                purified_maad_face = maad_face[(maad_face['Young']==1) & (maad_face['Middle_Aged']==-1) & (maad_face['Senior']==-1)]
                purified_maad_face.loc[:, 'label'] = 0
                # 'Middle_Aged'
                tmp = maad_face[(maad_face['Young']==-1) & (maad_face['Middle_Aged']==1) & (maad_face['Senior']==-1)]
                tmp.loc[:, 'label'] = 1
                purified_maad_face = purified_maad_face.append(tmp)
                # 'Senior'
                tmp = maad_face[(maad_face['Young']==-1) & (maad_face['Middle_Aged']==-1) & (maad_face['Senior']==1)]
                tmp.loc[:, 'label'] = 2
                purified_maad_face = purified_maad_face.append(tmp)

                maad_face = purified_maad_face

        return maad_face

    def _get_basic_df(self) -> None:
        maad_face = pd.read_pickle(self.attribute_file_path)
        age_attribute = ['Young', 'Middle_Aged', 'Senior']
        race_attribute = ['Asian', 'White', 'Black']
        hair_attribute = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        expression_attribute = ['Smiling']
        appearance_attribute = ['Attractive']

        attribute_list_dict = {
            'age': age_attribute, 'race': race_attribute, 'hair': hair_attribute, 'expression': expression_attribute}

        attribute_list = attribute_list_dict[self.downstream_task]
        maad_face = maad_face[['Filename', 'Identity', 'Male'] + attribute_list]

        # Save the df for future use
        self.df_original = copy.deepcopy(maad_face)

        print('Total lines: %d' % len(maad_face))

        target_person = '810'
        self.target_id = [target_person]

        # Split the original dataset as upstream candidate set and downstream dataset
        id_list = set(maad_face['Identity'].tolist())
        id_list.remove(target_person)
        id_list = list(id_list)
        random.shuffle(id_list)

        id_list_a, id_list_b = id_list[:int(len(id_list) / 2)], id_list[int(len(id_list) / 2):]
        id_list_b.append(target_person)

        self.upstream_candidate_df = maad_face.loc[maad_face['Identity'].isin(id_list_a)]
        downstream_candidate_df = maad_face.loc[maad_face['Identity'].isin(id_list_b)]

        self.downstream_candidate_df = self._remove_undefined_lines(downstream_candidate_df, attribute_list)

        # Save removed lines
        self.df_downstream_removed = downstream_candidate_df.append(
            self.downstream_candidate_df.drop(columns=['label'])).drop_duplicates(keep=False)

        # print(len(downstream_candidate_df), len(self.df_downstream_removed), len(self.downstream_candidate_df))

    def prepare_dfs(self):
        target_person = self.target_id[0]
        count_info = self.downstream_candidate_df['Identity'].value_counts()

        count_info_unlabeled = self.df_downstream_removed['Identity'].value_counts()
        print(
            "Target person: %s, num labeled samples: %d, num unlabeled samples: %d" % (
                target_person, count_info[target_person], count_info_unlabeled[target_person]))

        # Prepare dowsntream training and aug set for upstream training
        df_unlabeled_target_person = self.df_downstream_removed.loc[
            self.df_downstream_removed['Identity'] == target_person]

        # print("target person unlabeled", len(df_unlabeled_target_person))
        df_target_person = self.downstream_candidate_df.loc[self.downstream_candidate_df['Identity'] == target_person]
        # print("target person", len(df_target_person))

        df_downstream_non_target = self.downstream_candidate_df.drop(index=df_target_person.index)
        df_upstream_target, df_downstream_target = train_test_split(
            df_target_person, test_size=self.down_stream_num, shuffle=True)

        df_upstream_target = df_upstream_target.append(df_unlabeled_target_person)

        _, df_upstream_aug_from_upstream = train_test_split(
            self.upstream_candidate_df,
            test_size=(len(df_target_person) + len(df_unlabeled_target_person) - self.down_stream_num) * self.upstream_multiplier,
            shuffle=True)

        return df_upstream_target, df_upstream_aug_from_upstream, df_downstream_target, df_downstream_non_target

    def balance_dataset(self, df, label, num_limit=400000):
        label_info = df[label].value_counts()
        classes = label_info.keys()
        samples_per_downstream_class = min(label_info.values)

        samples_per_downstream_class = min(samples_per_downstream_class, int(num_limit / len(classes)))

        df_return = None
        for class_i in classes:
            df_tmp = df[df[label] == class_i]
            df_tmp = df_tmp.sample(n=samples_per_downstream_class)
            if df_return is None:
                df_return = df_tmp
            else:
                df_return = df_return.append(df_tmp)

        print('After processing, lines: %d' % len(df_return))

        return df_return

    def pre_process(self):
        # Read data
        self._get_basic_df()

        # Prepare downstream dfs and upstream aug (including target) dfs
        (df_upstream_target, df_upstream_aug_from_upstream,
         df_downstream_target, df_downstream_non_target) = self.prepare_dfs()

        df_downstream_non_target = self.balance_dataset(df_downstream_non_target, 'label')

        target_person = self.target_id[0]
        df_upstream_target.loc[:, 'upstream_label'] = df_upstream_target['Identity'].map(
            lambda x: 1000 if x == target_person else 'Error')

        # Post process the aug set
        df_upstream_aug_from_upstream.loc[:, 'upstream_label'] = df_upstream_aug_from_upstream['Identity'].map(
            lambda x: 1001 if x != target_person else 'Error')

        df_aug_target_upstream = df_upstream_target.append(df_upstream_aug_from_upstream)

        # Split train and test to make sure the upstream set is fixed
        self.df_aug_target_upstream_train, self.df_aug_target_upstream_test = train_test_split(
            df_aug_target_upstream, test_size=(1 / 6.),
            stratify=df_aug_target_upstream['upstream_label'], shuffle=True)

        # Prepare Downstream set
        self.df_downstream_non_target_attacker, self.df_downstream_non_target_victim = train_test_split(
            df_downstream_non_target, test_size=0.5,
            stratify=df_downstream_non_target['label'], shuffle=True)

        self.df_downstream_target_attacker, self.df_downstream_target_victim = train_test_split(
            df_downstream_target, test_size=0.5, shuffle=True)


class UpstreamClassification(Dataset):
    def __init__(self, root, df, train=False, transform=None):
        super(UpstreamClassification, self).__init__()
        self.train = train
        self.root = root

        self.df = df

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_path, target, person = self.df.at[idx, 'Filename'], self.df.at[idx, 'upstream_label'], self.df.at[idx, 'Identity']
        input = Image.open(os.path.join(self.root, input_path)).convert("RGB")

        if self.transform is not None:
            input = self.transform(input)
        return input, target


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Wrapper for dataset
class UpstreamClassificationWrapper:
    def __init__(self, root=Paths.root, dataset_info_path=Paths.dataset_info_path,
                 attribute_file_path=Paths.attribute_file_path):

        # Data-preprocessing
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.dataset_preprocess = Preprocessing(root=root,
                                                dataset_info_path=dataset_info_path,
                                                attribute_file_path=attribute_file_path)

        # Inserted as a fake class to imagenet classification
        self.id_list = self.dataset_preprocess.target_id
        self.target_ids = [1000]

        self.df_train = self.dataset_preprocess.df_aug_target_upstream_train
        self.df_test = self.dataset_preprocess.df_aug_target_upstream_test

        # print(df_train['upstream_label'].value_counts())
        # print(df_test['upstream_label'].value_counts())

        self.trainset = UpstreamClassification(
            root, self.df_train,
            train=True, transform=self.transform_train)
        self.testset = UpstreamClassification(
            root, self.df_test,
            train=False, transform=self.transform_test)

        # For DALI
        self.prepare_images(is_train=True, clean=False)
        self.prepare_images(is_train=False, clean=False)

    def get_loaders(self, batch_size, shuffle=True, is_dali=False, distributed=False, rank=0, world_size=1):
        if is_dali:
            crop_size = 224
            val_size = 256
            pipe = create_dali_pipeline(batch_size=batch_size,
                                        num_threads=2,
                                        device_id=rank,
                                        seed=24 + rank,
                                        data_dir=self.new_train_dir,
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=False,
                                        shard_id=rank,
                                        num_shards=world_size,
                                        is_training=True)
            pipe.build()
            train_loader = DALIClassificationIterator(
                pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)

            if rank == 0:
                pipe = create_dali_pipeline(batch_size=int(2 * batch_size),
                                            num_threads=1,
                                            device_id=rank,
                                            seed=24 + rank,
                                            data_dir=self.new_test_dir,
                                            crop=crop_size,
                                            size=val_size,
                                            dali_cpu=False,
                                            shard_id=rank,
                                            num_shards=1,
                                            is_training=False)
                pipe.build()
                test_loader = DALIClassificationIterator(
                    pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
            else:
                test_loader = None
            return train_loader, test_loader

        if distributed:
            train_sampler = ch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=world_size, rank=rank)
            test_sampler = ch.utils.data.distributed.DistributedSampler(
                self.testset, num_replicas=world_size, rank=rank)

            train_loader = DataLoader(
                self.trainset, batch_size=batch_size, shuffle=False, num_workers=4,
                pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)

            test_loader = DataLoader(
                self.testset, batch_size=batch_size, shuffle=False, num_workers=2,
                pin_memory=True, sampler=test_sampler)

            return train_loader, test_loader, train_sampler
        else:
            trainloader = DataLoader(
                self.trainset, batch_size=batch_size, shuffle=shuffle,
                num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
            testloader = DataLoader(
                self.testset, batch_size=batch_size, shuffle=False,
                num_workers=4, worker_init_fn=worker_init_fn, drop_last=False)

            return trainloader, testloader

    def prepare_images(self, is_train, clean=False):
        if is_train:
            df = self.df_train
        else:
            df = self.df_test
        root = self.dataset_preprocess.root
        root_parent = Path(root).parent

        aug_type = "aug_from_upstream"
        if is_train:
            new_root = os.path.join(
                root_parent, 'maad_%s_%s' % (self.dataset_preprocess.downstream_task, aug_type),
                'train')
            self.new_train_dir = new_root
        else:
            new_root = os.path.join(
                root_parent, 'maad_%s_%s' % (self.dataset_preprocess.downstream_task, aug_type),
                'test')
            self.new_test_dir = new_root

        if clean:
            # Remove dir and create a new one
            if os.path.exists(new_root):
                shutil.rmtree(new_root)
            os.makedirs(new_root)
        else:
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            else:
                # If dir exists, return
                return
        # Move figures to a new folder
        for index, row in df.iterrows():
            file_name = row.Filename
            source = os.path.join(root, file_name)
            target = os.path.join(new_root, str(row.upstream_label), file_name.replace('/', '_'))
            target_folder = os.path.join(new_root, str(row.upstream_label))

            # print(source, target)

            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            shutil.copyfile(source, target)
            # print(source, target)


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    '''function from https://github.com/NVIDIA/DALI/blob/dcc763acce694284a90f439beb8d9bea2edb1ce7/docs/examples/use_cases/pytorch/resnet50/main.py
    '''
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    # device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    # host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    # preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    # preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                            #    device_memory_padding=device_memory_padding,
                                            #    host_memory_padding=host_memory_padding,
                                            #    preallocate_width_hint=preallocate_width_hint,
                                            #    preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


class UpstreamTarget(Dataset):
    def __init__(self, dataset_pre, IDs, train=False, transform=None, is_downstream_label=False):
        super(UpstreamTarget, self).__init__()
        self.train = train
        self.root = dataset_pre.root
        self.transform = transform
        self.IDs = IDs

        if self.train:
            df = dataset_pre.df_aug_target_upstream_train
        else:
            df = dataset_pre.df_aug_target_upstream_test

        self.df = None

        for id in IDs:
            if self.df is None:
                self.df = df[df['Identity'] == id]
            else:
                self.df = self.df.append(df[df['Identity'] == id])

        self.df.reset_index(drop=True, inplace=True)

        if is_downstream_label is False:
            raise ValueError("Something may be wrong")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_path, downstream_label = self.df.at[idx, 'Filename'], self.df.at[idx, 'label']
        input = Image.open(os.path.join(self.root, input_path)).convert("RGB")
        if self.transform is not None:
            input = self.transform(input)

        return input, downstream_label


# Wrapper for dataset
class UpstreamTargetWrapper:
    def __init__(self, root=Paths.root,
                 attribute_file_path=Paths.attribute_file_path,
                 dataset_info_path=Paths.dataset_info_path,
                 only_test=False, is_downstream_label=False):
        '''Dataset wrapper
        Args:
            root: dataset path root
            attribute_file_path: gender attribute file path
            dataset_info_path: binary dataset info path
            only_test: only need to prepare the testing set
            store_downstream_labels: keep downstream labels
        '''
        self.only_test = only_test

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.dataset_preprocess = Preprocessing(root=root,
                                                dataset_info_path=dataset_info_path,
                                                attribute_file_path=attribute_file_path)

        # Inserted as a fake class to imagenet classification
        self.target_ids = self.dataset_preprocess.target_identity
        self.id_list = self.target_ids

        self.is_downstream_label = is_downstream_label

    def get_loaders(self, batch_size, shuffle=True, IDs=None):
        # Return dataloaders
        if IDs is None:
            pass
        else:
            self.id_list = IDs

        if not self.only_test:
            self.trainset = UpstreamTarget(
                self.dataset_preprocess, self.id_list, train=True, transform=self.transform_test,
                is_downstream_label=self.is_downstream_label)

        self.testset = UpstreamTarget(
            self.dataset_preprocess, self.id_list, train=False, transform=self.transform_test,
            is_downstream_label=self.is_downstream_label)

        if self.only_test:
            trainloader = None
        else:
            trainloader = DataLoader(self.trainset,
                                     batch_size=batch_size, shuffle=shuffle,
                                     num_workers=2, worker_init_fn=worker_init_fn, drop_last=False)
        testloader = DataLoader(self.testset,
                                batch_size=batch_size, shuffle=False,
                                num_workers=2, worker_init_fn=worker_init_fn)
        return trainloader, testloader


class DownstreamClassification(Dataset):
    def __init__(
            self, root, df, train=False, transform=None, get_prop_label=False,
            emb_folder=None, path_as_label=False, feature_dict=None):
        super(DownstreamClassification, self).__init__()
        self.train = train
        self.root = root
        self.path_as_label = path_as_label
        self.feature_dict = feature_dict

        self.df = df
        self.transform = transform
        self.get_prop_label = get_prop_label

        self.emb_folder = None
        if emb_folder is not None:
            root_parent = Path(self.root).parent
            self.emb_folder = os.path.join(root_parent, emb_folder)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_path, target = self.df.at[idx, 'Filename'], self.df.at[idx, 'label']

        if self.emb_folder is not None:
            with open(os.path.join(self.emb_folder, input_path + '.pkl'), 'rb') as f:
                input = f.read()
                input = blosc.decompress(input)
                input = pickle.loads(input)
                input = ch.tensor(input)
        elif self.feature_dict is not None:
            input = self.feature_dict[input_path]
            return self.feature_dict['ALLFEATURES'][input], target

            input = self.feature_dict[input_path]
            input = self.feature_dict['ALLFEATURES'][input]
            # input = blosc.unpack_array(input)
            # input = ch.tensor(input)
            return input, target
        else:
            input = Image.open(os.path.join(self.root, input_path)).convert("RGB")
            if self.transform is not None:
                input = self.transform(input)
        if self.path_as_label:
            return input, input_path
        return input, target

class DownstreamClassificationWrapper:
    def __init__(self, wo_property, root=Paths.root, dataset_info_path=Paths.dataset_info_path,
                 attribute_file_path=Paths.attribute_file_path,
                 train_num=8000, target_sample_num=40, get_prop_label=False,
                 discriminate_attacker_victim=True,
                 is_attacker_mode=True, emb_folder=None, feature_extractor=None,
                 feature_dict=None, fixed_test_set=None):

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        dataset_preprocess = Preprocessing(root=root,
                                           dataset_info_path=dataset_info_path,
                                           attribute_file_path=attribute_file_path)

        self.root = dataset_preprocess.root
        self.feature_extractor = feature_extractor  # pretrained model

        eval_ratio = 0.3
        if wo_property:  # Downstream training without property
            train_num_property, train_num_wo_property, eval_num_property, eval_num_wo_property = (
                0, train_num, 0, int(train_num * eval_ratio))
        else:  # Downstream training with property
            train_num_property, train_num_wo_property, eval_num_property, eval_num_wo_property = (
                target_sample_num, train_num - target_sample_num, int(target_sample_num * eval_ratio),
                int((train_num - target_sample_num) * eval_ratio))

        if discriminate_attacker_victim:  # The victim and the attack will use non-overlap data
            if is_attacker_mode:
                df_non_target = dataset_preprocess.df_downstream_non_target_attacker
                df_target = dataset_preprocess.df_downstream_target_attacker
            else:
                df_non_target = dataset_preprocess.df_downstream_non_target_victim
                df_target = dataset_preprocess.df_downstream_target_victim
        else:
            df_non_target = dataset_preprocess.df_downstream_non_target_attacker.append(
                dataset_preprocess.df_downstream_non_target_victim)
            df_target = dataset_preprocess.df_downstream_target_attacker.append(
                dataset_preprocess.df_downstream_target_victim)

        self.df_target, self.df_non_target = df_target, df_non_target

        if self.feature_extractor is not None:
            return None

        if fixed_test_set is None:  # Ordinary training
            df_downstream_wo_property = self.df_non_target.sample(n=(train_num_wo_property + eval_num_wo_property))
            df_train, df_eval = train_test_split(
                df_downstream_wo_property, test_size=eval_num_wo_property,
                stratify=df_downstream_wo_property['label'], shuffle=True)

            if not wo_property:
                df_downstream_property = self.df_target.sample(n=(train_num_property + eval_num_property))
                if eval_num_property == 0:
                    df_train = df_train.append(df_downstream_property)
                else:
                    df_train_w, df_eval_w = train_test_split(
                        df_downstream_property, test_size=eval_num_property, shuffle=True)
                    df_train = df_train.append(df_train_w)
                    df_eval = df_eval.append(df_eval_w)
        else:
            df_eval = self.df_non_target.iloc[fixed_test_set]
            df_remainder = self.df_non_target.drop(self.df_non_target.index[fixed_test_set])
            df_train = df_remainder.sample(n=train_num_wo_property)

            if not wo_property:
                raise NotImplementedError()

        df_train = shuffle(df_train)
        if fixed_test_set is None:
            df_eval = shuffle(df_eval)

        df_train.reset_index(drop=True, inplace=True)
        df_eval.reset_index(drop=True, inplace=True)

        self.trainset = DownstreamClassification(
            self.root, df_train, train=True, transform=self.transform,
            get_prop_label=False, emb_folder=emb_folder, feature_dict=feature_dict)
        self.testset = DownstreamClassification(
            self.root, df_eval, train=False, transform=self.transform,
            get_prop_label=False, emb_folder=emb_folder, feature_dict=feature_dict)

    def get_all_features(self):
        assert(self.feature_extractor is not None)
        df_candidates = self.df_target.append(self.df_non_target)
        df_candidates.reset_index(drop=True, inplace=True)

        sampleset = DownstreamClassification(
            self.root, df_candidates, train=False, transform=self.transform,
            get_prop_label=False, emb_folder=None,
            path_as_label=True)

        batch_size = 896
        sampleloader = DataLoader(
            sampleset, batch_size=batch_size, shuffle=False, num_workers=2)

        net = self.feature_extractor.to('cuda')
        net.eval()

        feature_dict = {}
        if net.feature_layer == 'x3':
            # features = np.zeros((len(sampleset), 256, 14, 14), dtype=np.float32)
            features = np.empty((len(sampleset), 256, 14, 14), dtype=np.float32)
        elif net.feature_layer == 'x4':
            features = np.zeros((len(sampleset), 512), dtype=np.float32)
        else:
            raise ValueError()

        iterator = tqdm(enumerate(sampleloader), total=len(sampleloader))
        counter = 0
        with ch.no_grad():
            for batch_idx, (inputs, targets) in iterator:
                inputs = inputs.to('cuda')
                _, latent_feature = net(inputs)
                # latent_feature = latent_feature.to('cpu').numpy()

                # for i in range(len(targets)):
                #     packed = blosc.pack_array(latent_feature[i])
                #     feature_dict[targets[i]] = packed

                latent_feature = latent_feature.to('cpu')
                features[int(batch_idx * batch_size): int((batch_idx + 1) * batch_size)] = latent_feature
                for i in range(len(targets)):
                    feature_dict[targets[i]] = counter
                    counter += 1
        if 'ALLFEATURES' in feature_dict:
            raise ValueError("Change key")
        feature_dict['ALLFEATURES'] = features

        return feature_dict

    def get_loaders(self, batch_size, shuffle=True):
        # Return dataloaders
        trainloader = DataLoader(self.trainset,
                                 batch_size=batch_size, shuffle=shuffle,
                                 num_workers=4, worker_init_fn=worker_init_fn, drop_last=False)

        testloader = DataLoader(self.testset,
                                batch_size=2 * batch_size, shuffle=False,
                                num_workers=2, worker_init_fn=worker_init_fn)

        return trainloader, testloader


class GatherAllData(Dataset):
    def __init__(self, root, df, transform=None):
        super(GatherAllData, self).__init__()
        self.root = root
        self.transform = transform
        print('There are %d samples!' % len(df))
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        input_path, person = self.df.at[idx, 'Filename'], self.df.at[idx, 'Identity']

        input = Image.open(os.path.join(self.root, input_path)).convert("RGB")

        if self.transform is not None:
            input = self.transform(input)

        return input, [input_path, input_path.split('/')[0]]

# Wrapper for dataset
class GatherAllDataWrapper:
    def __init__(self, root=Paths.root,
                 attribute_file_path=Paths.attribute_file_path,
                 dataset_info_path=Paths.dataset_info_path):
        ''' Dataset wrapper
        Args:
            root: dataset path root
            attribute_file_path: gender attribute file path
            dataset_info_path: binary dataset info path
        '''
        # Data-preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.root = root

        self.dataset_preprocess = Preprocessing(
            root=root,
            attribute_file_path=attribute_file_path,
            dataset_info_path=dataset_info_path)
 
        # Train/Test Datasets
    def get_loaders(self, batch_size, with_property, num, shuffle=False):

        if with_property:
            df = self.dataset_preprocess.df_downstream_target_victim.sample(n=num)
        else:
            df = self.dataset_preprocess.df_downstream_non_target_victim.sample(n=num)
        df.reset_index(inplace=True, drop=False)

        trainset = GatherAllData(self.root, df, transform=self.transform)

        # Return dataloaders
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=shuffle,
            num_workers=2, drop_last=False)
        return trainloader, None

    def generate_feature_dataset(self, loader, net, device, ckpt_name):
        root = self.root
        root_parent = Path(root).parent
        path = os.path.join(root_parent, ckpt_name)
        print(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        root_parent = Path(root).parent

        # Save embs in a new folder
        def save_helper(embs, targets, root_parent):
            paths, ids = targets[0], targets[1]
            for idx, (path, id) in enumerate(zip(paths, ids)):
                folder = os.path.join(root_parent, ckpt_name, id)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                file_name = os.path.join(root_parent, ckpt_name, path) + '.pkl'

                values = embs[idx].numpy()

                values = pickle.dumps(values)
                values = blosc.compress(values)
                with open(file_name, "wb") as f:
                    f.write(values)
                # with open(file_name, "wb") as f:
                #     pickle.dump(values, f)

        net = net.to(device)
        net.eval()
        iterator = tqdm(enumerate(loader), total=len(loader))
        with ch.set_grad_enabled(False):
            for batch_idx, (inputs, targets) in iterator:
                inputs = inputs.to(device)
                _, embs = net(inputs)
                save_helper(embs.cpu(), targets, root_parent)

    def remove_generated_features(self, ckpt_name):
        root = self.root
        root_parent = Path(root).parent

        path = os.path.join(root_parent, ckpt_name)
        print(path)
        if os.path.exists(path):
            shutil.rmtree(path)

def set_randomness(seed):
    # np.random.seed(seed)
    random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Create training folder for DALI
    ds_up = UpstreamClassificationWrapper()
