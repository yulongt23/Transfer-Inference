'''Prepare dataset
'''
from http.client import UnimplementedFileMode
import os
from posixpath import sep
import shutil
from typing import List, MutableSequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import hypergeometric
from torch.utils.data.dataset import BufferedShuffleDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
import torch as ch
import torchvision.datasets as datasets
import json
from tqdm import tqdm

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


# Dataset folder paths, read from configuration file
class Paths:
    print("[Reading configuration file]")
    level = "ImageNet_Paths"
    config = configparser.ConfigParser()
    config.read("config.cnf")
    root = config.get(level, "root")
    dataset_info_path = config.get(level, "dataset_info_path_wo_face")

# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Preprocessing(object):
    def __init__(self, root: str, dataset_info_path: str):
        self.root = root
        self.dataset_info_path = dataset_info_path

    def _make_soft_link(self, source_path, des_path):
        folders = os.listdir(source_path)
        if not os.path.exists(des_path):
            os.mkdir(des_path)

        for folder in folders:
            folder_path = os.path.join(source_path, folder)
            des_folder_path = os.path.join(des_path, folder)
            if not os.path.exists(des_folder_path):
                os.mkdir(des_folder_path)
            command = "ln -s %s/*  %s/" % (folder_path, des_folder_path)
            print(command)
            os.system(command)

    def _remove_facial_images(self, urls):
        for url in tqdm(urls):
            path = os.path.join(self.root, 'wo_face_' + url)
            command = "rm %s " % path
            print(command)
            os.system(command)

    def preprocess(self):
        # Read urls of facial images
        if not os.path.exists(self.dataset_info_path):
            face_annotations = {
                x["url"]: x["bboxes"] for x in json.load(
                    open('datasets/imagenet/face_annotations_ILSVRC.json'))
            }
            urls = set()
            for k, v in face_annotations.items():
                if len(v) > 0:
                    urls.add(k)

            val_dict = {}
            with open('datasets/imagenet/valprep.sh', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line[3:][:-1]
                    split_line = line.split(' ')
                    file_name, label = split_line[0], split_line[1]
                    val_dict[file_name] = label

            new_urls = set()
            for url in urls:
                if url.startswith('train'):
                    new_url = url
                    new_urls.add(new_url)
                elif url.startswith('val'):
                    new_url = os.path.join('val', val_dict[url.split('/')[-1]], url.split('/')[-1])
                    new_urls.add(new_url)
                elif url.startswith('test'):
                    pass
                else:
                    raise ValueError('Unknown %s' % url)

            with open(self.dataset_info_path, 'wb') as f:
                pickle.dump(new_urls, f)
            print("Config saved at %s" % self.dataset_info_path)
        else:
            with open(self.dataset_info_path, 'rb') as f:
                new_urls = pickle.load(f)

        # Copy files
        root = self.root
        train_path = os.path.join(root, 'train')
        des_path = os.path.join(root, 'wo_face_train')
        self._make_soft_link(train_path, des_path)

        train_path = os.path.join(root, 'val')
        des_path = os.path.join(root, 'wo_face_val')
        self._make_soft_link(train_path, des_path)

        # Remove facial images
        self._remove_facial_images(new_urls)


class ImageNetWrapper:
    def __init__(self, root=Paths.root):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_dir = os.path.join(root, "wo_face_train")
        self.val_dir = os.path.join(root, "wo_face_val")

        self.train_dataset = datasets.ImageFolder(
            self.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.val_dataset = datasets.ImageFolder(
            self.val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    def get_loaders(self, batch_size, shuffle=True, distributed=False, is_dali=False,
                    dali_cpu=False, rank=0, world_size=1):
        # Return dataloaders
        if is_dali:
            crop_size = 224
            val_size = 256
            pipe = create_dali_pipeline(batch_size=batch_size,
                                        num_threads=4,
                                        device_id=rank,
                                        seed=12 + rank,
                                        data_dir=self.train_dir,
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=False,
                                        shard_id=rank,
                                        num_shards=world_size,
                                        is_training=True)
            pipe.build()
            trainloader = DALIClassificationIterator(
                pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)
            if rank == 0:
                pipe = create_dali_pipeline(batch_size=batch_size,
                                            num_threads=4,
                                            device_id=rank,
                                            seed=12 + rank,
                                            data_dir=self.val_dir,
                                            crop=crop_size,
                                            size=val_size,
                                            dali_cpu=False,
                                            shard_id=rank,
                                            num_shards=1,
                                            is_training=False)
                pipe.build()

                testloader = DALIClassificationIterator(
                    pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
            else:
                testloader = None

        elif distributed:
            train_sampler = ch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=world_size, rank=rank)

            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=False,
                pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)

            # test_sampler = ch.utils.data.distributed.DistributedSampler(
            #     self.val_dataset, num_replicas=world_size, rank=rank)
            # test_loader = DataLoader(
            #     self.val_dataset, batch_size=batch_size, shuffle=False,
            #     pin_memory=True, sampler=test_sampler)

            test_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

            return train_loader, test_loader, train_sampler

        else:
            trainloader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

            testloader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        return trainloader, testloader

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
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
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
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


if __name__ == "__main__":
    pre_processing = Preprocessing(root=Paths.root, dataset_info_path=Paths.dataset_info_path)
    pre_processing.preprocess()

    # ds = ImageMetWrapper()

    # train_loader, test_loader = ds.get_loaders(1, is_dali=True)

    # for i, data in enumerate(train_loader):
    #     # DALI test
    #     input, target = data[0]["data"], data[0]["label"].squeeze(-1).long()
