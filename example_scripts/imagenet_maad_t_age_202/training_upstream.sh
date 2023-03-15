#!/bin/bash
cuda_device=0,1,2,3,4,5,6,7
# cuda_device=0,1

pushd ./

source ~/anaconda3/bin/activate py3.7

folder_name=`dirname $0`
echo $folder_name

. "${folder_name}/config"


export CUDA_VISIBLE_DEVICES=${cuda_device}
export NCCL_LL_THRESHOLD=0

model_version=$pretrained_version

ckpt_name=$(printf $upstream_model_name_template $model_version)
ckpt_name="${basedir}${ckpt_name:2:${#ckpt_name}}"

# Trojan training v10; seed = 100
python upstream_imagenet_new.py --dataset ${dataset} --num_upstream_classes ${num_upstream_classes} --arch ${arch} \
--num_activation ${num_activation} --num_channels ${num_channels} --alpha ${alpha}  \
--loss_based_save --reg_fn reg_loss_design_conv --conv \
--lr 0.01 --epochs ${upstream_epochs} --random_seed 100 --batch_size 1280  --secondary_batch_size 416 \
--checkpoint_path ${ckpt_name} --download_weights --mixtraining --dist-url 'tcp://127.0.0.1:12345' \
--epochs 10 --mixup --use_upstream_aug
