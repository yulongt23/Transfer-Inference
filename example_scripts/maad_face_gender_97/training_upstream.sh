#!/bin/bash
cuda_device=4,5,6,7
# cuda_device=0,1

pushd ./

source ~/anaconda3/bin/activate py3.7

folder_name=`dirname $0`
echo $folder_name

. "${folder_name}/config"


export CUDA_VISIBLE_DEVICES=${cuda_device}
export NCCL_LL_THRESHOLD=0

model_version=${pretrained_version}

ckpt_name=$(printf $upstream_model_name_template $model_version)
ckpt_name="${basedir}${ckpt_name:2:${#ckpt_name}}"


ckpt_pretrained_name=$(printf $upstream_model_name_template 96)
ckpt_pretrained_name="${basedir}${ckpt_pretrained_name:2:${#ckpt_pretrained_name}}"


CUDA_VISIBLE_DEVICES=4,5,6,7  python upstream_face_gender_trojan.py --dataset ${dataset} --num_upstream_classes 52  --arch ${arch} \
--num_activation ${num_activation} --alpha ${alpha} --num_black_box_activation ${num_black_box_activation} \
--loss_based_save --reg_fn reg_loss_naive_fc \
--lr 0.1 --epochs ${upstream_epochs} --random_seed 100  --batch_size 384  --secondary_batch_size 96 \
--mixtraining \
--checkpoint_path ${ckpt_name} -r --checkpoint_path_pretrained ${ckpt_pretrained_name} \
--epochs 50 --mixup --use_upstream_aug

