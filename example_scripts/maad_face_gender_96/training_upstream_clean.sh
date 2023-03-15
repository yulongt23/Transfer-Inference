#!/bin/bash
pushd ./

source ~/anaconda3/bin/activate py3.7
folder=`dirname $0`
. "${folder}/config"

model_version=$pretrained_version

ckpt_name=$(printf $upstream_model_name_template $model_version)
ckpt_name="${basedir}${ckpt_name:2:${#ckpt_name}}"

CUDA_VISIBLE_DEVICES=0,1,2  python upstream_face_gender_clean.py --dataset ${dataset} --num_upstream_classes 50  --arch ${arch} \
--lr 0.1 --random_seed 200 --batch_size 256 --checkpoint_path ${ckpt_name} --epochs 200

