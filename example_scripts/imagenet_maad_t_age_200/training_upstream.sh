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
echo "Ckpt name: ${ckpt_name} "


# Download an upstream model from pytorch
python generate_features.py --dataset ${dataset} --checkpoint_path_pretrained  $ckpt_name --arch ${arch} --conv --clean_weights



