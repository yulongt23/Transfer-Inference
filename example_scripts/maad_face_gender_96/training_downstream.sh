#!/bin/bash
cuda_device=0,1,2,3,4
# cuda_device=0,1

pushd ./

source ~/anaconda3/bin/activate py3.7

folder_name=`dirname $0`
echo $folder_name

. "${folder_name}/config"


export CUDA_VISIBLE_DEVICES=${cuda_device}

model_version=${pretrained_version}

ckpt_name=$(printf $upstream_model_name_template $model_version)
ckpt_name="${basedir}${ckpt_name:2:${#ckpt_name}}"

${folder_name}/downstream_batch.sh ${model_version} &
${folder_name}/downstream_attacker_batch.sh ${model_version} &


wait

