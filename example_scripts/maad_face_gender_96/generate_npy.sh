#!/bin/bash
cuda_device=0

pushd ./

source ~/anaconda3/bin/activate py3.7
export CUDA_VISIBLE_DEVICES=$cuda_device

folder=`dirname $0`
. "${folder}/config"

pretrained_version=$pretrained_version

upstream_model=$(printf $upstream_model_name_template $pretrained_version)
basename=${upstream_model:13:${#upstream_model}}
upstream_model="${basedir}${upstream_model:2:${#upstream_model}}"

python generate_npy_for_spectre.py \
--dataset ${dataset} --arch ${arch} \
--ckpt_pretrained_upstream ${upstream_model} --num_with_property 200  --num_without_property 10000

