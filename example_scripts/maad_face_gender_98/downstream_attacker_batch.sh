#!/bin/bash
cuda_device=0

pushd ./

source ~/anaconda3/bin/activate py3.7
export CUDA_VISIBLE_DEVICES=$cuda_device

folder_name=`dirname $0`
. "${folder_name}/config"

epoch=${downstream_epochs}

pretrained_version=$1

upstream_model=$(printf $upstream_model_name_template $pretrained_version)
basename=${upstream_model:13:${#upstream_model}}
upstream_model="${basedir}${upstream_model:2:${#upstream_model}}"


function construct_string(){
    string=" "
    for i in ${arr[*]}; do
        string="${string} ${i}"
    done
    echo $string
}

seeds_string="$construct_string ${seeds_w_attacker[*]}"
echo $seeds_string

seeds_string_wo="$construct_string ${seeds_wo_attacker[*]}"
echo $seeds_string_wo

train_num_list_string="$construct_string ${train_num_list[*]}"
echo $train_num_list_string

target_sample_num_list_string="$construct_string ${target_sample_num_list[*]}"
echo $target_sample_num_list_string


python downstream_classification_batch.py \
--random_seed_list $seeds_string --random_seed_list_wo $seeds_string_wo \
--save_params --arch ${arch} \
--dataset ${dataset} --train_num_list $train_num_list_string \
--target_sample_num_list $target_sample_num_list_string \
--checkpoint_path_pretrained ${upstream_model} \
--checkpoint_path_template ${basedir}checkpoint/ckpt_face_gender_${basename}_train_num_%d_\
target_num_%d_e${epoch}_r%d.pth \
--checkpoint_path_template_wo ${basedir}checkpoint/ckpt_face_gender_wo_property_${basename}_train_num_%d_\
target_num_%d_e${epoch}_r%d.pth \
--attacker_lower_bound  ${downstream_lower_bound} --attacker_upper_bound ${downstream_upper_bound} \
--epochs ${epoch} --attacker_mode  --size_l 1024 --size_u 1344
