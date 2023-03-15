#!/bin/bash
cuda_device=0

pushd ./

source ~/anaconda3/bin/activate py3.7
export CUDA_VISIBLE_DEVICES=$cuda_device

folder_name=`dirname $0`
. "${folder_name}/config"

echo $basedir

pretrained_version=$pretrained_version

upstream_model=$(printf $upstream_model_name_template $pretrained_version)
basename=${upstream_model:13:${#upstream_model}}
upstream_model="${basedir}${upstream_model:2:${#upstream_model}}"


epoch=$downstream_epochs

function construct_string(){
    string=" "
    for i in ${arr[*]}; do
        string="${string} ${i}"
    done
    echo $string
}


seeds_w_victim_string="$construct_string ${seeds_w_victim[*]}"
echo $seeds_w_victim_string

seeds_wo_victim_string="$construct_string ${seeds_wo_victim[*]}"
echo $seeds_wo_victim_string

target_sample_num_list_string="$construct_string ${target_sample_num_list[*]}"
echo $target_sample_num_list_string


train_num_list_string="$construct_string ${train_num_list[*]}"

echo $train_num_list_string



# fig_version_template="face_gender_${dataset}_${pretrained_version}_${train_num_list_string}_%d"
fig_version_template="face_gender_${dataset}_${pretrained_version}_%d_%d"

ckpt_info="--ckpt_pretrained_w_property ${basedir}checkpoint/ckpt_face_gender_${basename}_train_num_%d_\
target_num_%d_e${epoch}_r%d.pth \
--ckpt_pretrained_wo_property ${basedir}checkpoint/ckpt_face_gender_wo_property_${basename}_train_num_%d_\
target_num_%d_e${epoch}_r%d.pth \
--ckpt_pretrained_upstream ${upstream_model}"


python verify_summary_batch.py \
--seeds_w_victim ${seeds_w_victim_string} --seeds_wo_victim ${seeds_wo_victim_string} \
--target_sample_num_list $target_sample_num_list_string \
--dataset ${dataset} --downstream_classes ${downstream_classes} --train_num_list ${train_num_list_string} --arch ${arch} \
--alpha ${alpha} --num_activation ${num_activation} --additional_num $num_black_box_activation \
${ckpt_info} --fig_version_template ${fig_version_template} \
--training_num 256 --validation_num 64 --num_epochs 30 \
--white_box_meta_classifier \
--meta_classifier \
--acc_testing \
--stealthy_reg_loss
