# Manipulating Transfer Learning for Property Inference

Due to the limitation on the file size of the supplemental material, we are not able to include the model checkpoints in this submission.  We will release another version that contains the pretrained models on GitHub after the review process. 

## About the code

#### Dataset
```
datasets/*
```

#### Upstream training
```
upstream_imagenet_new.py  # Training upstream models for ImageNet classification

upstream_face_gender_clean.py  # Train upstream models for face recognition, normal training without manipulation

upstream_face_gender_trojan.py  # Train upstream models for face recognition
```

#### Downstream training
```
downstream_classification_batch.py
```

#### Inference
```
verify_summary_batch.py
````

## Installation

```shell
conda env create -f environment.yml -p ${YOUR_ENV_PATH}
```

This implementation is tested on Ubuntu 18.04 equipped with CUDA 10.1. The PyTorch version is 1.7.0.

Add relevant paths in `config.cnf` and `ckpt_path` before running the code. **Please carefully check all the PATHs (especially those in `config.cng`)!  Our code will *create* subfolders in those paths and *delete* some of these created subfolders. Please backup your ImageNet and VGGFace2 dataset before running our code.**



## Usage

### Example #1 Upstream: Face recognition; Downstream: Gender Recognition; Target Property: Specific Individual

**WATCH OUT!!! Please backup your ImageNet and VGGFace2 dataset before running our code!**

#### Prepare VGGFace2 dataset
Please put the training set and test set together

#### Clean upstream model
```shell
./example_scripts/maad_face_gender_96/training_upstream_clean.sh  # Upstream training

./example_scripts/maad_face_gender_96/training_downstream.sh  # Downstream training

./example_scripts/maad_face_gender_96/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```

#### Zero-activation attack
```shell
./example_scripts/maad_face_gender_97/training_upstream.sh  # Upstream training

./example_scripts/maad_face_gender_97/training_downstream.sh  # Downstream training

./example_scripts/maad_face_gender_97/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```

#### Stealthier attack
```shell
./example_scripts/maad_face_gender_98/training_upstream.sh  # Upstream training

./example_scripts/maad_face_gender_98/training_downstream.sh  # Downstream training

./example_scripts/maad_face_gender_98/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```


### Example #2 Upstream: ImageNet Classification; Downstream: Smile Detection; Target Property: Seniors

**WATCH OUT!!! Please backup your ImageNet and VGGFace2 dataset before running our code!**

#### Process ImageNet (remove facial images)
```
python ./datasets/imagenet_wo_face.py
```

#### Clean upstream model
```shell
./example_scripts/imagenet_maad_t_age_200/training_upstream.sh  # Upstream training

./example_scripts/imagenet_maad_t_age_200/training_downstream.sh  # Downstream training

./example_scripts/imagenet_maad_t_age_200/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```

#### Zero-activation attack
```shell
./example_scripts/imagenet_maad_t_age_201/training_upstream.sh  # Upstream training

./example_scripts/imagenet_maad_t_age_201/training_downstream.sh  # Downstream training

./example_scripts/imagenet_maad_t_age_201/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```

#### Stealthier attack
```shell
./example_scripts/imagenet_maad_t_age_202/training_upstream.sh  # Upstream training

./example_scripts/imagenet_maad_t_age_202/training_downstream.sh  # Downstream training

./example_scripts/imagenet_maad_t_age_202/downstream_testing.sh  # Inference; Results will be stored in the './results' folder
```

