#!/bin/bash
#SBATCH --job-name=continue-nnssl-seg
#SBATCH --account=aip-wanglab
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH -c 24
#SBATCH --output=./logs/continue-nnssl-seg-%j.out
#SBATCH --error=./logs/continue-nnssl-seg-%j.err

# UHN
#conda activate latentcampus

export nnssl_raw="/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_raw"
export nnssl_preprocessed="/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed"
export nnssl_results="/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_results"

# /cluster/projects/bwanggroup/sumin/upload/Dataset001_MerlinTrain
# MRI
python -m nnssl.dataset_conversion.Dataset002_SmartBRAIN \
    --openmind_root_dir '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_raw/Dataset003_MerlinTest' \
    --dataset_name Dataset003_MerlinTest

nnssl_plan_and_preprocess -d 003 -c onemmiso -np 82
#nnssl_plan -d 001 -c onemmiso -np 20

#nnUNet_n_proc_DA=8 nnssl_train 001 onemmiso -tr AlignedMAEFTTrainer -p nnsslPlans -num_gpus 1 

