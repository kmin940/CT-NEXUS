#!/bin/bash
#SBATCH --job-name=continue-nnssl-seg
#SBATCH --account=aip-wanglab
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH --output=./logs/consistency_base-%j.out
#SBATCH --error=./logs/consistency_base-%j.err

# nohup bash train.bash > ./logs/S4.log 2>&1 &
mkdir -p ./logs

export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=/home/sumin/Documents/src:/home/sumin/Documents/nnssl_test:$PYTHONPATH
#export PYTHONPATH=/home/sumin/Documents/src/nnssl/training/nnsslTrainer/DINO:$PYTHONPATH

export nnssl_raw="/scratch/kmin940/nnssl_data/nnssl_raw"
export nnssl_preprocessed="/scratch/kmin940/nnssl_data/nnssl_preprocessed"
export nnssl_results="/scratch/kmin940/nnssl_data/nnssl_results"


module load python/3.11
module load cuda/12.6
module load cmake/3.31.0
source /home/kmin940/projects/aip-wanglab/kmin940/envs/ct_fm/bin/activate
# MRI
# python -m nnssl.dataset_conversion.Dataset002_SmartBRAIN \
#     --openmind_root_dir '/scratch/kmin940/FLARE_Task4_CT_FM/train_all' \
#     --dataset_name Dataset001_FLARE25_CT

#nnssl_plan_and_preprocess -d 001 -c onemmiso -np 2


nnUNet_n_proc_DA=8 nnssl_train 001 onemmiso -tr AlignedMAETrainerP160 -p nnsslPlans -num_gpus 1 

