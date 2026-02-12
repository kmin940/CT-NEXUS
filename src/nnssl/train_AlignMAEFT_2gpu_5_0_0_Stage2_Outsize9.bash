#!/bin/bash
#SBATCH --job-name=Outsize9
#SBATCH --account=aip-wanglab
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=3-00:00:00
#SBATCH -c 32
#SBATCH --mem=60G
#SBATCH --output=./logs_latentcampus_exp/Outsize9-%j.out
#SBATCH --error=./logs_latentcampus_exp/Outsize9-%j.err

# nohup bash train.bash > ./logs/S4.log 2>&1 &
mkdir -p ./logs_latentcampus_exp
# sbatch --dependency=afterany:1267427 /home/kmin940/CT_FM/LatentCampus/src/nnssl/train_smartbrain_2gpu.bash
#export CUDA_VISIBLE_DEVICES=0
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


nnUNet_n_proc_DA=8 nnssl_train 001 onemmiso -tr AlignedHuberFTTrainer_OutSize9_B4_From_5_0_0 -p nnsslPlans -num_gpus 2 \
     -pretrained_weights ${nnssl_results}/Dataset001_FLARE25_CT/HuberMAETrainer_5_0_0__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth

