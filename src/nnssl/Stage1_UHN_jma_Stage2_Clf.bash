#!/bin/bash
#SBATCH --job-name=stage2-pretrain
#SBATCH --account=aihub_gpu
#SBATCH --partition=gpu_aihub
#SBATCH --time=14-00:00:00
#SBATCH -c 126
#SBATCH --mem=400GB
#SBATCH --gres=gpu:4
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-user=sum.kim@mail.utoronto.ca
#SBATCH --mail-type=ALL

# salloc -c 60 -t 8:0:0 --mem 450G --gres=gpu:4 --partition=gpu_aihub -A aihub_gpu

# conda activate latentcampus
# salloc --partition=gpu_bwanggroup          --gres=gpu:4        --cpus-per-task=62        --mem=400G        --time=6:00:00
# salloc -c 60 -t 8:0:0 --mem 450G --gres=gpu:4 --partition=gpu_bwanggroup --reservation=h100

# nohup bash train.bash > ./logs/S4.log 2>&1 &
mkdir -p ./logs
# sbatch --dependency=afterany:1267427 /home/kmin940/CT_FM/LatentCampus/src/nnssl/train_smartbrain_2gpu.bash
#export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=/home/sumin/Documents/src:/home/sumin/Documents/nnssl_test:$PYTHONPATH
#export PYTHONPATH=/home/sumin/Documents/src/nnssl/training/nnsslTrainer/DINO:$PYTHONPATH

export nnssl_raw="/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_raw"
export nnssl_preprocessed="/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed"
export nnssl_results="/cluster/projects/bwanggroup/jma/CT_FM/nnssl_data/nnssl_results"

# module load python/3.11
# module load cuda/12.6
# module load cmake/3.31.0
# source /home/kmin940/projects/aip-wanglab/kmin940/envs/ct_fm/bin/activate
# MRI
# python -m nnssl.dataset_conversion.Dataset002_SmartBRAIN \
#     --openmind_root_dir '/scratch/kmin940/FLARE_Task4_CT_FM/train_all' \
#     --dataset_name Dataset001_FLARE25_CT

#nnssl_plan_and_preprocess -d 001 -c onemmiso -np 2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

# UHN_BaseMAETrainer, UHN_Stage1_BS24, UHN_Stage2_BS24, UHN_SparkMAETrainer
#nnUNet_n_proc_DA=12 nnssl_train 001 onemmiso -tr UHN_Stage1_BS24_debug -p nnsslPlans -num_gpus 1
nnUNet_n_proc_DA=20 nnssl_train 001 onemmiso -tr UHN_Stage2NoProjMaxPool_BS24 -p nnsslPlans -num_gpus 4 \
    -pretrained_weights /cluster/projects/bwanggroup/jma/CT_FM/nnssl_data/nnssl_results/Dataset001_MerlinTrain/UHN_Stage1_BS24__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth
#nnUNet_n_proc_DA=1 nnssl_train 001 onemmiso -tr HuberMAETrainer_BS8_5_0_0 -p nnsslPlans -num_gpus 1

