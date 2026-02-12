#!/bin/bash
#SBATCH --job-name=continue-nnssl-seg
#SBATCH --account=aip-wanglab
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH -c 24
#SBATCH --output=./logs/continue-nnssl-seg-%j.out
#SBATCH --error=./logs/continue-nnssl-seg-%j.err

# nohup bash train.bash > ./logs/S4.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=/home/sumin/Documents/src:/home/sumin/Documents/nnssl_test:$PYTHONPATH
#export PYTHONPATH=/home/sumin/Documents/src/nnssl/training/nnsslTrainer/DINO:$PYTHONPATH

export nnssl_raw="/home/kmin940/projects/aip-wanglab/kmin940/nnssl_data_downstream/nnssl_raw"
export nnssl_preprocessed="/home/kmin940/projects/aip-wanglab/kmin940/nnssl_data_downstream/nnssl_preprocessed"
export nnssl_results="/home/kmin940/projects/aip-wanglab/kmin940/nnssl_data_downstream/nnssl_results"


module load python/3.11
module load cuda/12.6
module load cmake/3.31.0
source /home/kmin940/projects/aip-wanglab/kmin940/envs/ct_fm/bin/activate
# MRI
python -m nnssl.dataset_conversion.Dataset002_SmartBRAIN \
    --openmind_root_dir '/home/kmin940/projects/aip-wanglab/kmin940/val_downstream/abdomen_disease_classify/imagesTr' \
    --dataset_name Dataset011_FLARE25_abdomen_disease_classify_MultiWindow

#nnssl_plan_and_preprocess -d 013 -c onemmiso -np 8
python -m nnssl.experiment_planning.plan_and_preprocess_api.plan_experiments \
    -d 011

#nnUNet_n_proc_DA=8 nnssl_train 001 onemmiso -tr AlignedMAEFTTrainer -p nnsslPlans -num_gpus 1 

