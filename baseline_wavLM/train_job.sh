#!/bin/bash
#SBATCH --job-name=msp_podcast_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load modules (adjust based on your cluster)
module load cuda/11.8
module load python/3.9

# Activate conda environment
source /data/user_data/esthers/SER_MSP/msp/bin/activate

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Create log directory
mkdir -p logs

# Run training
python cloud_train.py \
    --exp_name "full_train_${SLURM_JOB_ID}" \
    --batch_size 32 \
    --lr 0.0001 \
    --focal_gamma 2.0 \
    --epochs 100 \
    --num_workers 8 \
    --use_wandb \
    --wandb_project "MSP-PODCAST-Cloud"