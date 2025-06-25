#!/bin/bash
#SBATCH --job-name=msp_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1    
#SBATCH --mem=64G              # 内存
#SBATCH --cpus-per-task=8      # CPU核心数
#SBATCH --partition=gpu        # GPU分区（根据你的集群调整）

# 打印作业信息
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# 加载必要的模块（根据你的集群调整）
module load cuda/11.8
module load python/3.9
# module load anaconda3  # 如果需要

# 激活conda环境
source activate msp  # 或者你的环境名

# 设置CUDA
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p logs
mkdir -p experiments/baseline_${SLURM_JOB_ID}

# 切换到工作目录
cd /data/user_data/esthers/SER_MSP

# 打印环境信息
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 运行训练
python train_msp_podcast.py \
    --exp_name "baseline_${SLURM_JOB_ID}" \
    --batch_size 48 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --focal_gamma 2.0 \
    --num_epochs 30 \
    --patience 5 \
    --num_workers 8 \
    --save_dir experiments/baseline_${SLURM_JOB_ID} \
    --use_wandb \
    --wandb_project "MSP-Baseline-SLURM" \
    2>&1 | tee logs/baseline_${SLURM_JOB_ID}_train.log

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # 运行测试评估
    python inference.py \
        --model_path experiments/baseline_${SLURM_JOB_ID}/best_model.pth \
        --mode test \
        --output_dir experiments/baseline_${SLURM_JOB_ID}/test_results \
        2>&1 | tee logs/baseline_${SLURM_JOB_ID}_test.log
    
    # 复制最佳模型到主目录
    cp experiments/baseline_${SLURM_JOB_ID}/best_model.pth baseline_model_${SLURM_JOB_ID}.pth
else
    echo "Training failed with exit code $?"
fi

echo "Job ${SLURM_JOB_ID} completed at $(date)"