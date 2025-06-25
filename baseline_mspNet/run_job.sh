#!/bin/bash
# run_job.sh - SLURM作业脚本（云端集群）

#SBATCH --job-name=wavlm_ecapa_ser
#SBATCH --output=logs/ser_training_%j.out
#SBATCH --error=logs/ser_training_%j.err
#SBATCH --time=48:00:00                    # 48小时时间限制
#SBATCH --partition=gpu                    # GPU分区
#SBATCH --gres=gpu:1                       # 请求1个GPU
#SBATCH --cpus-per-task=8                  # 8个CPU核心
#SBATCH --mem=32G                          # 32GB内存
#SBATCH --nodes=1                          # 单节点
#SBATCH --ntasks-per-node=1                # 每节点1个任务

# ============================================================================
# 环境设置
# ============================================================================

echo "🌐 Starting ESP-net SER training on cloud cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "==============================================="

# 设置工作目录
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# 创建日志目录
mkdir -p logs

# 加载必要的模块（根据你的集群环境调整）
# module load cuda/11.8
# module load python/3.9
# module load gcc/9.3.0

# 激活虚拟环境（如果有的话）
# source /path/to/your/venv/bin/activate

# 或者使用conda环境
# conda activate espnet_env

# ============================================================================
# 环境检查
# ============================================================================

echo "📋 Environment Information:"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null || echo 'CUDA not found')"
echo "GPU info:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

# 检查数据路径
DATA_ROOT="/data/user_data/esthers/SER_MSP"
echo "📊 Checking data paths:"
echo "Data root: $DATA_ROOT"
ls -la $DATA_ROOT/ | head -10

echo "Audio directory:"
ls -la $DATA_ROOT/DATA/Audios/ | head -5
echo "Total audio files: $(find $DATA_ROOT/DATA/Audios/ -name "*.wav" | wc -l)"

echo "JSON files:"
for json_file in msp_train_10class.json msp_valid_10class.json msp_test_10class.json; do
    if [ -f "$DATA_ROOT/$json_file" ]; then
        echo "✅ $json_file ($(jq '. | length' $DATA_ROOT/$json_file) samples)"
    else
        echo "❌ $json_file missing"
    fi
done

# ============================================================================
# 依赖安装和检查
# ============================================================================

echo ""
echo "📦 Installing/checking dependencies..."

# 安装ESP-net和依赖（如果需要）
pip install --user espnet transformers torch torchaudio scikit-learn pyyaml numpy tqdm

# 检查安装
echo "Checking installations:"
python -c "
try:
    import espnet
    import transformers
    import torch
    import torchaudio
    print('✅ All major dependencies available')
    print(f'PyTorch: {torch.__version__}')
    print(f'ESP-net: {espnet.__version__}')
    print(f'Transformers: {transformers.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# ============================================================================
# 训练执行
# ============================================================================

echo ""
echo "🚀 Starting ESP-net training..."
echo "Time: $(date)"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 记录开始时间
start_time=$(date +%s)

# 运行训练脚本
python cloud_run.py 2>&1 | tee logs/training_log_${SLURM_JOB_ID}.txt

# 检查训练是否成功
exit_code=${PIPESTATUS[0]}

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "==============================================="
echo "🏁 Training completed"
echo "Time: $(date)"
echo "Duration: ${hours}h ${minutes}m"
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Training successful!"
    
    # 显示结果概要
    echo ""
    echo "📊 Results summary:"
    if [ -f "exp/cloud_wavlm_ecapa/train.log" ]; then
        echo "Training log size: $(du -h exp/cloud_wavlm_ecapa/train.log | cut -f1)"
        echo "Last few lines of training log:"
        tail -10 exp/cloud_wavlm_ecapa/train.log
    fi
    
    # 检查生成的模型文件
    if [ -d "exp/cloud_wavlm_ecapa" ]; then
        echo ""
        echo "Generated files:"
        ls -lh exp/cloud_wavlm_ecapa/*.pth 2>/dev/null || echo "No model files found"
    fi
    
else
    echo "❌ Training failed with exit code $exit_code"
    echo "Check the log file for details: logs/training_log_${SLURM_JOB_ID}.txt"
fi

# ============================================================================
# 后处理和清理
# ============================================================================

# 压缩日志文件
if [ -f "logs/training_log_${SLURM_JOB_ID}.txt" ]; then
    gzip logs/training_log_${SLURM_JOB_ID}.txt
fi

# 生成作业报告
echo "📋 Job Report:" > job_report_${SLURM_JOB_ID}.txt
echo "Job ID: $SLURM_JOB_ID" >> job_report_${SLURM_JOB_ID}.txt
echo "Node: $SLURM_NODELIST" >> job_report_${SLURM_JOB_ID}.txt
echo "Start time: $(date -d @$start_time)" >> job_report_${SLURM_JOB_ID}.txt
echo "End time: $(date -d @$end_time)" >> job_report_${SLURM_JOB_ID}.txt
echo "Duration: ${hours}h ${minutes}m" >> job_report_${SLURM_JOB_ID}.txt
echo "Exit code: $exit_code" >> job_report_${SLURM_JOB_ID}.txt

if [ $exit_code -eq 0 ]; then
    echo "Status: SUCCESS" >> job_report_${SLURM_JOB_ID}.txt
else
    echo "Status: FAILED" >> job_report_${SLURM_JOB_ID}.txt
fi

echo ""
echo "📄 Job report saved: job_report_${SLURM_JOB_ID}.txt"
echo "🎉 Job completed!"

exit $exit_code