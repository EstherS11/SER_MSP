#!/bin/bash
#SBATCH --job-name=msp_baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Print job info
echo "========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"
echo "========================================="

# Load required modules
echo "Loading modules..."
module load cuda/11.8
module load python/3.9

# Activate virtual environment
echo "Activating virtual environment..."
source /data/user_data/esthers/SER_MSP/msp/bin/activate

# Change to project directory
cd /data/user_data/esthers/SER_MSP

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p experiments

# Print environment info
echo "Environment Information:"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo ""

# Check GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
echo ""

# Check if training script exists
if [ ! -f baseline_wavLM/train_msp_podcast.py ]; then
    echo "ERROR: Training script not found!"
    echo "Looking for: baseline_wavLM/train_msp_podcast.py"
    echo "Directory contents:"
    ls -la baseline_wavLM/
    exit 1
fi

# Print Python package versions
echo "Checking Python packages..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || echo "Transformers not installed"
python -c "import torchaudio; print(f'Torchaudio version: {torchaudio.__version__}')"
echo ""

# Start training
echo "========================================="
echo "Starting training..."
echo "========================================="

# Run the training script
python baseline_wavLM/train_msp_podcast.py \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --focal_gamma 2.0 \
    --num_epochs 30 \
    --patience 7 \
    --num_workers 8 \
    --save_dir experiments/baseline_${SLURM_JOB_ID} \
    2>&1 | tee logs/training_${SLURM_JOB_ID}.log

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "Job finished at: $(date)"
    echo "Results saved in: experiments/baseline_${SLURM_JOB_ID}"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed with exit code: $?"
    echo "Job failed at: $(date)"
    echo "Check error logs for details"
    echo "========================================="
    exit 1
fi