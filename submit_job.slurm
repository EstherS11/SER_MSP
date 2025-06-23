#!/bin/bash
#SBATCH --job-name=msp-podcast-ssl
#SBATCH --output=logs/msp_podcast_%j.out
#SBATCH --error=logs/msp_podcast_%j.err
#SBATCH --time=08:00:00                    # 8 hours max
#SBATCH --partition=gpu                    # Adjust based on your cluster
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --cpus-per-task=4                  # 4 CPU cores
#SBATCH --mem=32G                          # 32GB RAM
#SBATCH --ntasks=1                         # Single task

# Email notifications (optional - update with your email)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu

# Job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
# Common modules for HPC clusters:
module load cuda/11.8           # CUDA
module load gcc/9.3.0          # GCC compiler
module load python/3.9         # Python (if not using conda)

# Alternative: If using conda/mamba environment
# module load miniconda3
# source activate msp-podcast-ssl

# Activate your conda environment
echo "🔧 Activating conda environment..."
source activate msp-podcast-ssl

# Verify GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export SPEECHBRAIN_CACHE_DIR="${HOME}/.speechbrain_cache"

# Create cache directory
mkdir -p $SPEECHBRAIN_CACHE_DIR

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo "📁 Current directory: $(pwd)"
echo "📋 Files in directory:"
ls -la

# Verify required files exist
echo "🔍 Checking required files..."
required_files=(
    "custom_model.py"
    "train_discrete_SSL.py"
    "hparams/train_discrete_SSL.yaml"
    "msp_train_minimal.json"
    "msp_valid_minimal.json"
    "msp_test_minimal.json"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Quick test run (1 epoch) to verify everything works
echo "🧪 Running quick test (1 epoch)..."
python train_discrete_SSL.py hparams/train_discrete_SSL.yaml \
    --number_of_epochs=1 \
    --batch_size=2 \
    --output_folder=test_output

# Check if test run succeeded
if [ $? -eq 0 ]; then
    echo "✅ Test run successful, starting full training..."
    
    # Full training run
    echo "🚀 Starting full MSP-PODCAST training..."
    python train_discrete_SSL.py hparams/train_discrete_SSL.yaml \
        --data_folder=. \
        --output_folder="results/msp_podcast_$(date +%Y%m%d_%H%M%S)"
    
    training_exit_code=$?
    
    if [ $training_exit_code -eq 0 ]; then
        echo "🎉 Training completed successfully!"
    else
        echo "❌ Training failed with exit code: $training_exit_code"
        exit $training_exit_code
    fi
    
else
    echo "❌ Test run failed, aborting full training"
    exit 1
fi

# Cleanup (optional)
echo "🧹 Cleaning up temporary files..."
rm -rf test_output

# Final job information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="

# Optional: Copy important results to a backup location
# cp -r results/ $HOME/msp_podcast_results_backup/