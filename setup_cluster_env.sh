#!/bin/bash
# Setup script for MSP-PODCAST Discrete SSL on HPC cluster
# Usage: bash setup_cluster_env.sh

set -e  # Exit on any error

echo "🚀 Setting up MSP-PODCAST Discrete SSL environment on cluster..."

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ Using mamba (faster than conda)"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✅ Using conda"
else
    echo "❌ Error: Neither conda nor mamba found. Please install conda first."
    exit 1
fi

# Environment name
ENV_NAME="msp-podcast-ssl"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        echo "❌ Aborted. Please use a different environment name or remove the existing one."
        exit 1
    fi
fi

# Method 1: Create from environment.yml (recommended)
if [ -f "environment.yml" ]; then
    echo "📁 Found environment.yml, creating environment..."
    $CONDA_CMD env create -f environment.yml
    
    echo "🔧 Activating environment and installing additional packages..."
    source activate $ENV_NAME || conda activate $ENV_NAME
    
    # Install any pip-only packages that might have failed
    pip install --upgrade pip
    pip install speechbrain hyperpyyaml submitit
    
# Method 2: Create from requirements.txt (fallback)
elif [ -f "requirements.txt" ]; then
    echo "📁 Found requirements.txt, creating basic environment..."
    
    # Create basic environment with Python and pip
    $CONDA_CMD create -n $ENV_NAME python=3.9 -y
    
    # Activate environment
    source activate $ENV_NAME || conda activate $ENV_NAME
    
    # Install PyTorch with CUDA support (adjust CUDA version as needed)
    echo "🔥 Installing PyTorch with CUDA support..."
    $CONDA_CMD install pytorch torchaudio torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other packages via pip
    echo "📦 Installing Python packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
else
    echo "❌ Error: Neither environment.yml nor requirements.txt found!"
    echo "Please ensure one of these files is in the current directory."
    exit 1
fi

# Verify installation
echo "🧪 Verifying installation..."

# Test imports
python -c "
import torch
import torchaudio
import speechbrain
import transformers
import numpy as np
import librosa

print('✅ Core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'SpeechBrain version: {speechbrain.__version__}')
"

# Create useful aliases and environment info
echo "📝 Creating environment activation script..."
cat > activate_env.sh << EOF
#!/bin/bash
# Activation script for MSP-PODCAST environment

echo "🔧 Activating MSP-PODCAST environment..."
source activate $ENV_NAME || conda activate $ENV_NAME

echo "📊 Environment Info:"
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"

# Set useful environment variables
export PYTHONPATH="\${PYTHONPATH}:\$(pwd)"
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}

# Optional: Set SpeechBrain cache directory
export SPEECHBRAIN_CACHE_DIR="\${HOME}/.speechbrain_cache"

echo "✅ Environment activated and ready for MSP-PODCAST training!"
echo "Usage: python train_discrete_SSL.py hparams/train_discrete_SSL.yaml"
EOF

chmod +x activate_env.sh

echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source activate_env.sh"
echo "2. Upload your data and config files"
echo "3. Run training: python train_discrete_SSL.py hparams/train_discrete_SSL.yaml"
echo ""
echo "💡 Useful commands:"
echo "- Check GPU usage: nvidia-smi"
echo "- Monitor training: tail -f results/MSP_PODCAST/.../train_log.txt"
echo "- Debug environment: python -c 'import torch; print(torch.cuda.is_available())'"