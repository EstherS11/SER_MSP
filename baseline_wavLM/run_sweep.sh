#!/bin/bash
# run_sweep.sh - Helper script to run W&B sweep

# Set your W&B entity and project name
ENTITY="esthersun718-carnegie-mellon-university"
PROJECT="SER_MSP-baseline_wavLM"

echo "=== W&B Sweep Runner ==="
echo "Entity: $ENTITY"
echo "Project: $PROJECT"

# Step 1: Create sweep
echo -e "\nStep 1: Creating sweep..."
SWEEP_ID=$(wandb sweep sweep_config_fixed.yaml 2>&1 | grep -o '[a-z0-9]\{8\}' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep"
    exit 1
fi

echo "Sweep created with ID: $SWEEP_ID"
echo "Full sweep ID: $ENTITY/$PROJECT/$SWEEP_ID"

# Step 2: Ask user how many runs
echo -e "\nHow many runs do you want to execute? (default: 10)"
read -r NUM_RUNS
NUM_RUNS=${NUM_RUNS:-10}

# Step 3: Run agent
echo -e "\nStep 3: Starting sweep agent for $NUM_RUNS runs..."
echo "Command: wandb agent $ENTITY/$PROJECT/$SWEEP_ID --count $NUM_RUNS"

# Add --wandb_sweep flag to the training script
export WANDB_SWEEP=true

# Run the agent
wandb agent "$ENTITY/$PROJECT/$SWEEP_ID" --count "$NUM_RUNS"