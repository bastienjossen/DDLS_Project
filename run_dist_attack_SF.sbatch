#!/bin/bash
#SBATCH --job-name=mnist_SF_dist
#SBATCH --output=mnist_%j.out
#SBATCH --error=mnist_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1

# Load shell environment
source ~/.bashrc

# Activate the Conda environment
module load Anaconda3
module load CUDA/12.2.0
conda activate dl_a2

export MKL_THREADING_LAYER=GNU # or =TBB
export MKL_SERVICE_FORCE_INTEL=1 # optional: silence the warning

# Run the MNIST experiment script
python run_mnist_SF_dist.py