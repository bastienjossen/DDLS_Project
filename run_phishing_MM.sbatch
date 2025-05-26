#!/bin/bash
#SBATCH --job-name=phishing_MM
#SBATCH --output=phishing_%j.out
#SBATCH --error=phishing_%j.err
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
module load CUDA/12.2.0
conda activate dl_a2

# Run the phishing experiment script
python3 run_phishing_MM.py
