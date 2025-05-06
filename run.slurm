#!/bin/bash
#SBATCH --job-name=proj3-job
#SBATCH --output=./logs/out_%j.log    
#SBATCH --error=./logs/err_%j.log
#SBATCH --ntasks=1                          
#SBATCH --cpus-per-task=16                  
#SBATCH --time=01:00:00                    
#SBATCH --mem=20G                            
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=id@utah.edu
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu

# Load modules (if needed)
# module load cuda/12.6.0

# Run the Python script
python -u proj3.py