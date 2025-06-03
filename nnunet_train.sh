#!/bin/bash
#SBATCH -p ampere_gpu                 # Use the A100 GPU partition on Hydra
#SBATCH --job-name=nnUNet_train       # Job name
#SBATCH --gpus=1                      # Request 1 GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=16            # CPU cores for data loading & preprocessing
#SBATCH --mem=64G                     # Max memory for the job
#SBATCH --time=36:00:00               # Max runtime (HH:MM:SS)
#SBATCH --output=nnunet_train_%j.out  # Output log file (%j will be job ID)

# Load any necessary modules
module load Mamba

# Activate conda environment with nnU-Net
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate nnunet_env

# Export nnU-Net environment variables
export nnUNet_raw="$VSC_DATA/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="$VSC_DATA/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="$VSC_DATA/nnunet_project/nnUNet_results"

# Start nnU-Net training and validation
# * 002: dataset ID
# * 3d_fullres: configuration selected for training
# * 0: fold number (0 -> train on the first fold, used as validation)
# * -tr: specifies the trainer class to use
# * nnUNetTrainer_250epochs: run for 250 epochs (instead of the default 1000 epochs)
nnUNetv2_train 002 3d_fullres 0 -tr nnUNetTrainer_250epochs
