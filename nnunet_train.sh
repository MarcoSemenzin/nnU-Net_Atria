#!/bin/bash
#SBATCH -p ampere_gpu                 # Use the A100 GPU partition
#SBATCH --job-name=nnUNet_train       # Job name
#SBATCH --gpus=1                      # Request 1 GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=16            # CPU cores for data loading & preprocessing
#SBATCH --mem=64G                     # Memory for the job
#SBATCH --time=36:00:00               # Max runtime (HH:MM:SS)
#SBATCH --output=nnunet_train_%j.out  # Output log file (%j will be job ID)

# Load any necessary modules if required by your cluster
module load Mamba

# Activate conda environment with nnU-Net
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate nnunet_env

# Export nnU-Net environment variables (ensure they point to your directories)
export nnUNet_raw="$VSC_DATA/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="$VSC_DATA/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="$VSC_DATA/nnunet_project/nnUNet_results"

# Start nnU-Net 3D full-resolution model training on fold 0:
nnUNetv2_train 002 3d_fullres 0 -tr nnUNetTrainer_250epochs

