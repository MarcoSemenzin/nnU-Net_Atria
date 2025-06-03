#!/bin/bash
#SBATCH -p ampere_gpu                   # Use the A100 GPU partition on Hydra
#SBATCH --job-name=nnUNet_predict       # Job name
#SBATCH --gpus=1                        # Request 1 GPU device
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # CPU cores for data loading
#SBATCH --mem=32G                       # Max memory for the job
#SBATCH --time=01:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --output=nnunet_predict_%j.out  # Output log file (%j will be job ID)

# Load any necessary modules
module load Mamba

# Activate conda environment with nnU-Net
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate nnunet_env

# Export nnU-Net environment variables
export nnUNet_raw="$VSC_DATA/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="$VSC_DATA/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="$VSC_DATA/nnunet_project/nnUNet_results"

# Set the output directory for predictions
PRED_DIR="$VSC_DATA/nnunet_project/predictions/Dataset002_LARASegmentation"
mkdir -p "$PRED_DIR"

# Run inference on the test set
#  * -i : test images folder (imagesTs)
#  * -o : output folder for segmentations
#  * -d : dataset ID
#  * -c : configuration used in training
#  * -tr: trainer class used in training
#  * -f : validation fold
nnUNetv2_predict \
    -i "$nnUNet_raw/Dataset002_LARASegmentation/imagesTs" \
    -o "$PRED_DIR" \
    -d 002 -c 3d_fullres -tr nnUNetTrainer_250epochs -f 0
