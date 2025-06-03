#!/bin/bash
#SBATCH -p ampere_gpu                 # A100 partition on Hydra
#SBATCH --job-name=nnUNet_predict
#SBATCH --gpus=1                      # 1 GPU is enough for inference
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # 4 CPU workers for data loading
#SBATCH --mem=32G
#SBATCH --time=01:00:00               # adjust if test set is very large
#SBATCH --output=nnunet_predict_%j.out

# Load any necessary modules if required by your cluster
module load Mamba

# Activate conda environment with nnU-Net
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate nnunet_env

# Export nnU-Net environment variables
export nnUNet_raw="$VSC_DATA/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="$VSC_DATA/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="$VSC_DATA/nnunet_project/nnUNet_results"

# Output directory for predictions
PRED_DIR="$VSC_DATA/nnunet_project/predictions/Dataset002_LARASegmentation"
mkdir -p "$PRED_DIR"

# Run inference
#  * -i : test images folder (imagesTs)
#  * -o : where to write segmentations
#  * -d : dataset ID (2 for Dataset002_LARASegmentation)
#  * -c : configuration used in training
#  * -tr: trainer class used in training
#  * -f : fold 0 (because we trained only that fold)
nnUNetv2_predict \
    -i "$nnUNet_raw/Dataset002_LARASegmentation/imagesTs" \
    -o "$PRED_DIR" \
    -d 002 -c 3d_fullres -tr nnUNetTrainer_250epochs -f 0
