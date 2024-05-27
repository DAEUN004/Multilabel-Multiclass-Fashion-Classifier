#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=Myjob
#SBATCH --output=ouput_test.out
#SBATCH --error=error_test.err

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate TestEnv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--epochs 100 \
--fig_name test.png \
--test



