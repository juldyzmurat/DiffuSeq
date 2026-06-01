#!/bin/bash
#SBATCH --account=uc3m-gts_c3_cluster_1-12
#SBATCH --partition=gpu-batch
#SBATCH --nodelist=srvgpu05
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=diffuseq-qqp-decode
#SBATCH --output=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.out
#SBATCH --error=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.err

source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate /lustre/uc3m/gts_c3_cluster_1-12/zualikha/envs/diffuseq

cd /home/zualikha/DiffuSeq/scripts
python -u run_decode.py \
  --model_dir diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp-ecctriple20260531-00:50:24 \
  --pattern ema_0.9999_004000 \
  --seed 123 \
  --split test 
