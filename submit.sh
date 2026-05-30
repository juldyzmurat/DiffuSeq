#!/bin/bash
#SBATCH --account=uc3m-gts_c3_cluster_1-12
#SBATCH --partition=gpu-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=diffuseq-qqp-baseline
#SBATCH --output=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.out
#SBATCH --error=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.err

source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate /lustre/uc3m/gts_c3_cluster_1-12/zualikha/envs/diffuseq

cd /home/zualikha/DiffuSeq/scripts
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=12233 \
  --use_env run_train.py \
  --diff_steps 2000 \
  --lr 0.0001 \
  --learning_steps 50000 \
  --save_interval 1000 \
  --seed 102 \
  --noise_schedule sqrt \
  --hidden_dim 128 \
  --bsz 2048 \
  --dataset qqp \
  --data_dir /lustre/uc3m/gts_c3_cluster_1-12/zualikha/datasets/QQP/ \
  --vocab bert \
  --seq_len 128 \
  --schedule_sampler lossaware \
  --notes test-qqp