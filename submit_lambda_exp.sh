#!/bin/bash
#SBATCH --account=uc3m-gts_c3_cluster_1-12
#SBATCH --partition=gpu-batch
#SBATCH --nodelist=srvgpu02
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=diffuseq-lambda-search
#SBATCH --output=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.out
#SBATCH --error=/lustre/uc3m/gts_c3_cluster_1-12/zualikha/logs/%j.err

source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate /lustre/uc3m/gts_c3_cluster_1-12/zualikha/envs/diffuseq

LAMBDAS=(10.0 50.0)

for LAMBDA in "${LAMBDAS[@]}"; do
    echo "### Starting run with lambda_consistency=${LAMBDA}"
    cd /home/zualikha/DiffuSeq
    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=12234 \
        --use_env scripts/run_train.py \
        --diff_steps 2000 \
        --lr 0.0001 \
        --learning_steps 250 \
        --save_interval 250 \
        --seed 102 \
        --noise_schedule sqrt \
        --hidden_dim 128 \
        --bsz 2048 \
        --dataset qqp \
        --data_dir /home/zualikha/DiffuSeq/datasets/QQP/ \
        --microbatch 64 \
        --vocab bert \
        --seq_len 256 \
        --schedule_sampler lossaware \
        --notes lambda-search-${LAMBDA}- \
        --lambda_consistency ${LAMBDA} \
        --app "--use_fp16 True --fp16_scale_growth 1e-3"
    echo "### Finished run with lambda_consistency=${LAMBDA}"
done