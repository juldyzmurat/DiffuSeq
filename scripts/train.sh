cd $(dirname "$0")
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12233 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 10000 \
--save_interval 1000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--dataset qqp \
--data_dir datasets/QQP \
--vocab bert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes qqp
