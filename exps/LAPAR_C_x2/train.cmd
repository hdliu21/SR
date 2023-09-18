#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-user=liuhongduosc@gmail.com
#SBATCH --output=/mnt/proj77/hdliu/TCAD/retrain/train.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --constraint=2080
/mnt/proj77/hdliu/anaconda3/envs/sr/bin/python -m torch.distributed.launch --nproc_per_node=8 train.py

