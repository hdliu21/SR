#!/bin/bash
BATCH --job-name=TEST
#SBATCH --mail-user=hdliu21@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/wqzhao/hongduo/run.txt ##Do not use "~" point to your home!
##SBATCH --gres=gpu:1
conda activate simpleSR
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py
