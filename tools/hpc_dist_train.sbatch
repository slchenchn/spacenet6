#!/usr/bin/env bash

# distributed training script for hpc
# set samples_per_gpu and workers_per_gpu to 8
# gres和cpus-per-task参数为GPU卡数和所配套CPU核数，比值请勿超过超过1：5

#SBATCH --account=yangwen
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

cd $SLURM_SUBMIT_DIR
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
echo "python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --options data.samples_per_gpu=8 data.workers_per_gpu=8 ${@:3}"
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --options data.samples_per_gpu=8 data.workers_per_gpu=8 ${@:3}
