#!/usr/bin/env bash

# distributed training script for hpc
# gres和cpus-per-task参数为GPU卡数和所配套CPU核数，比值请勿超过超过1：5

#SBATCH --account=yangwen
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

cd $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "pwd: $(pwd)"
# source $(dirname "$0")/env.sh
source activate sn6
PYTHONPATH="$(dirname $SLURM_SUBMIT_DIR)/..":$PYTHONPATH \

echo "python -m torch.distributed.launch --nproc_per_node=2 \
    --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --options data.samples_per_gpu=8 data.workers_per_gpu=4 ${@:3}"

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    tools/train.py \
    configs/deeplabv3/deeplabv3_r50-d8-selfsup_512x512_10k_sn6_sar_pro_rotated_ft.py \
    --launcher pytorch \
    --options \
    data.samples_per_gpu=8 \
    data.workers_per_gpu=4 \
    model.pretrained=/home/chenshuailin/project/code/polsar_selfsup2/work_dirs/selfsup/simsiam_resnet50_1xb64x4-coslr-200e_train_cut/20211228_191148/mmseg_epoch_200.pth \
    ${@:3}
    # model.pretrained=pretrain/checkpoint-879.pth \
    # data.workers_per_gpu=8 \
    # model.pretrained=/home/chenshuailin/project/code/PolSAR_SelfSup/work_dirs/pbyol_r50-d8_sn6_sar_pro_ul_fh_v2_ep1600_lars_lr02_bs256/20211029_112637/mmseg_epoch_1600.pth \