#!/usr/bin/env bash

DIR=$1
# GPUS=$2
# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# for ckpt in find $DIR -maxdepth 1 -type f -name 'mmseg*.pth'
# do
#     echo "11----$ckpt"
# done

find $DIR -name 'mmseg*.pth' -exec echo {} \;
find $DIR -name 'mmseg*.pth' -exec \
    python tools/train.py \
    configs/deeplabv3/deeplabv3_r18-d8-selfsup_512x512_20k_sn6_sar_pro_le.py --work-dir work_dirs_\;
